import sys
import argparse, os
import random
import cv2
import torch
import numpy as np
# import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from imwatermark import WatermarkEncoder
from pathlib import Path

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

torch.set_grad_enabled(False)


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    with torch.no_grad(), \
            torch.autocast("cuda"):
        batch = make_batch_sd(image, mask, txt=prompt,
                              device=device, num_samples=num_samples)

        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(
                    model.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels, h // 8, w // 8]
        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=1.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
        )
        x_samples_ddim = model.decode_first_stage(samples_cfg)

        result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                             min=0.0, max=1.0)

        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

def predict(input_image, prompt, ddim_steps, num_samples, scale, seed):
    init_image = input_image["image"].convert("RGB")
    init_mask = input_image["mask"].convert("RGB")
    image = pad_image(init_image) # resize to integer multiple of 32
    mask = pad_image(init_mask) # resize to integer multiple of 32
    width, height = image.size
    print("Inpainting...", width, height)

    result = inpaint(
        sampler=sampler,
        image=image,
        mask=mask,
        prompt=prompt,
        seed=seed,
        scale=scale,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        h=height, w=width
    )

    return result

def load_img(path):
    image = Image.open(path).convert("RGB")
    return image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=str,
        # nargs="?",
        help="dir to write results to",
        # default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        # nargs="?",
        help="dir to write results to",
        # default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of an astronaut riding a triceratops",
        help="the prompt to render"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="num_samples",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=45,
        help="ddim_steps",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(0, 2147483647),
        help="randomseed",
    )
    opt = parser.parse_args()
    return opt


'''
sampler = initialize_model(sys.argv[1], sys.argv[2])

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Stable Diffusion Inpainting")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', tool='sketch', type="pil")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(
                    label="Images", minimum=1, maximum=4, value=4, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1,
                                       maximum=50, value=45, step=1)
                scale = gr.Slider(
                    label="Guidance Scale", minimum=0.1, maximum=30.0, value=10, step=0.1
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=2147483647,
                    step=1,
                    randomize=True,
                )
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(
                grid=[2], height="auto")

    run_button.click(fn=predict, inputs=[
                     input_image, prompt, ddim_steps, num_samples, scale, seed], outputs=[gallery])


# block.launch()
# block.launch(share=True)
'''


if __name__ == "__main__":
    opt = parse_args()
    
    sampler = initialize_model(opt.config, opt.ckpt)
    # input_image = gr.Image(source='upload', tool='sketch', type="pil")
    image = load_img(opt.img_path)
    mask = load_img(opt.mask_path)
    input_image = {
        "image": image,
        "mask": mask
    }
    # prompt = gr.Textbox(label="Prompt")
    # run_button = gr.Button(label="Run")
    # num_samples = gr.Slider(
    #     label="Images", minimum=1, maximum=4, value=4, step=1)
    # ddim_steps = gr.Slider(label="Steps", minimum=1,
    #                         maximum=50, value=45, step=1)
    
    # scale = gr.Slider(
    #     label="Guidance Scale", minimum=0.1, maximum=30.0, value=10, step=0.1
    # )
    # seed = gr.Slider(
    #     label="Seed",
    #     minimum=0,
    #     maximum=2147483647,
    #     step=1,
    #     randomize=True,
    # )
    
    prompt = opt.prompt
    num_samples = opt.num_samples
    ddim_steps = opt.ddim_steps
    scale = opt.scale
    seed = opt.seed
    
    # gallery = gr.Gallery(label="Generated images", show_label=False).style(
    #         grid=[2], height="auto")

    
    gallery_images = predict(input_image, prompt, ddim_steps, num_samples, scale, seed)
    print('generated done')
    print('len of gallery', len(gallery_images))
    for i, gal in enumerate(gallery_images):
        print(gal.size)
        print(gal)
        
        gal.save(f"/apdcephfs/share_1330077/terryqchen/Experiments/stablediffusion/inpainting/output/giants_inpainting_gallery_mask_v2_{i}.png")
    # print(gallery_images.size)