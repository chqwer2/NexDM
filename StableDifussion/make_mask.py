
from PIL import Image
import argparse
import random
import numpy as np

def load_img(path):
    image = Image.open(path)
    print(np.array(image).max())
    image = image.convert("RGB")
    print(np.array(image).max())
    return image

def makemask(input_image):
    input_mask = np.zeros_like(np.array(input_image))
    h,w = input_image.size[0], input_image.size[1]
    
    print(h, w)
    input_mask[2*w//10 : 6*w//10, 4*h//10 : 8*h//10] = 255
    
    input_mask = Image.fromarray(input_mask)
    input_mask.save('/apdcephfs/share_1330077/terryqchen/Experiments/stablediffusion/inpainting/input/food_mask_v2.png')
    
    
    return input_mask
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=str,
        # nargs="?",
        help="dir to write results to",
        default="/apdcephfs/share_1330077/terryqchen/Experiments/stablediffusion/inpainting/input/food_frame_0119.png"
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        # nargs="?",
        help="dir to write results to",
        # default="outputs/txt2img-samples"
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_args()
    
    # input_image = gr.Image(source='upload', tool='sketch', type="pil")
    image = load_img(opt.img_path)
    print(image.size)
    print(np.array(image).shape)
    mask = makemask(image)
    print(mask.size)
    # mask = load_img(opt.mask_path)
    
    # input_image = {
    #     "image": image,
    #     "mask": mask
    # }