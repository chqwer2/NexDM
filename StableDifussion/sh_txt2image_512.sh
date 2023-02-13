python scripts/txt2img.py \
        --prompt "a professional photograph of an astronaut riding a horse" \
        --ckpt /apdcephfs/share_1330077/terryqchen/Datasets/stablediffusion/512-base-ema.ckpt \
        --config configs/stable-diffusion/v2-inference-v.yaml \
        --outdir /apdcephfs/share_1330077/terryqchen/Experiments/stablediffusion/outputs/txt2img-samples \
        --H 512 --W 512