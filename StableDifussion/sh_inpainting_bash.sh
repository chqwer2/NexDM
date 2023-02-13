python ./sh_inpainting.py \
    --config configs/stable-diffusion/v2-inpainting-inference.yaml \
    --ckpt /apdcephfs/share_1330077/terryqchen/Datasets/stablediffusion/512-inpainting-ema.ckpt \
    --img_path "/apdcephfs/share_1330077/terryqchen/Experiments/stablediffusion/inpainting/input/giants_frame_0119.png" \
    --mask_path "/apdcephfs/share_1330077/terryqchen/Experiments/stablediffusion/inpainting/input/food_mask_v2.png" \
    --prompt "" \
    --num_samples 8 \
    --ddim_steps 500 \
    --scale 10.0
