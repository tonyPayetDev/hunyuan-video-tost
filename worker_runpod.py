import os, json, requests, random, time, runpod

import torch
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

with torch.inference_mode():
    args = parse_args()
    args.flow_reverse = True
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained("/content/HunyuanVideo/ckpts", args=args)

@torch.inference_mode()
def generate(input):
    values = input["input"]

    positive_prompt = values['positive_prompt']
    height = values['height']
    width = values['width']
    video_length = values['video_length']
    seed = values['seed']
    negative_prompt = values['negative_prompt']
    infer_steps = values['infer_steps']
    guidance_scale = values['guidance_scale']
    num_videos_per_prompt = values['num_videos_per_prompt']
    flow_shift = values['flow_shift']
    batch_size = values['batch_size']
    embedded_guidance_scale = values['embedded_guidance_scale']

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)

    outputs = hunyuan_video_sampler.predict(
        prompt=positive_prompt, 
        height=height,
        width=width,
        video_length=video_length,
        seed=seed,
        negative_prompt=negative_prompt,
        infer_steps=infer_steps,
        guidance_scale=guidance_scale,
        num_videos_per_prompt=num_videos_per_prompt,
        flow_shift=flow_shift,
        batch_size=batch_size,
        embedded_guidance_scale=embedded_guidance_scale
    )
    samples = outputs['samples']
    sample = samples[0].unsqueeze(0)
    save_videos_grid(sample, f"/content/hunyuan-video-{seed}-tost.mp4", fps=24)

    result = f"/content/hunyuan-video-{seed}-tost.mp4"
    video_path = f"/content/hunyuan-video-{seed}-tost.mp4"
    save_videos_grid(sample, video_path, fps=24)

    try:
        # Convertir la vidéo en base64
        with open(video_path, "rb") as file:
            video_base64 = base64.b64encode(file.read()).decode("utf-8")

        return {"video_base64": video_base64, "status": "DONE"}

    except Exception as e:
        return {"error": str(e), "status": "FAILED"}

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

runpod.serverless.start({"handler": generate})
