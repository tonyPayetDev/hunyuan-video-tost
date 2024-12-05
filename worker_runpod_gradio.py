import random, time

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
    return result

import gradio as gr

def generate_wrapper(
    positive_prompt, height, width, video_length, seed, 
    negative_prompt, infer_steps, guidance_scale, 
    num_videos_per_prompt, flow_shift, batch_size, 
    embedded_guidance_scale
):
    input_data = {
        "positive_prompt": positive_prompt,
        "height": height,
        "width": width,
        "video_length": video_length,
        "seed": seed,
        "negative_prompt": negative_prompt,
        "infer_steps": infer_steps,
        "guidance_scale": guidance_scale,
        "num_videos_per_prompt": num_videos_per_prompt,
        "flow_shift": flow_shift,
        "batch_size": batch_size,
        "embedded_guidance_scale": embedded_guidance_scale,
    }
    result = generate({"input": input_data})
    return result

with gr.Blocks(css=".gradio-container {max-width: 1080px !important}", analytics_enabled=False) as demo:
    with gr.Row():
        with gr.Column():
            positive_prompt = gr.Textbox(
                label="Positive Prompt", 
                value="a cat is running, realistic."
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt", 
                value="Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion", 
                placeholder="Negative prompt (optional)"
            )
            height = gr.Slider(
                label="Height", minimum=128, maximum=2048, step=8, value=368
            )
            width = gr.Slider(
                label="Width", minimum=128, maximum=2048, step=8, value=640
            )
            video_length = gr.Slider(
                label="Video Length (frames)", minimum=1, maximum=258, step=1, value=129
            )
            seed = gr.Number(
                label="Seed (If 0 Random)", value=0, precision=0
            )
            infer_steps = gr.Slider(
                label="Inference Steps", minimum=10, maximum=100, step=1, value=50
            )
            guidance_scale = gr.Slider(
                label="Guidance Scale", minimum=0.1, maximum=20.0, step=0.1, value=1.0
            )
            num_videos_per_prompt = gr.Slider(
                label="Number of Videos", minimum=1, maximum=4, step=1, value=1
            )
            flow_shift = gr.Slider(
                label="Flow Shift", minimum=-10.0, maximum=10.0, step=0.1, value=7.0
            )
            batch_size = gr.Slider(
                label="Batch Size", minimum=1, maximum=16, step=1, value=1
            )
            embedded_guidance_scale = gr.Slider(
                label="Embedded Guidance Scale", minimum=0.1, maximum=10.0, step=0.1, value=6.0
            )
        with gr.Row():
            with gr.Column():
                video_output = gr.Video(label="Generated Video")
                generate_button = gr.Button("Generate Video")
    
    generate_button.click(
        fn=generate_wrapper,
        inputs=[
            positive_prompt, height, width, video_length, seed, 
            negative_prompt, infer_steps, guidance_scale, 
            num_videos_per_prompt, flow_shift, batch_size, 
            embedded_guidance_scale
        ],
        outputs=[video_output],
        show_progress=True
    )

demo.queue().launch(inline=False, share=False, debug=True, server_name='0.0.0.0', server_port=7860, allowed_paths=["/content"])