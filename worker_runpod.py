import os, json, requests, random, time, runpod

import torch
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

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

    args = parse_args()
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained("/content/HunyuanVideo/ckpts", args=args)
    args = hunyuan_video_sampler.args

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
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})