import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import os, json, requests, random, time, runpod, base64
import torch
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

# Variables de connexion à Supabase
SUPABASE_URL = "https://rvsykocedohfdfdvbrfe.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ2c3lrb2NlZG9oZmRmZHZicmZlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDAxNDA3NTcsImV4cCI6MjA1NTcxNjc1N30.HLU5iDMlk-rFvNiIOXhQFF8-KNTSJwlaR7wIQPiacDM"
SUPABASE_BUCKET = "video"  

# Fonction d'envoi de la vidéo en base64 à Supabase
def upload_to_supabase(video_base64, file_name):
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{file_name}"
    
    headers = {
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "file": video_base64,
        "file_name": file_name
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        print("Video uploaded successfully!")
        return response.json()  # Retourne la réponse de Supabase
    else:
        print(f"Error uploading video: {response.status_code}, {response.text}")
        return {"error": response.text}

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
    video_path = f"/content/hunyuan-video-{seed}-tost.mp4"
    save_videos_grid(sample, video_path, fps=24)

    try:
        # Convertir la vidéo en base64
        with open(video_path, "rb") as file:
            video_base64 = base64.b64encode(file.read()).decode("utf-8")

        # Upload de la vidéo sur Supabase
        file_name = f"hunyuan-video-{seed}.mp4"  # Le nom que tu souhaites donner à la vidéo
        upload_response = upload_to_supabase(video_base64, file_name)

        if "error" not in upload_response:
            return {"video_base64": video_base64, "status": "DONE", "supabase_response": upload_response}

        return {"status": "FAILED", "error": upload_response.get("error", "Unknown error")}

    except Exception as e:
        return {"error": str(e), "status": "FAILED"}

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


runpod.serverless.start({"handler": generate})
