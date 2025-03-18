import os
import json
import requests
import random
import time
import runpod
import base64
import torch
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

# Variables de connexion à Supabase 
SUPABASE_URL = "https://rvsykocedohfdfdvbrfe.supabase.co"
SUPABASE_API_KEY = ${SUPA_ROLE_TOKEN}
SUPABASE_BUCKET = "video"

# Fonction d'envoi de la vidéo à Supabase
def upload_to_supabase(video_path, file_name):
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{file_name}"
    
    headers = {
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
    }

    with open(video_path, "rb") as file:
        files = {"file": file}
        response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        print("Video uploaded successfully!")
        return response.json()
    else:
        print(f"Error uploading video: {response.status_code}, {response.text}")
        return {"error": response.text}

# Initialisation du sampler
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

    # Vérification des outputs
    samples = outputs.get("samples", [])
    if not isinstance(samples, list) or len(samples) == 0:
        return {"status": "FAILED", "error": "No valid sample found"}

    # Traitement du premier sample
    sample = samples[0].unsqueeze(0)
    video_path = f"/content/hunyuan-video-{seed}-tost.mp4"

    # Sauvegarde de la vidéo
    save_videos_grid(sample, video_path, fps=24)

    # Vérification si le fichier existe
    if not os.path.exists(video_path):
        return {"status": "FAILED", "error": f"File {video_path} not found"}

    try:
        # Upload sur Supabase
        file_name = f"hunyuan-video-{seed}.mp4"
        upload_response = upload_to_supabase(video_path, file_name)

        # Vérification du retour de Supabase
        if "error" in upload_response:
            raise Exception(upload_response["error"])

        return {"status": "DONE", "supabase_response": upload_response}

    except Exception as e:
        return {"status": "FAILED", "error": str(e)}

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

# Lancement du serveur Runpod
runpod.serverless.start({"handler": generate})
