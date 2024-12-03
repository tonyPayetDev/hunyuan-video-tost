FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run -d /content -o cuda_12.6.2_560.35.03_linux.run && sh cuda_12.6.2_560.35.03_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    git clone https://github.com/aristocratos/btop /content/btop && cd /content/btop && make && make install && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home
    
USER camenduru

RUN pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install xformers==0.0.28.post3 && \
    pip install opencv-contrib-python imageio imageio-ffmpeg ffmpeg-python av runpod && \
    pip install torchsde einops diffusers transformers accelerate loguru && \
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    git clone https://github.com/Tencent/HunyuanVideo /content/HunyuanVideo && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt -d /content/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/transformers -o mp_rank_00_model_states.pt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/raw/main/hunyuan-video-t2v-720p/vae/config.json -d /content/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/vae/pytorch_model.pt -d /content/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/vae -o pytorch_model.pt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/raw/main/text_encoder/config.json -d /content/HunyuanVideo/ckpts/text_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/raw/main/text_encoder/generation_config.json -d /content/HunyuanVideo/ckpts/text_encoder -o generation_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/resolve/main/text_encoder/model-00001-of-00004.safetensors -d /content/HunyuanVideo/ckpts/text_encoder -o model-00001-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/resolve/main/text_encoder/model-00002-of-00004.safetensors -d /content/HunyuanVideo/ckpts/text_encoder -o model-00002-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/resolve/main/text_encoder/model-00003-of-00004.safetensors -d /content/HunyuanVideo/ckpts/text_encoder -o model-00003-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/resolve/main/text_encoder/model-00004-of-00004.safetensors -d /content/HunyuanVideo/ckpts/text_encoder -o model-00004-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/raw/main/text_encoder/model.safetensors.index.json -d /content/HunyuanVideo/ckpts/text_encoder -o model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/raw/main/text_encoder/special_tokens_map.json -d /content/HunyuanVideo/ckpts/text_encoder -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/resolve/main/text_encoder/tokenizer.json -d /content/HunyuanVideo/ckpts/text_encoder -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/raw/main/text_encoder/tokenizer_config.json -d /content/HunyuanVideo/ckpts/text_encoder -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/raw/main/text_encoder_2/config.json -d /content/HunyuanVideo/ckpts/text_encoder_2 -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/raw/main/text_encoder_2/merges.txt -d /content/HunyuanVideo/ckpts/text_encoder_2 -o merges.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/resolve/main/text_encoder_2/model.safetensors -d /content/HunyuanVideo/ckpts/text_encoder_2 -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/raw/main/text_encoder_2/preprocessor_config.json -d /content/HunyuanVideo/ckpts/text_encoder_2 -o preprocessor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/resolve/main/text_encoder_2/pytorch_model.bin -d /content/HunyuanVideo/ckpts/text_encoder_2 -o pytorch_model.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/raw/main/text_encoder_2/special_tokens_map.json -d /content/HunyuanVideo/ckpts/text_encoder_2 -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/raw/main/text_encoder_2/tokenizer.json -d /content/HunyuanVideo/ckpts/text_encoder_2 -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/raw/main/text_encoder_2/tokenizer_config.json -d /content/HunyuanVideo/ckpts/text_encoder_2 -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/HunyuanVideo/raw/main/text_encoder_2/vocab.json -d /content/HunyuanVideo/ckpts/text_encoder_2 -o vocab.json

COPY ./worker_runpod.py /content/HunyuanVideo/worker_runpod.py
WORKDIR /content/HunyuanVideo
CMD python worker_runpod.py