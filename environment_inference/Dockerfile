# Base image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# 日本語設定
RUN apt-get update && apt-get install -y locales \
    && localedef -i ja_JP -f UTF-8 ja_JP.UTF-8 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

## Pythonパッケージ統合インストール
# ベースイメージに既にTorch 2.1.2 + CUDA12.1 が含まれるため再インストールしない。
# 必要になればここでバージョンを揃えて更新（例: base image を合わせて更新）
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchcodec \
    ffmpeg-python \
    ffmpeg \
    pillow \
    ipykernel \
    jupyter \
    transformers==4.57.0 \
    deepspeed \
    decord \
    accelerate \
    triton \
    sentencepiece==0.2.0 \
    einops==0.8.0 \
    timm==0.9.16 \
    peft==0.11.1 \
    datasets==2.19.1 \
    gradio==4.31.5 \
    matplotlib==3.9.0 \
    scipy==1.13.1 \
    qwen_vl_utils==0.0.11 \
    tensorboard \
    bitsandbytes \
    trl \
    cbor2 \
    # flash_attn==2.7.4.post1  # PyTorch 2.1.2 + CUDA 12.1 ベースでソースビルド失敗のため後で必要ならランタイムインストール検討
    # vllm==0.11.0 \
    # lmdeploy==0.10.0 \
    && pip cache purge

RUN pip install vllm==0.11.0 

# Set environment variables
ENV PYTHONPATH=/workspace \
    LANG=ja_JP.UTF-8 \
    LC_ALL=ja_JP.UTF-8

CMD ["/bin/bash"]