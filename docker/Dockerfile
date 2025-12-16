FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ffmpeg libgl1 \
    build-essential cmake ninja-build \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# GroundingDINO source
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git /opt/GroundingDINO \
 && pip install --no-cache-dir -e /opt/GroundingDINO \
 && cd /opt/GroundingDINO \
 && TORCH_CUDA_ARCH_LIST="8.9" FORCE_CUDA=1 python setup.py build_ext --inplace

# Build GroundingDINO custom ops (fix "CPU mode only")
RUN python -c "import torch; print('CUDA:', torch.cuda.is_available());" \
 && cd /opt/GroundingDINO \
 && python setup.py build_ext --inplace

# App code
COPY server /app/server

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
