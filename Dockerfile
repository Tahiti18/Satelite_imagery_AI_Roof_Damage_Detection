FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

# Clean out any accidental preinstalled/broken numerical stack.
RUN pip uninstall -y numpy torch torchvision ultralytics opencv-python opencv-python-headless || true

# Install NumPy first and pin it below NumPy 2.
RUN pip install --no-cache-dir numpy==1.26.4

# Install CPU-only PyTorch after NumPy is stable.
RUN pip install --no-cache-dir torch==2.4.1+cpu torchvision==0.19.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest. requirements.txt intentionally excludes torch/torchvision.
RUN pip install --no-cache-dir -r requirements.txt

# Smoke test during build so failed NumPy/Ultralytics imports fail before deploy.
RUN python - <<'PY'
import numpy
print("numpy", numpy.__version__)
import torch
print("torch", torch.__version__)
import cv2
print("cv2", cv2.__version__)
import ultralytics
print("ultralytics import ok")
PY

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
