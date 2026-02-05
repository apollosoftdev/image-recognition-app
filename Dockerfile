# syntax=docker/dockerfile:1.4
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Pre-download the OCR models during build (with cache)
RUN --mount=type=cache,target=/root/.keras \
    python -c "import keras_ocr; keras_ocr.pipeline.Pipeline()" || true
RUN --mount=type=cache,target=/root/.cache \
    python -c "from doctr.models import ocr_predictor; ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)" || true
RUN --mount=type=cache,target=/root/.paddleocr \
    python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)" || true

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p static/uploads

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Run the application
CMD ["python", "app.py"]
