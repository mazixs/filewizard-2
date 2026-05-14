# Build arguments for different configurations
ARG CUDA_VERSION=12.6.3
ARG PYTHON_VERSION=3.12-slim

# ==============================================================================
# STAGE 0: CUDA Build Stage
# Builds Python dependencies using the CUDA development image.
# Uses a venv to avoid PEP 668 restrictions on Ubuntu 24.04 system Python.
# ==============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS cuda-builder

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Install system dependencies and Python 3.12 (shipped with Ubuntu 24.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    curl \
    libxml2-dev \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a virtual environment and upgrade pip/setuptools
RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel

WORKDIR /app

# Copy CUDA-specific requirements and install them.
# PyTorch is installed first from its specific index for CUDA compatibility.
COPY requirements_cuda.txt requirements.txt
RUN /opt/venv/bin/pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && /opt/venv/bin/pip install --upgrade -r requirements.txt

# ==============================================================================
# STAGE 1: Full Build Stage
# Builds Python dependencies for the non-GPU full version.
# ==============================================================================
FROM python:${PYTHON_VERSION} AS full-builder

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    curl \
    libxml2-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install the standard requirements.
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --upgrade -r requirements.txt

# ==============================================================================
# STAGE 2: Small Build Stage
# Builds Python dependencies for the non-GPU small version.
# ==============================================================================
FROM python:${PYTHON_VERSION} AS small-builder

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    curl \
    libxml2-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the small-build requirements and install them.
COPY requirements_small.txt requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --upgrade -r requirements.txt

# ==============================================================================
# STAGE 3: CUDA Final Stage
# Creates the final, runnable CUDA image.
# ==============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS cuda-final

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

# Install Python 3.12 and the venv so we can copy the pre-built one
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu \
    ghostscript poppler-utils libreoffice \
    texlive-xetex texlive-latex-recommended texlive-fonts-recommended \
    unpaper calibre ffmpeg libvips-tools libxml2-dev graphicsmagick inkscape \
    potrace pngquant sox jpegoptim libsox-fmt-mp3 lame \
    libportaudio2 libportaudiocpp0 portaudio19-dev \
    libxml2 supervisor libcudnn9-cuda-12 libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-built venv from the builder stage
COPY --from=cuda-builder /opt/venv /opt/venv

# Copy application code and configuration.
COPY supervisor.conf /etc/supervisor/conf.d/supervisor.conf
COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
RUN chmod +x run.sh
CMD ["/usr/bin/supervisord", "-n"]

# ==============================================================================
# STAGE 4: Full Final Stage
# Creates the final, runnable full image (non-GPU).
# ==============================================================================
FROM python:${PYTHON_VERSION} AS full-final

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Install all runtime dependencies in a single layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu \
    ghostscript poppler-utils libreoffice \
    pandoc lmodern texlive-xetex texlive-latex-recommended texlive-fonts-recommended \
    unpaper calibre ffmpeg libvips-tools libxml2-dev graphicsmagick inkscape \
    resvg pngquant sox jpegoptim libsox-fmt-mp3 lame \
    libportaudio2 libportaudiocpp0 portaudio19-dev \
    libxml2 supervisor \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from the full builder stage.
COPY --from=full-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=full-builder /usr/local/bin /usr/local/bin

# Copy application code and configuration.
COPY supervisor.conf /etc/supervisor/conf.d/supervisor.conf
COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
RUN chmod +x run.sh
CMD ["/usr/bin/supervisord", "-n"]

# ==============================================================================
# STAGE 5: Small Final Stage
# Creates the final, runnable small image (non-GPU).
# ==============================================================================
FROM python:${PYTHON_VERSION} AS small-final

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Install the reduced set of runtime dependencies in a single layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu \
    ghostscript poppler-utils libreoffice pandoc \
    unpaper calibre ffmpeg libvips-tools libxml2-dev graphicsmagick \
    pngquant sox jpegoptim libsox-fmt-mp3 lame \
    libportaudio2 libportaudiocpp0 portaudio19-dev \
    libxml2 supervisor \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from the small builder stage.
COPY --from=small-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=small-builder /usr/local/bin /usr/local/bin

# Copy application code and configuration.
COPY supervisor.conf /etc/supervisor/conf.d/supervisor.conf
COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
RUN chmod +x run.sh
CMD ["/usr/bin/supervisord", "-n"]
