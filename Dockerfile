FROM python:3.10-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    COSYVOICE_HOME=/opt/CosyVoice \
    PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS

# Install system packages required for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        ffmpeg \
        sox \
        libsox-dev \
        libsndfile1 \
        build-essential

WORKDIR /app

# Pre-install PyTorch CPU wheels to avoid pulling CUDA runtimes
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1 \
        torchaudio==2.3.1

# Install Python dependencies
COPY dependencies/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade \
    "pip<25" \
    "setuptools==80.9.0" \
    wheel \
    packaging \
    nvidia-stub

# CosyVoice is not distributed as a normal PyPI package; install its source tree
# into PYTHONPATH so `from cosyvoice...` works at runtime.
RUN git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git "$COSYVOICE_HOME" \
    && cd "$COSYVOICE_HOME" \
    && for attempt in 1 2 3; do \
        git submodule update --init --recursive && break; \
        if [ "$attempt" = "3" ]; then exit 1; fi; \
        sleep 2; \
    done

COPY dependencies/CosyVoice/requirements-cpu.txt /tmp/requirements-cosyvoice.txt
RUN pip install --no-cache-dir --no-build-isolation -r /tmp/requirements-cosyvoice.txt

# Clean apt packages and cache
RUN apt remove -y build-essential git && apt autoremove -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p /app/temp /app/data /app/voices /app/logs \
    && chmod +x start.py

EXPOSE 8000

CMD ["python", "start.py"]
