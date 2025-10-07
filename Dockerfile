FROM python:3.10-slim

# Install system deps
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# Copy code
WORKDIR /app
COPY diarize_fast.py /app/

# Install dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio openai-whisper pyannote.audio librosa soundfile numpy

# Env var for Hugging Face token
ENV HF_TOKEN=""

CMD ["python", "collab.py"]
