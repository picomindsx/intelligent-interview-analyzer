import whisper
import datetime
import subprocess
import torch
import wave
import contextlib
import numpy as np
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from sklearn.cluster import AgglomerativeClustering

# -------- CONFIG --------
path = "audio.m4a"   # Input file
num_speakers = 2               # Adjust if known
language = 'English'           # 'any' or 'English'
model_size = 'medium'          # ['tiny','base','small','medium','large']
# -------------------------

# Select Whisper model
model_name = model_size
if language == 'English' and model_size != 'large':
    model_name += '.en'

# Convert to wav if needed
if path[-3:] != 'wav':
    subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
    path = 'audio.wav'

# Load Whisper
model = whisper.load_model(model_name)
result = model.transcribe(path)
segments = result["segments"]

# Audio setup
with contextlib.closing(wave.open(path,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

audio = Audio()

# Embedding model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=device)

def segment_embedding(segment):
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    return embedding_model(waveform[None])

# Collect embeddings
embeddings = np.zeros((len(segments), 192))
for i, segment in enumerate(segments):
    embeddings[i] = segment_embedding(segment)

embeddings = np.nan_to_num(embeddings)

# Clustering
clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(embeddings)
labels = clustering.labels_

for i in range(len(segments)):
    segments[i]["speaker"] = f"SPEAKER {labels[i] + 1}"

# Save transcript
def time(secs):
    return datetime.timedelta(seconds=round(secs))

with open("transcript.txt", "w") as f:
    for i, segment in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            f.write(f"\n{segment['speaker']} {time(segment['start'])}\n")
        f.write(segment["text"] + " ")
