import whisper
import torch
import datetime
import subprocess
import numpy as np
import time
import os
from pyannote.audio import Pipeline
from pyannote.core import Segment

# -------- CONFIG --------
AUDIO_FILE = "audio.m4a"           # Input file
MODEL_SIZE = "medium"              # tiny, base, small, medium, large
LANGUAGE = "English"               # 'English' or 'any'
HF_TOKEN = os.getenv("HF_TOKEN")   # Hugging Face token from env
OUTPUT_FILE = "final_transcript.txt"
# -------------------------

# Utility function for formatted time
def format_time(secs):
    return str(datetime.timedelta(seconds=round(secs)))

# Start global timer
start_all = time.time()

# ---------------------------------
# STEP 1: Convert to WAV (if needed)
# ---------------------------------
t0 = time.time()
if not AUDIO_FILE.endswith(".wav"):
    print("🔄 Converting to WAV using FFmpeg...")
    subprocess.call(["ffmpeg", "-i", AUDIO_FILE, "audio.wav", "-y", "-loglevel", "quiet"])
    AUDIO_FILE = "audio.wav"
print(f"✅ Conversion done in {time.time() - t0:.2f}s")

# ---------------------------------
# STEP 2: Load Whisper model
# ---------------------------------
t1 = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = MODEL_SIZE
if LANGUAGE == "English" and MODEL_SIZE != "large":
    model_name += ".en"

print(f"\n⚙️ Loading Whisper model: {model_name} on {device}...")
whisper_model = whisper.load_model(model_name, device=device)
print(f"✅ Whisper model loaded in {time.time() - t1:.2f}s")

# ---------------------------------
# STEP 3: Transcription
# ---------------------------------
t2 = time.time()
print("\n🎧 Transcribing with Whisper...")
result = whisper_model.transcribe(AUDIO_FILE)
segments = result["segments"]
print(f"✅ Transcription complete in {time.time() - t2:.2f}s")
print(f"📄 Total segments: {len(segments)}")

# ---------------------------------
# STEP 4: Speaker Diarization
# ---------------------------------
t3 = time.time()
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN not found in environment. Please set it before running.")
print("\n🧩 Running Pyannote diarization pipeline...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)
diarization = pipeline(AUDIO_FILE)
print(f"✅ Diarization complete in {time.time() - t3:.2f}s")

# ---------------------------------
# STEP 5: Match Whisper segments with speaker segments
# ---------------------------------
t4 = time.time()
print("\n🧠 Assigning speakers to text segments...")
def get_speaker_for_segment(start, end):
    segment = Segment(start, end)
    overlapping = diarization.crop(segment)
    speakers = [label for _, _, label in overlapping.itertracks(yield_label=True)]
    if not speakers:
        return "Unknown"
    return max(set(speakers), key=speakers.count)

for seg in segments:
    seg["speaker"] = get_speaker_for_segment(seg["start"], seg["end"])
print(f"✅ Speaker alignment done in {time.time() - t4:.2f}s")

# ---------------------------------
# STEP 6: Write final transcript
# ---------------------------------
t5 = time.time()
print("\n💾 Saving final transcript...")
with open(OUTPUT_FILE, "w") as f:
    for i, seg in enumerate(segments):
        start = format_time(seg["start"])
        speaker = seg["speaker"]
        text = seg["text"].strip()

        # Add new speaker header if speaker changes
        if i == 0 or seg["speaker"] != segments[i - 1]["speaker"]:
            f.write(f"\n[{start}] {speaker}:\n")
        f.write(text + " ")
print(f"✅ Transcript saved as '{OUTPUT_FILE}' in {time.time() - t5:.2f}s")

# ---------------------------------
# STEP 7: Summary Analytics
# ---------------------------------
end_all = time.time()
print("\n📊 -------- PROCESS ANALYTICS --------")
print(f"Audio conversion:     {time.time() - t0:.2f}s")
print(f"Model loading:        {time.time() - t1:.2f}s")
print(f"Transcription:        {time.time() - t2:.2f}s")
print(f"Diarization:          {time.time() - t3:.2f}s")
print(f"Speaker alignment:    {time.time() - t4:.2f}s")
print(f"Transcript writing:   {time.time() - t5:.2f}s")
print("-------------------------------------")
print(f"🕒 TOTAL PIPELINE TIME: {end_all - start_all:.2f}s")
print("-------------------------------------")
print(f"✅ Diarization complete! Transcript saved to '{OUTPUT_FILE}'")
