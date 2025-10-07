import whisperx
import torch
import datetime
import subprocess
import time
import warnings
import platform
from pyannote.audio import Pipeline
warnings.filterwarnings("ignore")

import os

from dotenv import load_dotenv
load_dotenv()

# -------- CONFIG --------
AUDIO_FILE = "audio.m4a"
MODEL_SIZE = "medium"  # tiny, base, small, medium, large-v2, large-v3
LANGUAGE = "en"
HF_TOKEN = os.getenv("HF_TOKEN")
OUTPUT_FILE = f"final_transcript_{MODEL_SIZE}.txt"
NUM_SPEAKERS = 2
BATCH_SIZE = 16  # Increase for faster processing
# -------------------------

def format_time(secs):
    return str(datetime.timedelta(seconds=round(secs)))

def detect_device():
    """Auto-detect best device - WhisperX only supports CUDA and CPU"""
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        print(f"🚀 Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        # WhisperX doesn't support MPS, use CPU even on Mac
        device = "cpu"
        compute_type = "int8"  # Use int8 for faster CPU processing
        if torch.backends.mps.is_available():
            print(f"🚀 Apple Silicon detected - using optimized CPU (int8)")
        else:
            print(f"💻 Using CPU on {platform.system()}")
    
    return device, compute_type

start_all = time.time()

# ---------------------------------
# STEP 1: Convert to WAV
# ---------------------------------
t0 = time.time()
wav_file = "audio.wav"

if not AUDIO_FILE.endswith(".wav"):
    print("🔄 Converting to WAV...")
    try:
        subprocess.call([
            "ffmpeg", "-i", AUDIO_FILE,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            wav_file, "-y", "-loglevel", "quiet"
        ])
        AUDIO_FILE = wav_file
        print(f"✅ Conversion done in {time.time() - t0:.2f}s")
    except FileNotFoundError:
        print("❌ FFmpeg not found!")
        exit(1)

# ---------------------------------
# STEP 2: Load audio
# ---------------------------------
t1 = time.time()
device, compute_type = detect_device()

print("\n📂 Loading audio file...")
audio = whisperx.load_audio(AUDIO_FILE)
print(f"✅ Audio loaded in {time.time() - t1:.2f}s")

# ---------------------------------
# STEP 3: Transcribe with WhisperX
# ---------------------------------
t2 = time.time()
print(f"\n🎧 Transcribing with WhisperX ({MODEL_SIZE})...")

model = whisperx.load_model(
    MODEL_SIZE, 
    device, 
    compute_type=compute_type,
    language=LANGUAGE
)

result = model.transcribe(
    audio, 
    batch_size=BATCH_SIZE,
    language=LANGUAGE
)

print(f"✅ Transcription complete in {time.time() - t2:.2f}s")
print(f"📄 Initial segments: {len(result['segments'])}")

# ---------------------------------
# STEP 4: Align whisper output (word-level timestamps)
# ---------------------------------
t3 = time.time()
print("\n🎯 Aligning word-level timestamps...")

model_a, metadata = whisperx.load_align_model(
    language_code=LANGUAGE, 
    device=device
)

result = whisperx.align(
    result["segments"], 
    model_a, 
    metadata, 
    audio, 
    device,
    return_char_alignments=False
)

print(f"✅ Alignment complete in {time.time() - t3:.2f}s")

# ---------------------------------
# STEP 5: Speaker Diarization with WhisperX
# ---------------------------------
t4 = time.time()
print(f"\n🧩 Running speaker diarization ({NUM_SPEAKERS} speakers)...")

# Use pyannote directly through whisperx
from pyannote.audio import Pipeline

diarize_model = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

# Move to appropriate device
if device == "cuda":
    diarize_model.to(torch.device("cuda"))

diarize_segments = diarize_model(
    AUDIO_FILE,
    min_speakers=NUM_SPEAKERS,
    max_speakers=NUM_SPEAKERS
)

print(f"✅ Diarization complete in {time.time() - t4:.2f}s")

# ---------------------------------
# STEP 6: Assign speakers to words
# ---------------------------------
t5 = time.time()
print("\n🧠 Assigning speakers to words...")

result = whisperx.assign_word_speakers(diarize_segments, result)

print(f"✅ Speaker assignment complete in {time.time() - t5:.2f}s")

# Count speaker changes
speaker_changes = 0
prev_speaker = None
total_words = 0

for segment in result["segments"]:
    if "words" in segment:
        for word in segment["words"]:
            total_words += 1
            current_speaker = word.get("speaker", "UNKNOWN")
            if prev_speaker and current_speaker != prev_speaker:
                speaker_changes += 1
            prev_speaker = current_speaker

print(f"🔄 Speaker changes detected: {speaker_changes}")
print(f"📝 Total words with speakers: {total_words}")

# ---------------------------------
# STEP 7: Write formatted transcript
# ---------------------------------
t6 = time.time()
print("\n💾 Writing transcript...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    current_speaker = None
    current_line_start = None
    current_words = []
    
    for segment in result["segments"]:
        if "words" not in segment:
            continue
            
        for word in segment["words"]:
            speaker = word.get("speaker", "UNKNOWN")
            text = word.get("word", "").strip()
            start_time = word.get("start", 0)
            
            if not text:
                continue
            
            # Speaker changed
            if speaker != current_speaker:
                # Write previous line
                if current_words:
                    f.write(" ".join(current_words) + "\n")
                
                # Start new speaker line
                f.write(f"\n[{format_time(start_time)}] {speaker}:\n")
                current_speaker = speaker
                current_line_start = start_time
                current_words = [text]
            else:
                current_words.append(text)
    
    # Write last line
    if current_words:
        f.write(" ".join(current_words) + "\n")

print(f"✅ Transcript saved in {time.time() - t6:.2f}s")

# ---------------------------------
# Summary
# ---------------------------------
total_time = time.time() - start_all

print("\n📊 -------- SUMMARY --------")
print(f"System:               {platform.system()}")
print(f"Device:               {device}")
print(f"Total time:           {total_time:.2f}s")
print(f"Speaker changes:      {speaker_changes}")
print(f"Words processed:      {total_words}")
print("---------------------------")
print(f"✅ Transcript saved: '{OUTPUT_FILE}'")
print("\n💡 TIP: WhisperX provides much better speaker diarization for conversations!")