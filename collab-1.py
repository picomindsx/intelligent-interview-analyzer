from faster_whisper import WhisperModel
import torch
import datetime
import subprocess
import time
from pyannote.audio import Pipeline
import warnings
warnings.filterwarnings("ignore")
import os

from dotenv import load_dotenv
load_dotenv()

# -------- CONFIG --------
AUDIO_FILE = "audio.m4a"
MODEL_SIZE = "medium"  # tiny, base, small, medium, large-v2, large-v3
LANGUAGE = "en"  # Use language code: en, es, fr, etc.
HF_TOKEN = os.getenv("HF_TOKEN")
OUTPUT_FILE = f"final_transcript_{MODEL_SIZE}.txt"

# Optimization settings
COMPUTE_TYPE = "int8"  # int8 (fastest), float16, float32
BEAM_SIZE = 5  # Lower = faster, 5 is good balance
# -------------------------

def format_time(secs):
    return str(datetime.timedelta(seconds=round(secs)))

start_all = time.time()

# ---------------------------------
# STEP 1: Convert to WAV if needed
# ---------------------------------
t0 = time.time()
wav_file = "audio.wav"

if not AUDIO_FILE.endswith(".wav"):
    print("🔄 Converting to WAV...")
    subprocess.call([
        "ffmpeg", "-i", AUDIO_FILE,
        "-ar", "16000",  # 16kHz for Whisper
        "-ac", "1",      # Mono
        "-c:a", "pcm_s16le",
        wav_file, "-y", "-loglevel", "quiet"
    ])
    AUDIO_FILE = wav_file
    print(f"✅ Conversion done in {time.time() - t0:.2f}s")

# ---------------------------------
# STEP 2: Load faster-whisper model
# ---------------------------------
t1 = time.time()
print(f"\n⚙️ Loading faster-whisper model: {MODEL_SIZE}...")

# faster-whisper uses CPU efficiently on Mac M4
model = WhisperModel(
    MODEL_SIZE,
    device="cpu",  # Use CPU for M4 (optimized)
    compute_type=COMPUTE_TYPE,  # int8 for speed
    num_workers=4  # Parallel processing
)
print(f"✅ Model loaded in {time.time() - t1:.2f}s")

# ---------------------------------
# STEP 3: Fast transcription with timestamps
# ---------------------------------
t2 = time.time()
print("\n🎧 Transcribing with faster-whisper...")

segments_list = []
transcription = ''
segments_iter, info = model.transcribe(
    AUDIO_FILE,
    language=LANGUAGE,
    beam_size=BEAM_SIZE,
    vad_filter=True,  # Voice activity detection - removes silence
    vad_parameters=dict(min_silence_duration_ms=100),
    word_timestamps=False,  # Set to True if you need word-level timing
    condition_on_previous_text=True
)

# Convert generator to list and keep timestamps
for segment in segments_iter:
    segments_list.append({
        "start": segment.start,
        "end": segment.end,
        "text": segment.text
    })
    transcription += segment.text

print(transcription)

print(f"✅ Transcription complete in {time.time() - t2:.2f}s")
print(f"📄 Total segments: {len(segments_list)}")
print(f"🎵 Detected language: {info.language} (probability: {info.language_probability:.2%})")

# ---------------------------------
# STEP 4: Speaker diarization
# ---------------------------------
t3 = time.time()
print("\n🧩 Running speaker diarization...")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

# Use MPS if available on Mac
if torch.backends.mps.is_available():
    pipeline.to(torch.device("mps"))
    print("🚀 Using Apple Metal (MPS) for diarization")

diarization = pipeline(AUDIO_FILE)
print(f"✅ Diarization complete in {time.time() - t3:.2f}s")

# ---------------------------------
# STEP 5: Assign speakers to segments
# ---------------------------------
t4 = time.time()
print("\n🧠 Assigning speakers to segments...")

# Build speaker timeline for fast lookup
speaker_timeline = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    speaker_timeline.append((turn.start, turn.end, speaker))

def get_speaker_for_segment(start, end):
    """Find speaker for a time segment"""
    mid_point = (start + end) / 2
    best_overlap = 0
    best_speaker = "Unknown"
    
    for spk_start, spk_end, speaker in speaker_timeline:
        # Check if midpoint falls in speaker segment
        if spk_start <= mid_point <= spk_end:
            overlap = min(end, spk_end) - max(start, spk_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
    
    return best_speaker

# Assign speakers
for seg in segments_list:
    seg["speaker"] = get_speaker_for_segment(seg["start"], seg["end"])

print(f"✅ Speaker alignment done in {time.time() - t4:.2f}s")

# ---------------------------------
# STEP 6: Save transcript with timestamps
# ---------------------------------
t5 = time.time()
print("\n💾 Saving transcript...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    current_speaker = None
    
    for seg in segments_list:
        start_time = format_time(seg["start"])
        end_time = format_time(seg["end"])
        speaker = seg["speaker"]
        text = seg["text"].strip()
        
        # Add speaker header when speaker changes
        if speaker != current_speaker:
            f.write(f"\n[{start_time}] {speaker}:\n")
            current_speaker = speaker
        
        # Write text with timestamp
        f.write(f"{text} ")

print(f"✅ Transcript saved in {time.time() - t5:.2f}s")

# ---------------------------------
# STEP 7: Summary
# ---------------------------------
total_time = time.time() - start_all
audio_duration = segments_list[-1]["end"] if segments_list else 0

print("\n📊 -------- PERFORMANCE SUMMARY --------")
print(f"Audio duration:       {format_time(audio_duration)}")
print(f"Conversion time:      {time.time() - t0:.2f}s")
print(f"Model loading:        {time.time() - t1:.2f}s")
print(f"Transcription:        {time.time() - t2:.2f}s")
print(f"Diarization:          {time.time() - t3:.2f}s")
print(f"Speaker alignment:    {time.time() - t4:.2f}s")
print(f"Saving:               {time.time() - t5:.2f}s")
print("----------------------------------------")
print(f"🕒 TOTAL TIME: {total_time:.2f}s")
if audio_duration > 0:
    print(f"⚡ Speed factor: {audio_duration/total_time:.2f}x realtime")
print("----------------------------------------")
print(f"✅ Complete! Transcript saved to '{OUTPUT_FILE}'")