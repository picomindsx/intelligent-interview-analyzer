from faster_whisper import WhisperModel
import torch
import datetime
import subprocess
import time
from pyannote.audio import Pipeline
import warnings
import json
warnings.filterwarnings("ignore")
from db import InterviewDB

import os

from dotenv import load_dotenv
load_dotenv()

# Initialize DB
db = InterviewDB("interview_analyzer.db")

from gemini_client import (
    get_diarization, 
    get_questions_and_answers, 
    get_summary,
    extract_common_question_answers,
    get_questions_and_answers_as_summary, 
    extract_common_question_answers_as_summary,
    generate_overall_summary
    )

# -------- CONFIG --------
# AUDIO_FILE = "audio.m4a"
# AUDIO_FILE = "Addy intake interview.m4a"
# AUDIO_FILE = "Aaron Allen TOP Intake.m4a"
AUDIO_FILE = "audio.m4a"
MODEL_SIZE = "tiny"  # tiny, base, small, medium, large-v2, large-v3
LANGUAGE = "en"  # Use language code: en, es, fr, etc.
HF_TOKEN = os.getenv("HF_TOKEN")

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

# print(transcription)

print(f"✅ Transcription complete in {time.time() - t2:.2f}s")
print(f"📄 Total segments: {len(segments_list)}")
print(f"🎵 Detected language: {info.language} (probability: {info.language_probability:.2%})")

# ---------------------------------
# STEP 4: Speaker diarization
# ---------------------------------
t3 = time.time()
out = get_diarization(transcription)
print(f"✅ Diarization complete in {time.time() - t3:.2f}s")

# ---------------------------------
# STEP 5: Extract Q&A pairs
# ---------------------------------
t4 = time.time()
print("\n💬 Extracting question–answer pairs using Gemini...")
qa_output = get_questions_and_answers_as_summary(transcription)
print(f"✅ Q&A extraction complete in {time.time() - t4:.2f}s")
print("\n📋 Extracted Q&A Pairs:", qa_output)

# ---------------------------------
# STEP 5B: Extract Common Question Answers
# ---------------------------------
print("\n🔍 Extracting common question answers...")
common_answers = extract_common_question_answers_as_summary(transcription)
print(f"✅ Common question extraction complete in {time.time() - t4:.2f}s")
print("\n📋 Common Question Answers:", common_answers)

# ---------------------------------
# STEP 6: Save transcript with timestamps
# ---------------------------------
t5 = time.time()
print("\n💾 Saving transcript...")

# ---------------------------------
# STEP 7: Summary
# ---------------------------------
total_time = time.time() - start_all
audio_duration = segments_list[-1]["end"] if segments_list else 0

summary = generate_overall_summary(segments_iter, qa_output)

db.save_interview_results(AUDIO_FILE, summary, qa_output, common_answers)

print("\n📊 -------- PERFORMANCE SUMMARY --------")
print(f"Audio duration:       {format_time(audio_duration)}")
print(f"Conversion time:      {time.time() - t0:.2f}s")
print(f"Model loading:        {time.time() - t1:.2f}s")
print(f"Transcription:        {time.time() - t2:.2f}s")
print(f"Diarization:          {time.time() - t3:.2f}s")
print(f"Saving:               {time.time() - t5:.2f}s")
print("----------------------------------------")
print(f"🕒 TOTAL TIME: {total_time:.2f}s")
if audio_duration > 0:
    print(f"⚡ Speed factor: {audio_duration/total_time:.2f}x realtime")
print("----------------------------------------")
print("✅ Complete!")

