import whisper
import torch
import datetime
import subprocess
import time
from pyannote.audio import Pipeline
import warnings
import platform
warnings.filterwarnings("ignore")
import os

from dotenv import load_dotenv
load_dotenv()

# -------- CONFIG (User must update HF_TOKEN and check NUM_SPEAKERS) --------
AUDIO_FILE = "audio.m4a"
MODEL_SIZE = "medium"  # tiny, base, small, medium, large
LANGUAGE = "en"
# ⚠️ CRITICAL: Replace with your actual, valid Hugging Face User Access Token 
# and ensure you have accepted the terms for the pyannote/speaker-diarization-3.1 model.
HF_TOKEN = os.getenv("HF_TOKEN")
OUTPUT_FILE = f"final_transcript_diarized.txt" # Changed name for clarity
NUM_SPEAKERS = 2  # CRITICAL: Must be exactly the number of speakers in the audio
DEBUG_MODE = True  # Set to True to see detailed info
# -------------------------

def format_time(secs):
    return str(datetime.timedelta(seconds=round(secs)))

def detect_device():
    """Auto-detect best available device for Whisper"""
    system = platform.system()
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 Detected: NVIDIA GPU on {system}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and system == "Darwin":
        device = "cpu" # Using CPU for Whisper on MPS is often more stable
        print(f"🚀 Detected: Apple Silicon on macOS")
        print(f"   Using CPU for Whisper (stability)")
    else:
        device = "cpu"
        print(f"💻 Using CPU on {system}")
    
    return device

def detect_diarization_device():
    """Detect best device for diarization"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        try:
            # Use 'mps' if stable, otherwise fall back to 'cpu'
            return torch.device("mps")
        except:
            return torch.device("cpu")
    return torch.device("cpu")

start_all = time.time()

# ---------------------------------
# STEP 1: Convert to WAV
# ---------------------------------
t0 = time.time()
wav_file = "audio.wav"

if not AUDIO_FILE.endswith(".wav"):
    print("🔄 Converting to WAV...")
    try:
        # NOTE: Ensure FFmpeg is installed and in your PATH
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
        print("❌ FFmpeg not found! Please install it.")
        exit(1)

# ---------------------------------
# STEP 2: Load Whisper model
# ---------------------------------
t1 = time.time()
device = detect_device()

model_name = MODEL_SIZE
if LANGUAGE == "en" and MODEL_SIZE != "large":
    model_name += ".en"

print(f"\n⚙️ Loading Whisper model: {model_name}...")
whisper_model = whisper.load_model(model_name, device=device)
print(f"✅ Model loaded in {time.time() - t1:.2f}s")

# ---------------------------------
# STEP 3: Transcription with WORD timestamps
# ---------------------------------
t2 = time.time()
print("\n🎧 Transcribing with Whisper (word-level timestamps)...")

use_fp16 = (device == "cuda")

result = whisper_model.transcribe(
    AUDIO_FILE,
    language=LANGUAGE,
    fp16=use_fp16,
    temperature=0.0,
    word_timestamps=True,  # CRITICAL: Enable word-level timestamps
    verbose=False
)

segments = result["segments"]
print(f"✅ Transcription complete in {time.time() - t2:.2f}s")
print(f"📄 Total segments: {len(segments)}")

# ---------------------------------
# STEP 4: Diarization (2 speakers)
# ---------------------------------
t3 = time.time()
print(f"\n🧩 Running speaker diarization (forcing {NUM_SPEAKERS} speakers)...")

diarization = None
try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    
    diarization_device = detect_diarization_device()
    pipeline.to(diarization_device)
    print(f"   Using {diarization_device} for diarization")
    
    # Force the number of speakers as per config
    diarization = pipeline(
        AUDIO_FILE,
        min_speakers=NUM_SPEAKERS,
        max_speakers=NUM_SPEAKERS
    )
    
    print(f"✅ Diarization complete in {time.time() - t3:.2f}s")
    
    # Count unique speakers
    unique_speakers = set()
    speaker_timeline = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        unique_speakers.add(speaker)
        speaker_timeline.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })
        
    print(f"👥 Detected speakers: {len(unique_speakers)}")

    if DEBUG_MODE:
        # 💡 DEBUG: Print raw diarization segments to check for errors
        print("\n🔍 Raw Speaker Timeline (first 10 segments):")
        for i, spk in enumerate(speaker_timeline[:10]):
            print(f"   [{format_time(spk['start'])} - {format_time(spk['end'])}] {spk['speaker']}")
    
except Exception as e:
    print(f"❌ Diarization error: {e}")
    print("   Ensure your HF_TOKEN is valid and you have accepted the pyannote/speaker-diarization-3.1 user agreement.")
    diarization = None
    speaker_timeline = [] # Initialize empty if failed

# ---------------------------------
# STEP 5: Word-level speaker assignment
# ---------------------------------
t4 = time.time()

if speaker_timeline:
    print("\n🧠 Assigning speakers at word level...")
    
    # Sort timeline by start time
    speaker_timeline.sort(key=lambda x: x['start'])
    
    def get_speaker_at_time(time_point):
        """Get speaker at a specific time point"""
        # 1. Check for perfect overlap
        for spk in speaker_timeline:
            if spk['start'] <= time_point <= spk['end']:
                return spk['speaker']
        
        # 2. If not found, find the segment closest to the time point
        closest_speaker = None
        min_distance = float('inf')
        
        for spk in speaker_timeline:
            # Check proximity to the segment itself (not just the center)
            if time_point < spk['start']:
                distance = spk['start'] - time_point
            elif time_point > spk['end']:
                distance = time_point - spk['end']
            else: # Should be caught by step 1, but for safety
                distance = 0
            
            if distance < min_distance:
                min_distance = distance
                closest_speaker = spk['speaker']
        
        return closest_speaker if closest_speaker else "SPEAKER_UNKNOWN"
    
    # Assign speakers to each word
    total_words = 0
    for seg in segments:
        if 'words' in seg and seg['words']:
            for word in seg['words']:
                word_time = (word['start'] + word['end']) / 2
                word['speaker'] = get_speaker_at_time(word_time)
                total_words += 1
        else:
            # Fallback for segments without word timestamps
            seg_time = (seg['start'] + seg['end']) / 2
            seg['speaker'] = get_speaker_at_time(seg_time)
    
    print(f"✅ Assigned speakers to {total_words} words in {time.time() - t4:.2f}s")
    
else:
    # No diarization available
    print("⚠️ Diarization failed or not available. Assigning all to SPEAKER_00.")
    total_words = 0
    for seg in segments:
        seg['speaker'] = "SPEAKER_00"
        if 'words' in seg and seg['words']:
            for word in seg['words']:
                word['speaker'] = "SPEAKER_00"
                total_words += 1

# ---------------------------------
# STEP 6: Reconstruct transcript with proper speaker changes
# ---------------------------------
t5 = time.time()
print("\n💾 Reconstructing transcript with speaker changes and smoothing...")

# Collect all words with speakers and timestamps
all_words = []
for seg in segments:
    # Prioritize word-level timestamps if available
    if 'words' in seg and seg['words']:
        for word in seg['words']:
            all_words.append({
                'start': word['start'],
                'end': word['end'],
                'text': word['word'].strip(),
                'speaker': word.get('speaker', 'SPEAKER_00')
            })
    # Fallback to segment-level timestamps
    elif 'text' in seg and seg['text'].strip():
        all_words.append({
            'start': seg['start'],
            'end': seg['end'],
            'text': seg['text'].strip(),
            'speaker': seg.get('speaker', 'SPEAKER_00')
        })

# --- SPEAKER SMOOTHING (To fix single-word speaker errors) ---
if all_words:
    for i in range(1, len(all_words) - 1):
        prev_speaker = all_words[i-1]['speaker']
        current_speaker = all_words[i]['speaker']
        next_speaker = all_words[i+1]['speaker']
        
        # If a word's speaker is different from both its neighbors, revert it to the dominant neighbor's speaker
        if current_speaker != prev_speaker and current_speaker != next_speaker:
            all_words[i]['speaker'] = prev_speaker # Revert to previous speaker
# -------------------------------------------------------------

# Write transcript with speaker changes
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    if not all_words:
        f.write("No words transcribed.\n")
    else:
        current_speaker = None
        current_line_words = []
        
        for i, word in enumerate(all_words):
            speaker = word['speaker']
            text = word['text']
            
            # Skip empty or punctuation-only text
            if not text:
                continue
                
            # Speaker change detected
            if speaker != current_speaker:
                # Write previous line if exists
                if current_line_words:
                    f.write(' '.join(current_line_words).replace(" ,", ",").replace(" .", ".") + '\n')
                
                # Start new speaker section
                f.write(f"\n[{format_time(word['start'])}] {speaker}:\n")
                current_speaker = speaker
                current_line_words = [text]
            else:
                current_line_words.append(text)
        
        # Write last line
        if current_line_words:
            f.write(' '.join(current_line_words).replace(" ,", ",").replace(" .", ".") + '\n')

print(f"✅ Transcript saved in {time.time() - t5:.2f}s")

# Count speaker changes
speaker_changes = 0
prev_speaker = None
for word in all_words:
    if prev_speaker and word['speaker'] != prev_speaker:
        speaker_changes += 1
    prev_speaker = word['speaker']

print(f"🔄 Total speaker changes: {speaker_changes}")

if DEBUG_MODE and all_words:
    print("\n🔍 First 20 words with final speakers (after smoothing):")
    for i, word in enumerate(all_words[:20]):
        print(f"   [{format_time(word['start'])}] {word['speaker']}: {word['text']}")

# ---------------------------------
# Summary
# ---------------------------------
total_time = time.time() - start_all

print("\n📊 -------- SUMMARY --------")
print(f"System:               {platform.system()}")
print(f"Total time:           {total_time:.2f}s")
print(f"Speaker changes:      {speaker_changes}")
print(f"Total words:          {total_words}")
print("---------------------------")
print(f"✅ Transcript: '{OUTPUT_FILE}'")