import whisper
import torch
import datetime
import subprocess
import time
from pyannote.audio import Pipeline
import warnings
import platform
import os
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

# -------- CONFIG --------
AUDIO_FILE = "audio.m4a"
MODEL_SIZE = "medium"  # tiny, base, small, medium, large
LANGUAGE = "en"
HF_TOKEN = os.getenv("HF_TOKEN")
OUTPUT_FILE = f"final_transcript_{MODEL_SIZE}.txt"
NUM_SPEAKERS = 2  # Exactly 2 speakers
DEBUG_MODE = True  # Set to True to see detailed info
# -------------------------

def format_time(secs):
    return str(datetime.timedelta(seconds=round(secs)))

def detect_device():
    """Auto-detect best available device"""
    system = platform.system()
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 Detected: NVIDIA GPU on {system}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and system == "Darwin":
        device = "cpu"  # CPU for Whisper stability
        print(f"🚀 Detected: Apple Silicon on macOS")
        print(f"   Using CPU for Whisper")
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
print(f"\n🧩 Running speaker diarization (forcing 2 speakers)...")

try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    
    diarization_device = detect_diarization_device()
    pipeline.to(diarization_device)
    print(f"   Using {diarization_device} for diarization")
    
    # Force exactly 2 speakers
    diarization = pipeline(
        AUDIO_FILE,
        min_speakers=NUM_SPEAKERS,
        max_speakers=NUM_SPEAKERS
    )
    
    print(f"✅ Diarization complete in {time.time() - t3:.2f}s")
    
    # Count unique speakers
    unique_speakers = set()
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        unique_speakers.add(speaker)
    print(f"👥 Detected speakers: {len(unique_speakers)}")
    
except Exception as e:
    print(f"❌ Diarization error: {e}")
    diarization = None

# ---------------------------------
# STEP 5: Word-level speaker assignment
# ---------------------------------
t4 = time.time()

if diarization:
    print("\n🧠 Assigning speakers at word level...")
    
    # Build speaker timeline
    speaker_timeline = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_timeline.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })
    
    speaker_timeline.sort(key=lambda x: x['start'])
    
    if DEBUG_MODE:
        print("\n🔍 Speaker timeline (first 20 segments):")
        for i, spk in enumerate(speaker_timeline[:20]):
            print(f"   [{format_time(spk['start'])} - {format_time(spk['end'])}] {spk['speaker']}")
    
    def get_speaker_at_time(time_point):
        """Get speaker at a specific time point"""
        for spk in speaker_timeline:
            if spk['start'] <= time_point <= spk['end']:
                return spk['speaker']
        
        # If not found, find closest
        closest_speaker = None
        min_distance = float('inf')
        for spk in speaker_timeline:
            spk_mid = (spk['start'] + spk['end']) / 2
            distance = abs(time_point - spk_mid)
            if distance < min_distance:
                min_distance = distance
                closest_speaker = spk['speaker']
        
        return closest_speaker if closest_speaker else "SPEAKER_00"
    
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
    for seg in segments:
        seg['speaker'] = "SPEAKER_00"
        if 'words' in seg and seg['words']:
            for word in seg['words']:
                word['speaker'] = "SPEAKER_00"

# ---------------------------------
# STEP 6: Reconstruct transcript with proper speaker changes
# ---------------------------------
t5 = time.time()
print("\n💾 Reconstructing transcript with speaker changes...")

# Collect all words with speakers and timestamps
all_words = []
for seg in segments:
    if 'words' in seg and seg['words']:
        for word in seg['words']:
            all_words.append({
                'start': word['start'],
                'end': word['end'],
                'text': word['word'],
                'speaker': word.get('speaker', 'SPEAKER_00')
            })
    else:
        # Fallback if no word timestamps
        all_words.append({
            'start': seg['start'],
            'end': seg['end'],
            'text': seg['text'],
            'speaker': seg.get('speaker', 'SPEAKER_00')
        })

# Write transcript with speaker changes
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    if not all_words:
        print("⚠️ No words to write!")
    else:
        current_speaker = None
        current_line_start = None
        current_line_words = []
        
        for i, word in enumerate(all_words):
            speaker = word['speaker']
            text = word['text'].strip()
            
            # Speaker change detected
            if speaker != current_speaker:
                # Write previous line if exists
                if current_line_words:
                    f.write(' '.join(current_line_words) + '\n')
                
                # Start new speaker section
                f.write(f"\n[{format_time(word['start'])}] {speaker}:\n")
                current_speaker = speaker
                current_line_start = word['start']
                current_line_words = [text]
            else:
                current_line_words.append(text)
        
        # Write last line
        if current_line_words:
            f.write(' '.join(current_line_words) + '\n')

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
    print("\n🔍 First 20 words with speakers:")
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
print(f"Total words:          {len(all_words)}")
print("---------------------------")
print(f"✅ Transcript: '{OUTPUT_FILE}'")