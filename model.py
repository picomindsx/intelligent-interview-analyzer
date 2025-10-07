import os
import torch
import whisperx
import subprocess
import tempfile
from pydub import AudioSegment
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from app.logger import logger

# ------------------------------------------------------------
# Utility: Convert audio → wav (temp file)
# ------------------------------------------------------------
def convert_to_wav(input_file):
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    subprocess.run([
        "ffmpeg", "-y", "-i", input_file, "-ar", "16000", "-ac", "1", tmp_wav.name
    ], check=True)
    return tmp_wav.name

# ------------------------------------------------------------
# Utility: Split wav into chunks
# ------------------------------------------------------------
def split_audio(input_file, chunk_length_ms=5*60*1000):  # 5 min
    audio = AudioSegment.from_wav(input_file)
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
        end = min(start + chunk_length_ms, len(audio))
        chunk = audio[start:end]
        tmp_chunk = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_chunk.close()
        chunk.export(tmp_chunk.name, format="wav")
        chunks.append(tmp_chunk.name)
    return chunks

# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------
load_dotenv()
INPUT_FILE = "./Aaron Allen TOP Intake.m4a"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ File not found: {INPUT_FILE}")

if INPUT_FILE.lower().endswith(".m4a") or INPUT_FILE.lower().endswith(".mp3"):
    AUDIO_FILE = convert_to_wav(INPUT_FILE)
else:
    AUDIO_FILE = INPUT_FILE

# ------------------------------------------------------------
# Device setup
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

# ------------------------------------------------------------
# Load models
# ------------------------------------------------------------
logger.info("[green]Loading WhisperX model...[/green]")
model = whisperx.load_model("medium", device, compute_type=compute_type)

logger.info("[green]Loading diarization model...[/green]")
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("❌ HF_TOKEN missing. Add it in your .env or export it.")

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization", use_auth_token=hf_token
)

# ------------------------------------------------------------
# Transcription
# ------------------------------------------------------------
logger.info(f"[cyan]Transcribing {INPUT_FILE}...[/cyan]")
audio = whisperx.load_audio(AUDIO_FILE)
result = model.transcribe(audio)

# ------------------------------------------------------------
# Word alignment
# ------------------------------------------------------------
logger.info("[cyan]Aligning words...[/cyan]")
model_a, metadata = whisperx.load_align_model(result["language"], device)
result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)

# ------------------------------------------------------------
# Speaker diarization (chunked)
# ------------------------------------------------------------
logger.info("[cyan]Running speaker diarization in chunks...[/cyan]")
chunks = split_audio(AUDIO_FILE, chunk_length_ms=5*60*1000)  # 5 min
speaker_segments = []

for i, ch in enumerate(chunks):
    logger.info(f"[blue]Processing chunk {i+1}/{len(chunks)}[/blue]")
    diarization = diarization_pipeline(ch)
    for seg, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": seg.start + (i * 300),  # shift by chunk start (seconds)
            "end": seg.end + (i * 300),
            "speaker": speaker
        })
    os.remove(ch)  # cleanup

# ------------------------------------------------------------
# Map speakers to words
# ------------------------------------------------------------
final_output = []
for word in result_aligned["word_segments"]:
    speaker = "unknown"
    for seg in speaker_segments:
        if seg["start"] <= word["start"] <= seg["end"]:
            speaker = seg["speaker"]
            break
    final_output.append({
        "word": word["text"],
        "start": word["start"],
        "end": word["end"],
        "speaker": speaker
    })

logger.info("[green]✅ Transcription complete![/green]")

# ------------------------------------------------------------
# Print transcript grouped by speaker
# ------------------------------------------------------------
current_speaker = None
line = []
for w in final_output:
    if w["speaker"] != current_speaker:
        if line:
            print(f"{current_speaker}: {' '.join(line)}")
        current_speaker = w["speaker"]
        line = []
    line.append(w["word"])
if line:
    print(f"{current_speaker}: {' '.join(line)}")

# Cleanup
if AUDIO_FILE != INPUT_FILE and os.path.exists(AUDIO_FILE):
    os.remove(AUDIO_FILE)
    logger.info(f"[yellow]🧹 Deleted temporary file: {AUDIO_FILE}[/yellow]")
