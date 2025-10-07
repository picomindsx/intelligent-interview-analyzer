import os
import torch
import whisperx
from pyannote.audio import Pipeline
from .logger import logger
from .loader import loading_animation

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("❌ HF_TOKEN is missing. Run: export HF_TOKEN=your_token")

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

logger.info("[green]Initializing models...[/green]")

# Load WhisperX
def load_whisper_model():
    return whisperx.load_model("small", device, compute_type=compute_type)

model = loading_animation("Loading WhisperX...", load_whisper_model)

# Load Pyannote diarization
def load_diarization_pipeline():
    return Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)

diarization_pipeline = loading_animation("Loading Pyannote...", load_diarization_pipeline)


def process_audio(audio_path: str):
    logger.info("[blue]Starting transcription...[/blue]")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio)

    logger.info("[cyan]Aligning words...[/cyan]")
    model_a, metadata = whisperx.load_align_model(result["language"], device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)

    logger.info("[magenta]Running speaker diarization...[/magenta]")
    diarization = diarization_pipeline(audio_path)

    # Map speakers
    speaker_segments = [
        {"start": seg.start, "end": seg.end, "speaker": speaker}
        for seg, _, speaker in diarization.itertracks(yield_label=True)
    ]

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

    logger.info("[green]Transcription complete![/green]")
    return final_output
