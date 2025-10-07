import tempfile
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from .services import process_audio

router = APIRouter()

@router.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    result = process_audio(audio_path)
    return JSONResponse(content={"transcription": result})
