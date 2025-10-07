```
brew install ffmpeg
```

```
python3 -m venv venv
source venv/bin/activate
```

```
pip install --upgrade pip
pip install fastapi uvicorn python-multipart torch librosa
pip install git+https://github.com/m-bain/whisperx.git
pip install pyannote.audio
```

```
pip install -r requirements.txt
```

```
uvicorn main:app --reload
```

[pyannote segmentation - accept terms](https://huggingface.co/pyannote/segmentation)  
[pyannote speaker-diarization - accept terms](https://huggingface.co/pyannote/speaker-diarization)
