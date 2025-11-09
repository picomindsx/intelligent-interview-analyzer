import google.generativeai as genai
import json
import os

# Load Gemini API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(f"🔑 {GEMINI_API_KEY}")

if not GEMINI_API_KEY:
    raise EnvironmentError("Please set GEMINI_API_KEY as an environment variable.")

genai.configure(api_key=GEMINI_API_KEY)

def get_diarization(transcript: str):
    """
    Uses Gemini 1.5 Flash to split a conversation transcript into speaker turns.
    Returns a list of {"speaker": "Speaker A", "text": "..."} dictionaries.
    """

    prompt = f"""
    Split the following interview transcript into clear speaker turns.
    Label each as "Speaker A", "Speaker B", etc.
    Output ONLY valid JSON in this exact format:
    [
      {{ "speaker": "Speaker A", "text": "..." }},
      {{ "speaker": "Speaker B", "text": "..." }}
    ]

    Transcript:
    {transcript}
    """

    for m in genai.list_models():
      if "generateContent" in m.supported_generation_methods:
          print(m.name)

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    result = response.text.strip()

    # Try to extract valid JSON if model includes extra explanation
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        # Sometimes Gemini wraps JSON in text; attempt cleanup
        start = result.find("[")
        end = result.rfind("]") + 1
        if start != -1 and end != -1:
            json_str = result[start:end]
            data = json.loads(json_str)
        else:
            raise ValueError("Gemini did not return valid JSON:\n" + result)

    print(json.dumps(data, indent=2))
    return data

def get_questions_and_answers(transcript: str):
    """
    Extracts structured Q&A pairs from an interview transcript.
    Returns [{"question": "...", "answer": "..."}].
    """

    prompt = f"""
    You are a transcript analyzer.
    Identify the QUESTIONS and corresponding ANSWERS from the following transcript.
    Assume "Speaker A" is the interviewer and "Speaker B" is the interviewee unless context clearly shows otherwise.
    
    Output ONLY valid JSON in this format:
    [
      {{ "question": "What inspired you to start this project?", "answer": "I was always passionate about sustainability..." }},
      {{ "question": "How long did it take to complete?", "answer": "Around six months." }}
    ]

    Transcript:
    {transcript}
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    result = response.text.strip()

    # Extract valid JSON
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        start = result.find("[")
        end = result.rfind("]") + 1
        data = json.loads(result[start:end])

    print(json.dumps(data, indent=2))
    return data