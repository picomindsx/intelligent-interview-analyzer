import google.generativeai as genai
import json
import os

# Load Gemini API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

    # print(json.dumps(data, indent=2))
    return data

def get_questions_and_answers_as_summary(transcript: str):
    """
    Extracts structured Q&A pairs from an interview transcript.
    Also summarizes the quations and answers to provide a concise overview of the interviewee's responses.
    Returns [{"question": "...", "answer": "..."}].
    """

    prompt = f"""
    You are a transcript analyzer.
    Identify the QUESTIONS and corresponding ANSWERS from the following transcript.
    Also summarizes the quations and answers to provide a concise overview of the interviewee's responses.
    Assume "Speaker A" is the interviewer and "Speaker B" is the interviewee unless context clearly shows otherwise.
    
    Output ONLY valid JSON in this format:
    [
      {{ "question": "What inspired you to start this project?", "answer": "I was always passionate about sustainability...", "summary": "..." }},
      {{ "question": "How long did it take to complete?", "answer": "Around six months.", "summary": "..." }}
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

    # print(json.dumps(data, indent=2))
    return data


def generate_overall_summary(speaker_segments, qna_pairs):
    full_text = "\n".join(
        [f"{s['speaker']}: {s['text']}" for s in speaker_segments]
    )

    prompt = f"""
    Summarize the following meeting transcript in a clear and concise way.
    Mention:
    - What the meeting was about
    - Main topics discussed
    - Key decisions or outcomes
    - Any follow-ups or next steps if mentioned

    Transcript:
    {full_text}

    Questions and Answers:
    {json.dumps(qna_pairs, indent=2)}
    """

    return get_summary(prompt)

# ------------------------------------------------------
# HELPER FUNCTION: get_summary
# ------------------------------------------------------
def get_summary(prompt: str) -> str:
    """
    Sends a summarization or analysis prompt to Gemini and returns the raw text output.

    Parameters:
        prompt (str): The text prompt you want Gemini to process (summary, analysis, etc.)

    Returns:
        str: The cleaned, human-readable text response.
    """

    try:
        # Use Gemini 2.5 Flash (fast and accurate for summarization)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        text = response.text.strip()

        if not text:
            raise ValueError("Empty response from Gemini.")

        return text

    except Exception as e:
        print(f"⚠️ Error during Gemini summary generation: {e}")
        return "Summary generation failed. Please check logs or prompt."

def extract_common_question_answers(transcript: str):
    """
    Extract answers for the standard/common interview questions.
    Uses Gemini to identify responses in flexible natural language contexts.

    Returns a list of:
    [
      { "question": "...", "answer": "..." }
    ]
    """

    # These are the consistent questions across all 3 interviews
    common_questions = [
        "What is your name?",
        "What is your date of birth?",
        "What is your highest level of education?",
        "What challenges did you face in school?",
        "Do you want to achieve your GED or high school diploma?",
        "How far are you from achieving it?",
        "What are your hobbies or things you love to do?",
        "What did you come to the Opportunity Project for?",
        "What are your first thoughts about the program?",
        "Do you have career goals?",
        "How likely do you think you are to achieve your goal?",
        "Do you know the steps to get there?",
        "What do you want to achieve in the next 3 weeks?",
        "What challenges can we help you break down?",
        "Do you have reliable transportation?",
        "Are there financial challenges affecting your goals?",
        "How do you learn best?",
        "Have you ever worked a job before?",
        "What did you learn from that job?",
        "Have you been in any legal trouble or have a criminal record?",
        "Are there personal or family challenges affecting school or work?",
        "Do you face any mental-health or environmental challenges?"
    ]

    prompt = f"""
    Extract answers from the transcript for the following common interview questions.
    If no answer is found, set answer to an empty string "".
    The answers must reflect ONLY what the interviewee said.

    Common questions:
    {json.dumps(common_questions, indent=2)}

    Transcript:
    {transcript}

    Output ONLY valid JSON in the following format:
    [
      {{ "question": "What is your name?", "answer": "..." }},
      {{ "question": "What is your date of birth?", "answer": "..." }}
    ]
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    result = response.text.strip()

    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        start = result.find("[")
        end = result.rfind("]") + 1
        data = json.loads(result[start:end])

    print(json.dumps(data, indent=2))
    return data

def extract_common_question_answers_as_summary(transcript: str):
    """
    Extract answers for the standard/common interview questions.
    Also summarizes the quations answers to provide a concise overview of the interviewee's responses.
    Uses Gemini to identify responses in flexible natural language contexts.

    Returns a list of:
    [
      { "question": "...", "answer": "...", summary: "..." }
    ]
    """

    # These are the consistent questions across all 3 interviews
    common_questions = [
        "What is your name?",
        "What is your date of birth?",
        "What is your highest level of education?",
        "What challenges did you face in school?",
        "Do you want to achieve your GED or high school diploma?",
        "How far are you from achieving it?",
        "What are your hobbies or things you love to do?",
        "What did you come to the Opportunity Project for?",
        "What are your first thoughts about the program?",
        "Do you have career goals?",
        "How likely do you think you are to achieve your goal?",
        "Do you know the steps to get there?",
        "What do you want to achieve in the next 3 weeks?",
        "What challenges can we help you break down?",
        "Do you have reliable transportation?",
        "Are there financial challenges affecting your goals?",
        "How do you learn best?",
        "Have you ever worked a job before?",
        "What did you learn from that job?",
        "Have you been in any legal trouble or have a criminal record?",
        "Are there personal or family challenges affecting school or work?",
        "Do you face any mental-health or environmental challenges?"
    ]

    prompt = f"""
    Extract answers from the transcript for the following common interview questions.
    If no answer is found, set answer to an empty string "".
    The answers must reflect ONLY what the interviewee said.
    Also summarizes the quations and answers to provide a concise overview of the interviewee's responses.

    Common questions:
    {json.dumps(common_questions, indent=2)}

    Transcript:
    {transcript}

    Output ONLY valid JSON in the following format:
    [
      {{ "question": "What is your name?", "answer": "...", "summary": "..." }},
      {{ "question": "What is your date of birth?", "answer": "...", summary: "..." }}
    ]
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    result = response.text.strip()

    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        start = result.find("[")
        end = result.rfind("]") + 1
        data = json.loads(result[start:end])

    print(json.dumps(data, indent=2))
    return data
