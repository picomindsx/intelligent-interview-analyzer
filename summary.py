import datetime
import time
import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
load_dotenv()

from db import InterviewDB
from gemini_client import (
    get_questions_and_answers_as_summary, 
    extract_common_question_answers_as_summary,
    generate_overall_summary_text
)

# ---------------------------------
# INIT
# ---------------------------------
db = InterviewDB("interview_analyzer.db")
start_all = time.time()

def format_time(secs):
    return str(datetime.timedelta(seconds=round(secs)))

# ---------------------------------
# LOAD TRANSCRIPTION
# ---------------------------------
TRANSCRIPT_FILE = "Aaron_Allen_TOP_Intake.transcript.txt"

if not os.path.exists(TRANSCRIPT_FILE):
    raise FileNotFoundError(f"{TRANSCRIPT_FILE} not found")

with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
    transcription = f.read().strip()

if not transcription:
    raise ValueError("Transcription file is empty")

print("\n📄 Transcription loaded successfully.")

# ---------------------------------
# STEP 1: Extract General Q&A
# ---------------------------------
t1 = time.time()
print("\n💬 Extracting question–answer pairs...")
qa_output = get_questions_and_answers_as_summary(transcription) or []
print(f"✅ Q&A extraction complete in {time.time() - t1:.2f}s")

# ---------------------------------
# STEP 2: Extract Common Q&A
# ---------------------------------
t2 = time.time()
print("\n🔍 Extracting common question answers...")
common_answers = extract_common_question_answers_as_summary(transcription) or []
print(f"✅ Common question extraction complete in {time.time() - t2:.2f}s")

# ---------------------------------
# STEP 3: Generate Overall Summary
# ---------------------------------
t3 = time.time()
print("\n🧠 Generating overall summary...")
summary = generate_overall_summary_text(transcription, qa_output)
print(f"✅ Summary generated in {time.time() - t3:.2f}s")

# ---------------------------------
# STEP 4: Save to DB
# ---------------------------------
t4 = time.time()
print("\n💾 Saving results to database...")

db.save_interview_results_safe(
    filename=TRANSCRIPT_FILE,
    summary=summary,
    qa_general=qa_output,
    qa_common=common_answers
)

print(f"✅ Saved in {time.time() - t4:.2f}s")

# ---------------------------------
# PERFORMANCE SUMMARY
# ---------------------------------
total_time = time.time() - start_all

print("\n📊 -------- PERFORMANCE SUMMARY --------")
print(f"🕒 TOTAL TIME: {total_time:.2f}s")
print("----------------------------------------")
print("✅ Complete!")

db.close()