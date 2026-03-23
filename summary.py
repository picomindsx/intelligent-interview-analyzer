import datetime
import time
import argparse
import logging

import os
from dotenv import load_dotenv
load_dotenv()

from db import InterviewDB
from gemini_client import (
    get_questions_and_answers_as_summary, 
    extract_common_question_answers_as_summary,
    generate_overall_summary_text
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def format_time(secs):
    return str(datetime.timedelta(seconds=round(secs)))

def validate_qa_pairs(qa_items):
    valid_items = []
    for item in qa_items or []:
        if not isinstance(item, dict):
            continue
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()
        if question and answer:
            valid_items.append({"question": question, "answer": answer})
    return valid_items


def parse_args():
    parser = argparse.ArgumentParser(description="Generate interview summary and save it to DB.")
    parser.add_argument(
        "--transcript",
        default=os.getenv("TRANSCRIPT_FILE", "Aaron_Allen_TOP_Intake.transcript.txt"),
        help="Path to transcript text file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db = InterviewDB("interview_analyzer.db")
    start_all = time.perf_counter()

    try:
        transcript_file = args.transcript
        if not os.path.exists(transcript_file):
            raise FileNotFoundError(f"{transcript_file} not found")

        with open(transcript_file, "r", encoding="utf-8") as f:
            transcription = f.read().strip()

        if not transcription:
            raise ValueError("Transcription file is empty")

        logger.info("Transcription loaded successfully from %s", transcript_file)

        t1 = time.perf_counter()
        logger.info("Extracting question-answer pairs...")
        qa_output = validate_qa_pairs(get_questions_and_answers_as_summary(transcription) or [])
        logger.info("Q&A extraction complete in %.2fs", time.perf_counter() - t1)

        t2 = time.perf_counter()
        logger.info("Extracting common question answers...")
        common_answers = validate_qa_pairs(
            extract_common_question_answers_as_summary(transcription) or []
        )
        logger.info("Common question extraction complete in %.2fs", time.perf_counter() - t2)

        t3 = time.perf_counter()
        logger.info("Generating overall summary...")
        summary = generate_overall_summary_text(transcription, qa_output)
        logger.info("Summary generated in %.2fs", time.perf_counter() - t3)

        t4 = time.perf_counter()
        logger.info("Saving results to database...")
        db.save_interview_results_safe(
            filename=transcript_file,
            summary=summary,
            qa_general=qa_output,
            qa_common=common_answers,
        )
        logger.info("Saved in %.2fs", time.perf_counter() - t4)

        total_time = time.perf_counter() - start_all
        logger.info("-------- PERFORMANCE SUMMARY --------")
        logger.info("TOTAL TIME: %.2fs", total_time)
        logger.info("TOTAL TIME (rounded): %s", format_time(total_time))
        logger.info("Complete.")
    finally:
        db.close()


if __name__ == "__main__":
    main()