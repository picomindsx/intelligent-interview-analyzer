import sqlite3
import json

class InterviewDB:
    def __init__(self, db_name="interviews.db"):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        # 1. Main Interview Metadata
        cursor.execute('''CREATE TABLE IF NOT EXISTS interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            summary TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # 2. General Q&A (All questions found in transcript)
        cursor.execute('''CREATE TABLE IF NOT EXISTS qa_general (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id INTEGER,
            question TEXT,
            answer TEXT,
            FOREIGN KEY(interview_id) REFERENCES interviews(id)
        )''')

        # 3. Common/Intake Q&A (The 22 specific intake questions)
        cursor.execute('''CREATE TABLE IF NOT EXISTS qa_common (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id INTEGER,
            question TEXT,
            answer TEXT,
            FOREIGN KEY(interview_id) REFERENCES interviews(id)
        )''')
        self.conn.commit()

    def save_interview_results(self, filename, summary, qa_general, qa_common):
        cursor = self.conn.cursor()
        
        # Insert Interview record
        cursor.execute("INSERT INTO interviews (filename, summary) VALUES (?, ?)", 
                       (filename, summary))
        interview_id = cursor.lastrowid

        # Insert General QA
        gen_rows = [(interview_id, q['question'], q['answer']) for q in qa_general]
        cursor.executemany("INSERT INTO qa_general (interview_id, question, answer) VALUES (?,?,?)", gen_rows)

        # Insert Common/Intake QA
        com_rows = [(interview_id, q['question'], q['answer']) for q in qa_common]
        cursor.executemany("INSERT INTO qa_common (interview_id, question, answer) VALUES (?,?,?)", com_rows)

        self.conn.commit()
        print(f"✅ Saved Interview {interview_id}: {len(gen_rows)} general and {len(com_rows)} common pairs.")
        
    def close(self):
        self.conn.close()