import sqlite3
import json

class InterviewDB:
    def __init__(self, db_name="interviews.db"):
        self.conn = sqlite3.connect(db_name)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.create_tables()
        self.migrate_legacy_qa_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        # 1. Main Interview Metadata
        cursor.execute('''CREATE TABLE IF NOT EXISTS interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')

        # 2. Normalized Q&A storage for all question types
        cursor.execute('''CREATE TABLE IF NOT EXISTS qa_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id INTEGER NOT NULL,
            qa_type TEXT NOT NULL CHECK(qa_type IN ('general', 'common')),
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            FOREIGN KEY(interview_id) REFERENCES interviews(id) ON DELETE CASCADE
        )''')

        # Legacy tables are kept during transition window for backward compatibility.
        cursor.execute('''CREATE TABLE IF NOT EXISTS qa_general (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id INTEGER,
            question TEXT,
            answer TEXT,
            FOREIGN KEY(interview_id) REFERENCES interviews(id)
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS qa_common (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id INTEGER,
            question TEXT,
            answer TEXT,
            FOREIGN KEY(interview_id) REFERENCES interviews(id)
        )''')

        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_qa_pairs_interview_type
                          ON qa_pairs (interview_id, qa_type)''')
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_qa_pairs_question
                          ON qa_pairs (question)''')
        self.conn.commit()

    def migrate_legacy_qa_tables(self):
        cursor = self.conn.cursor()

        # Backfill legacy general rows into normalized table (idempotent).
        cursor.execute('''
            INSERT INTO qa_pairs (interview_id, qa_type, question, answer)
            SELECT g.interview_id, 'general', g.question, g.answer
            FROM qa_general g
            WHERE g.interview_id IS NOT NULL
              AND g.question IS NOT NULL
              AND g.answer IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM qa_pairs p
                  WHERE p.interview_id = g.interview_id
                    AND p.qa_type = 'general'
                    AND p.question = g.question
                    AND p.answer = g.answer
              )
        ''')

        # Backfill legacy common rows into normalized table (idempotent).
        cursor.execute('''
            INSERT INTO qa_pairs (interview_id, qa_type, question, answer)
            SELECT c.interview_id, 'common', c.question, c.answer
            FROM qa_common c
            WHERE c.interview_id IS NOT NULL
              AND c.question IS NOT NULL
              AND c.answer IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM qa_pairs p
                  WHERE p.interview_id = c.interview_id
                    AND p.qa_type = 'common'
                    AND p.question = c.question
                    AND p.answer = c.answer
              )
        ''')
        self.conn.commit()

    def _normalize_qa_rows(self, interview_id, qa_items, qa_type):
        if not qa_items:
            return []
        normalized_rows = []
        for item in qa_items:
            if not isinstance(item, dict):
                continue
            question = (item.get("question") or "").strip()
            answer = (item.get("answer") or "").strip()
            if question and answer:
                normalized_rows.append((interview_id, qa_type, question, answer))
        return normalized_rows

    def save_interview_results(self, filename, summary, qa_general, qa_common):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO interviews (filename, summary) VALUES (?, ?)",
                (filename or "unknown", summary or "")
            )
            interview_id = cursor.lastrowid

            gen_rows = self._normalize_qa_rows(interview_id, qa_general, "general")
            com_rows = self._normalize_qa_rows(interview_id, qa_common, "common")
            qa_rows = gen_rows + com_rows

            if qa_rows:
                cursor.executemany(
                    "INSERT INTO qa_pairs (interview_id, qa_type, question, answer) VALUES (?,?,?,?)",
                    qa_rows
                )
        print(f"✅ Saved Interview {interview_id}: {len(gen_rows)} general and {len(com_rows)} common pairs.")

    def save_interview_results_safe(self, filename=None, summary=None, qa_general=None, qa_common=None):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO interviews (filename, summary) VALUES (?, ?)",
                (filename or "unknown", summary or "")
            )
            interview_id = cursor.lastrowid

            gen_rows = self._normalize_qa_rows(interview_id, qa_general, "general")
            com_rows = self._normalize_qa_rows(interview_id, qa_common, "common")
            qa_rows = gen_rows + com_rows

            if qa_rows:
                cursor.executemany(
                    "INSERT INTO qa_pairs (interview_id, qa_type, question, answer) VALUES (?,?,?,?)",
                    qa_rows
                )

        print(f"✅ Saved Interview {interview_id}")

    def get_interview_qa(self, interview_id, qa_type=None):
        cursor = self.conn.cursor()
        if qa_type:
            cursor.execute(
                "SELECT question, answer FROM qa_pairs WHERE interview_id = ? AND qa_type = ?",
                (interview_id, qa_type)
            )
        else:
            cursor.execute(
                "SELECT qa_type, question, answer FROM qa_pairs WHERE interview_id = ?",
                (interview_id,)
            )
        return cursor.fetchall()

    def get_interview_qa_grouped(self, interview_id):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT qa_type, question, answer FROM qa_pairs WHERE interview_id = ?",
            (interview_id,)
        )
        grouped = {"general": [], "common": []}
        for qa_type, question, answer in cursor.fetchall():
            grouped.setdefault(qa_type, []).append({"question": question, "answer": answer})
        return grouped

    def close(self):
        self.conn.close()