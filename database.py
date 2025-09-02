import sqlite3
import json
from datetime import datetime
import pickle
from typing import List
from langchain_core.messages import BaseMessage

DATABASE_NAME = "workflows.db"

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    # workflows table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workflows (
            session_id TEXT PRIMARY KEY,
            agent_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            details TEXT,
            status TEXT NOT NULL
        )
    """)
    # Chat-history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            session_id TEXT PRIMARY KEY,
            history BLOB
        )
    """)
    conn.commit()
    conn.close()

def create_workflow(session_id: str, agent_name: str):
    """Creates a new workflow record in the database."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO workflows (session_id, agent_name, created_at, status) VALUES (?, ?, ?, ?)",
            (session_id, agent_name, datetime.now().isoformat(), "Processing")
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # Session already exists, do nothing.
        pass
    finally:
        conn.close()

def update_workflow(session_id: str, status: str, details: dict):
    """Updates the status and details of an existing workflow."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE workflows SET status = ?, details = ? WHERE session_id = ?",
        (status, json.dumps(details), session_id)
    )
    conn.commit()
    conn.close()
    print(f"--- Workflow {session_id} updated: Status={status} ---")

def get_all_workflows():
    """Retrieves all workflow records for the overview page."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM workflows ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def save_history(session_id: str, history: List[BaseMessage]):
    """Saves the chat history for a session."""
    with sqlite3.connect(DATABASE_NAME) as conn:
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO chat_history (session_id, history) VALUES (?, ?)", (session_id, pickle.dumps(history)))
        conn.commit()

def get_history(session_id: str) -> List[BaseMessage]:
    """Retrieves the chat history for a session."""
    with sqlite3.connect(DATABASE_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT history FROM chat_history WHERE session_id=?", (session_id,))
        row = c.fetchone()
        return pickle.loads(row[0]) if row else []

def workflow_exists(session_id: str) -> bool:
    """Checks if a workflow already exists for a given session_id."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM workflows WHERE session_id = ?", (session_id,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists