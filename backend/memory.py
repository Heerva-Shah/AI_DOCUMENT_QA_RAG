# backend/memory.py

import sqlite3
import json
import os

DB_PATH = "./conversation_history.db"


def init_db():
    """Create the conversation history table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def get_history(session_id: str) -> list[dict]:
    """Get last 5 exchanges (10 messages) for a session."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content FROM conversation_history
        WHERE session_id = ?
        ORDER BY timestamp ASC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()

    # Get last 10 messages (5 user + 5 assistant = 5 exchanges)
    last_10 = rows[-10:] if len(rows) > 10 else rows
    return [{"role": row[0], "content": row[1]} for row in last_10]


def save_message(session_id: str, role: str, content: str):
    """Save a single message to history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversation_history (session_id, role, content)
        VALUES (?, ?, ?)
    """, (session_id, role, content))
    conn.commit()
    conn.close()


def clear_history(session_id: str):
    """Clear history for a session (called when new document uploaded)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversation_history WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


# Initialize DB when module is imported
init_db()