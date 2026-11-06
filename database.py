# database.py
import sqlite3
from config import DB_PATH

# Connect to local SQLite file (check_same_thread=False for FastAPI)
conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)

# Only these platforms are allowed. Anything else raises an error instead of
# being silently filed under WhatsApp.
PLATFORM_COLUMNS = {
    "telegram": "telegram_id",
    "whatsapp": "whatsapp_id",
}


def init_db():
    """Creates the foundational tables for the AI's memory (safe to run every start)."""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS contacts (
            contact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT DEFAULT 'Unknown',
            telegram_id TEXT UNIQUE,
            whatsapp_id TEXT UNIQUE,
            relationship_tier TEXT DEFAULT 'stranger'
        )
    ''')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            contact_id INTEGER,
            role TEXT,
            content TEXT,
            app_source TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(contact_id) REFERENCES contacts(contact_id)
        )
    ''')
    conn.commit()
    print("Database schema verified and active.")


def get_or_create_contact(app_source: str, sender_id: str) -> int:
    """Finds an existing contact by their app ID, or creates a new one."""
    column = PLATFORM_COLUMNS.get(app_source)
    if column is None:
        raise ValueError(f"Unsupported app_source: {app_source!r}")

    # Aman99 and aman99 are the same person
    handle = sender_id.strip().lower()

    row = conn.execute(
        f"SELECT contact_id FROM contacts WHERE lower({column}) = ?", (handle,)
    ).fetchone()
    if row:
        return row[0]

    cur = conn.execute(f"INSERT INTO contacts ({column}) VALUES (?)", (handle,))
    conn.commit()
    return cur.lastrowid


def save_message(contact_id: int, role: str, content: str, app_source: str):
    """Saves a single message to the timeline."""
    conn.execute('''
        INSERT INTO messages (contact_id, role, content, app_source)
        VALUES (?, ?, ?, ?)
    ''', (contact_id, role, content, app_source))
    conn.commit()


def get_chat_history(contact_id: int, limit: int = 5) -> str:
    """Fetches the NEWEST N messages (oldest of those first) for the LLM prompt."""
    rows = conn.execute('''
        SELECT role, content FROM (
            SELECT message_id, role, content FROM messages
            WHERE contact_id = ?
            ORDER BY message_id DESC
            LIMIT ?
        )
        ORDER BY message_id ASC
    ''', (contact_id, limit)).fetchall()

    if not rows:
        return ""

    history = ""
    for role, content in rows:
        history += f"{role.capitalize()}: {content}\n"
    return history


if __name__ == "__main__":
    init_db()