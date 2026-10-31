import sqlite3

# Connect to local SQLite file (check_same_thread=False for FastAPI)
conn = sqlite3.connect("proxy_memory.db", check_same_thread=False)
cursor = conn.cursor()

def init_db():
    """Creates the foundational tables for the AI's memory."""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contacts (
            contact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT DEFAULT 'Unknown',
            telegram_id TEXT UNIQUE,
            whatsapp_id TEXT UNIQUE,
            relationship_tier TEXT DEFAULT 'stranger'
        )
    ''')
    
    cursor.execute('''
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
    column = "telegram_id" if app_source == "telegram" else "whatsapp_id"
    
    cursor.execute(f"SELECT contact_id FROM contacts WHERE {column} = ?", (sender_id,))
    result = cursor.fetchone()
    
    if result:
        return result[0]
        
    cursor.execute(f"INSERT INTO contacts ({column}) VALUES (?)", (sender_id,))
    conn.commit()
    return cursor.lastrowid

def save_message(contact_id: int, role: str, content: str, app_source: str):
    """Saves a single message to the timeline."""
    cursor.execute('''
        INSERT INTO messages (contact_id, role, content, app_source) 
        VALUES (?, ?, ?, ?)
    ''', (contact_id, role, content, app_source))
    conn.commit()

def get_chat_history(contact_id: int, limit: int = 5) -> str:
    """Fetches the last N messages to inject into the LLM prompt."""
    cursor.execute('''
        SELECT role, content FROM messages 
        WHERE contact_id = ? 
        ORDER BY timestamp ASC LIMIT ?
    ''', (contact_id, limit))
    
    rows = cursor.fetchall()
    if not rows:
        return ""
        
    history = ""
    for role, content in rows:
        history += f"{role.capitalize()}: {content}\n"
    return history

if __name__ == "__main__":
    init_db()