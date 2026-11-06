# app.py
import os
import time
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from config import STATUS_FILE, RAG_FOLDER, DB_PATH

# Biggest note file you can upload (the model only has a 2048-token window)
MAX_UPLOAD_BYTES = 10_000

# Ensure the exclusive RAG folder exists
RAG_FOLDER.mkdir(exist_ok=True)


def read_status() -> str:
    try:
        return STATUS_FILE.read_text(encoding="utf-8").strip().upper()
    except FileNotFoundError:
        return "AVAILABLE"


def write_status(value: str) -> None:
    """Writes to a temp file first, then swaps it in, so the server never
    reads a half-written (empty) status file."""
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(value, encoding="utf-8")
    for _ in range(5):
        try:
            os.replace(tmp, STATUS_FILE)
            return
        except PermissionError:  # Windows: the server may be reading it this instant
            time.sleep(0.1)
    raise PermissionError("Could not update the status file. Try again.")


# Ensure the status toggle file exists (fresh copies start silent)
if not STATUS_FILE.exists():
    write_status("AVAILABLE")

st.set_page_config(page_title="AI Proxy Dashboard", page_icon="🤖", layout="wide")
st.title("🤖 AI Proxy Control Center")

# ==========================================
# 1. THE STATUS TOGGLE & 5. FACT INJECTOR (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("⚙️ 1. Proxy Status")
    current_status = read_status()

    new_status = st.radio(
        "Set AI Auto-Reply Status:",
        options=["AVAILABLE", "BUSY"],
        index=1 if current_status == "BUSY" else 0,
        help="BUSY: AI replies. AVAILABLE: AI stays quiet."
    )

    if new_status != current_status:
        write_status(new_status)
        st.success(f"Status updated to {new_status}")

    st.markdown("---")
    st.header("💉 5. Quick Fact Injector")
    st.markdown("Temporarily inject a manual status without uploading a file.")
    injected_fact = st.text_area("Example: 'I am driving right now'")

    if st.button("Inject Fact"):
        (RAG_FOLDER / "manual_injected_fact.txt").write_text(injected_fact, encoding="utf-8")
        st.success("Fact successfully injected into memory!")

# ==========================================
# MAIN DASHBOARD AREA
# ==========================================
col1, col2 = st.columns(2)

with col1:
    # ------------------------------------------
    # 2. RAG FILE MANAGER (EXCLUSIVE FOLDER)
    # ------------------------------------------
    st.header("📂 2. RAG File Manager")
    st.markdown("Manage the exclusive folder. The AI will read these files to answer factual questions.")

    # Only .txt for now: the server ignores PDFs until PDF-to-text is added
    uploaded_file = st.file_uploader("Upload Timetable, Syllabus, or Notes (.txt)", type=["txt"])
    if uploaded_file is not None:
        # Keep only the file name, so a name like ../x cannot leave the folder
        safe_name = Path(uploaded_file.name).name
        if uploaded_file.size > MAX_UPLOAD_BYTES:
            st.error(f"{safe_name} is too big ({uploaded_file.size} bytes). Limit: {MAX_UPLOAD_BYTES} bytes.")
        else:
            (RAG_FOLDER / safe_name).write_bytes(uploaded_file.getbuffer())
            st.success(f"Saved: {safe_name}")

    # Managing existing files
    st.subheader("Files currently in memory:")
    files = sorted(p.name for p in RAG_FOLDER.iterdir() if p.is_file())
    if not files:
        st.info("The exclusive memory folder is currently empty.")
    else:
        for file in files:
            f_col1, f_col2 = st.columns([0.8, 0.2])
            f_col1.write(f"📄 {file}")
            if f_col2.button("Delete", key=file):
                (RAG_FOLDER / file).unlink()
                st.rerun()

with col2:
    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH))

        # ------------------------------------------
        # 3. LIVE MEMORY FEED
        # ------------------------------------------
        st.header("💬 3. Live Memory Feed")
        st.markdown("Recent cross-platform messages.")

        messages_df = pd.read_sql_query('''
            SELECT m.timestamp, c.name, m.app_source, m.role, m.content
            FROM messages m
            JOIN contacts c ON m.contact_id = c.contact_id
            ORDER BY m.message_id DESC LIMIT 8
        ''', conn)

        if not messages_df.empty:
            st.dataframe(messages_df, use_container_width=True, hide_index=True)
        else:
            st.info("No messages recorded in the database yet.")

        # ------------------------------------------
        # 4. ACTIVE PERSONA VIEWER
        # ------------------------------------------
        st.header("👥 4. Active Personas")
        st.markdown("Saved contacts mapped to custom styles.")

        contacts_df = pd.read_sql_query('''
            SELECT name, telegram_id, whatsapp_id, relationship_tier
            FROM contacts
        ''', conn)

        if not contacts_df.empty:
            st.dataframe(contacts_df, use_container_width=True, hide_index=True)
        else:
            st.info("No contacts saved yet.")

    except Exception as e:
        st.warning(f"Could not read the database. Start server.py once so it creates the tables. Error: {e}")
    finally:
        if conn is not None:
            conn.close()