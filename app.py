import streamlit as st
import os
import sqlite3
import pandas as pd

# --- Configuration & Paths ---
STATUS_FILE = "proxy_status.txt"
RAG_FOLDER = "rag_memory"
DB_PATH = "proxy_memory.db"

# Ensure the exclusive RAG folder exists
if not os.path.exists(RAG_FOLDER):
    os.makedirs(RAG_FOLDER)

# Ensure the status toggle file exists
if not os.path.exists(STATUS_FILE):
    with open(STATUS_FILE, "w") as f:
        f.write("AVAILABLE")

st.set_page_config(page_title="AI Proxy Dashboard", page_icon="🤖", layout="wide")
st.title("🤖 AI Proxy Control Center")

# ==========================================
# 1. THE STATUS TOGGLE & 5. FACT INJECTOR (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("⚙️ 1. Proxy Status")
    with open(STATUS_FILE, "r") as f:
        current_status = f.read().strip()

    new_status = st.radio(
        "Set AI Auto-Reply Status:",
        options=["AVAILABLE", "BUSY"],
        index=0 if current_status == "AVAILABLE" else 1,
        help="BUSY: AI replies. AVAILABLE: AI stays quiet."
    )

    if new_status != current_status:
        with open(STATUS_FILE, "w") as f:
            f.write(new_status)
        st.success(f"Status updated to {new_status}")

    st.markdown("---")
    st.header("💉 5. Quick Fact Injector")
    st.markdown("Temporarily inject a manual status without uploading a file.")
    injected_fact = st.text_area("Example: 'I am driving right now'")
    
    if st.button("Inject Fact"):
        with open(os.path.join(RAG_FOLDER, "manual_injected_fact.txt"), "w") as f:
            f.write(injected_fact)
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
    
    # Uploading to the exclusive folder
    uploaded_file = st.file_uploader("Upload Timetable, Syllabus, or Notes (.txt, .pdf)")
    if uploaded_file is not None:
        file_path = os.path.join(RAG_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved: {uploaded_file.name}")
        
    # Managing existing files
    st.subheader("Files currently in memory:")
    files = os.listdir(RAG_FOLDER)
    if not files:
        st.info("The exclusive memory folder is currently empty.")
    else:
        for file in files:
            f_col1, f_col2 = st.columns([0.8, 0.2])
            f_col1.write(f"📄 {file}")
            if f_col2.button("Delete", key=file):
                os.remove(os.path.join(RAG_FOLDER, file))
                st.rerun()

with col2:
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # ------------------------------------------
        # 3. LIVE MEMORY FEED
        # ------------------------------------------
        st.header("💬 3. Live Memory Feed")
        st.markdown("Recent cross-platform messages.")
        
        messages_df = pd.read_sql_query('''
            SELECT m.timestamp, c.name, m.app_source, m.role, m.content 
            FROM messages m
            JOIN contacts c ON m.contact_id = c.contact_id
            ORDER BY m.timestamp DESC LIMIT 8
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
            
        conn.close()
    except Exception as e:
        st.warning(f"Could not connect to database. Ensure your proxy server is running. Error: {e}")