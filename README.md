# Project EchoText 🤖
**An Offline, Omnichannel AI Messaging Proxy**

Project EchoText is a fully offline, privacy-first AI messaging proxy engineered to intercept incoming messages and generate autonomous, persona-aligned replies on your local machine. Designed specifically for low-end edge hardware, it prevents notification fatigue by acting as your digital proxy when you are busy, studying, or asleep—all without sending your private chats to cloud servers.

## ✨ Key Features
*   **100% Offline Privacy:** Executes all natural language processing locally. Zero API calls, zero cloud dependencies, and zero data harvesting.
*   **Lightning RAG Router:** Dynamically injects temporal logic (e.g., current time) and custom user facts into the AI's context window to prevent hallucinations.
*   **Unified Memory:** Uses an embedded SQLite database to map multiple app handles (e.g., Telegram, WhatsApp) to a single contact, maintaining seamless cross-platform conversational history.
*   **Simulated Native Typing:** App Interceptors natively capture messages and type back responses with randomized human-like delays.
*   **Control Center UI:** A centralized Streamlit dashboard to toggle your active proxy status (BUSY/AVAILABLE) and manage your persona rules.

## 🛠️ Technology Stack
*   **Backend API:** FastAPI (managed by Uvicorn for asynchronous routing)
*   **Inference Engine:** `llama.cpp` (optimized for CPU/Edge execution)
*   **Language Model:** 4-bit Quantized GGUF Llama-3.2-3B-Instruct (fine-tuned via QLoRA/unsloth)
*   **Database:** SQLite 
*   **User Interface:** Streamlit
*   **Interceptors:** Custom Python automation scripts

## ⚙️ Prerequisites
*   Windows 10/11, macOS, or Linux
*   Python 3.10 or higher
*   Minimum 4GB RAM (optimized for Intel Core i3 or equivalent)
*   A quantized `.gguf` model file

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/abhayrajjain/Project-EchoText-Core.git
cd Project-EchoText-Core
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
# On Windows use: venv\Scripts\activate
# On Linux/macOS use: source venv/bin/activate
pip install -r requirements.txt
```

**3. Setup the AI Model**
*   Download your fine-tuned `llama-3.2-3b-instruct.Q4_K_M.gguf` model.
*   Place the `.gguf` file inside the `models/` directory in the project folder.

**4. Initialize the Database**
```bash
python setup_db.py
```

## 🎮 Usage
To run the full Project EchoText environment locally, you will need to start the three core modules in separate terminal windows.

**Terminal 1: Start the FastAPI Backend (Central Brain)**
```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

**Terminal 2: Launch the Control Center (Streamlit UI)**
```bash
streamlit run app_ui.py
```
*(This will open the UI in your web browser where you can set your status to BUSY and configure your facts.)*

**Terminal 3: Start the App Interceptor (Telegram Example)**
```bash
python interceptors/telegram_interceptor.py
```

## 🎓 Academic Disclaimer
This project was developed by Abhay Raj Jain and Aman Rajput as part of the B.Tech Computer Science Macro Project - I curriculum at Madhav Institute of Technology and Science (MITS), Gwalior.
