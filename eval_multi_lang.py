#!/usr/bin/env python3
"""Multilingual & Multi-Tone Benchmark Evaluation Harness for EchoText.

Evaluates local LLM performance across:
- Languages: English, Hindi (Devanagari), Hinglish (Romanized Hindi-English)
- Relationship Tiers: Professor, Colleague, Family, Friend, Acquaintance, Unknown
- Tone Styles: formal_english, polite_hinglish, warm_family, casual_hinglish, concise
- Scenarios: Grounded queries, Ungrounded context tests ([UNSURE]), Greetings, Academic/Work requests
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

# Ensure unbuffered / line-buffered stdout with utf-8 encoding for real-time progress logging
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

# Ensure EchoText modules are on sys.path whether run from project root or EchoText subdir
CURRENT_DIR = Path(__file__).resolve().parent
if (CURRENT_DIR / "server.py").exists():
    BASE_DIR = CURRENT_DIR
elif (CURRENT_DIR / "EchoText" / "server.py").exists():
    BASE_DIR = CURRENT_DIR / "EchoText"
else:
    BASE_DIR = CURRENT_DIR

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import MODEL_PATH, OWNER_NAME  # type: ignore[import-not-found]
from intelligence import (  # type: ignore[import-not-found]
    build_Safe_chat_messages,
    clean_llm_output,
    detect_language_style,
    violates_relation_or_tone,
)
from server import get_llm  # type: ignore[import-not-found]


_MODEL_INSTANCE: Any = None


def get_benchmark_llm() -> Any:
    """Initialize Llama with optimal thread count (8 threads) for fast CPU throughput."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE
    from llama_cpp import Llama

    cpu_threads = min(8, max(2, (os.cpu_count() or 4) // 2)) if (os.cpu_count() or 4) > 8 else 4
    _MODEL_INSTANCE = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,
        n_threads=cpu_threads,
        n_batch=512,
        verbose=False,
    )
    return _MODEL_INSTANCE


def query_model(
    messages: list[dict[str, str]],
    max_tokens: int = 72,
    temperature: float = 0.2,
) -> str:
    """Run chat completion with conservative max_tokens tailored for proxy messaging."""
    llm = get_benchmark_llm()
    if llm is None:
        raise RuntimeError("LLM is not loaded.")
    res = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=[
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
            "<|eot_id|>",
            "<|end_of_text|>",
            "\nUser:",
            "\nSender:",
            "\nRecipient:",
            "\nContact:",
        ],
    )
    choices = res.get("choices") or []
    if not choices:
        return ""
    raw = str(choices[0].get("message", {}).get("content") or "").strip()
    return clean_llm_output(raw)

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_INFORMAL_SLANG_RE = re.compile(
    r"\b(bhai|yaar|bro|dude|tu|tere|tera|teri|apna|chal|arrey|arre|nhi|hn|h\b|pta|kya re)\b",
    re.IGNORECASE,
)
_RESPECTFUL_HINDI_RE = re.compile(r"\b(आप|जी|नमस्ते|प्रणाम|कृपया|धन्यवाद)\b", re.UNICODE)
_RESPECTFUL_HINGLISH_RE = re.compile(r"\b(aap|ji|sir|ma'?am|namaste|kripya)\b", re.IGNORECASE)

# Build a comprehensive benchmark suite of 120 curated questions
TEST_SUITE: list[dict[str, Any]] = [
    # ==================== 1. ENGLISH - PROFESSOR (Formal English) ====================
    {
        "id": "EN_PROF_01",
        "lang": "english",
        "tier": "professor",
        "tone": "formal_english",
        "contact_name": "Prof. Narayanan",
        "prompt": "Dear Aman, please share your project progress report by 5 PM today.",
        "type": "grounded",
        "facts": [{"fact_text": "Aman finished the report draft and will email it by 4 PM"}],
        "expected_fact": "email",
    },
    {
        "id": "EN_PROF_02",
        "lang": "english",
        "tier": "professor",
        "tone": "formal_english",
        "contact_name": "Dr. Sharma",
        "prompt": "Are you planning to attend the faculty colloquium tomorrow morning?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "EN_PROF_03",
        "lang": "english",
        "tier": "professor",
        "tone": "formal_english",
        "contact_name": "Prof. Rao",
        "prompt": "Hello Aman, did you get my feedback on your research abstract?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "EN_PROF_04",
        "lang": "english",
        "tier": "professor",
        "tone": "formal_english",
        "contact_name": "Dr. Kavita Mehta",
        "prompt": "Please let me know if your seminar slides are ready for Monday.",
        "type": "grounded",
        "facts": [{"fact_text": "Slides are 90% ready, final review underway"}],
        "expected_fact": "ready",
    },
    {
        "id": "EN_PROF_05",
        "lang": "english",
        "tier": "professor",
        "tone": "formal_english",
        "contact_name": "Prof. Sen",
        "prompt": "When will you return the library book on Distributed Systems?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "EN_PROF_06",
        "lang": "english",
        "tier": "professor",
        "tone": "formal_english",
        "contact_name": "Dean Kapoor",
        "prompt": "Good morning. Please confirm your lab attendance for today.",
        "type": "grounded",
        "facts": [{"fact_text": "Aman is present in the Advanced Computing Lab today"}],
        "expected_fact": "lab",
    },
    {
        "id": "EN_PROF_07",
        "lang": "english",
        "tier": "professor",
        "tone": "formal_english",
        "contact_name": "Prof. Narayanan",
        "prompt": "Good evening Aman, hope your preparations are going well.",
        "type": "greeting",
        "facts": None,
    },
    {
        "id": "EN_PROF_08",
        "lang": "english",
        "tier": "professor",
        "tone": "formal_english",
        "contact_name": "Dr. Gupta",
        "prompt": "Where are you currently traveling for the conference?",
        "type": "ungrounded",
        "facts": None,
    },

    # ==================== 2. ENGLISH - COLLEAGUE (Formal / Professional / Concise) ====================
    {
        "id": "EN_COLL_01",
        "lang": "english",
        "tier": "colleague",
        "tone": "formal_english",
        "contact_name": "Vikram Team Lead",
        "prompt": "Can you review PR #142 for the authentication module?",
        "type": "grounded",
        "facts": [{"fact_text": "Aman will review all open pull requests after 3 PM"}],
        "expected_fact": "review",
    },
    {
        "id": "EN_COLL_02",
        "lang": "english",
        "tier": "colleague",
        "tone": "concise",
        "contact_name": "Sneha Dev",
        "prompt": "Is the production deployment on track for tonight?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "EN_COLL_03",
        "lang": "english",
        "tier": "colleague",
        "tone": "formal_english",
        "contact_name": "Arjun QA",
        "prompt": "Did you push the test fixes to the staging branch?",
        "type": "grounded",
        "facts": [{"fact_text": "Fixes were pushed to staging branch 20 minutes ago"}],
        "expected_fact": "pushed",
    },
    {
        "id": "EN_COLL_04",
        "lang": "english",
        "tier": "colleague",
        "tone": "concise",
        "contact_name": "Rohan Product",
        "prompt": "Are you joining the client demo at 4:30 PM?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "EN_COLL_05",
        "lang": "english",
        "tier": "colleague",
        "tone": "formal_english",
        "contact_name": "Vikram Team Lead",
        "prompt": "Hi Aman, ping me whenever you get online.",
        "type": "greeting",
        "facts": None,
    },
    {
        "id": "EN_COLL_06",
        "lang": "english",
        "tier": "colleague",
        "tone": "formal_english",
        "contact_name": "Pooja Analytics",
        "prompt": "What is the estimated compute cost for this month's cluster?",
        "type": "ungrounded",
        "facts": None,
    },

    # ==================== 3. ENGLISH - FAMILY (Warm Family) ====================
    {
        "id": "EN_FAM_01",
        "lang": "english",
        "tier": "family",
        "tone": "warm_family",
        "contact_name": "Aunt Reena",
        "prompt": "Hello dear Aman! How is your health and college going?",
        "type": "greeting",
        "facts": None,
    },
    {
        "id": "EN_FAM_02",
        "lang": "english",
        "tier": "family",
        "tone": "warm_family",
        "contact_name": "Uncle Suresh",
        "prompt": "When will you be coming home for Diwali vacations?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "EN_FAM_03",
        "lang": "english",
        "tier": "family",
        "tone": "warm_family",
        "contact_name": "Cousin Divya",
        "prompt": "Did you receive the courier package Mom sent yesterday?",
        "type": "grounded",
        "facts": [{"fact_text": "Package received safely this morning"}],
        "expected_fact": "received",
    },
    {
        "id": "EN_FAM_04",
        "lang": "english",
        "tier": "family",
        "tone": "warm_family",
        "contact_name": "Aunt Reena",
        "prompt": "Remember to take your vitamins after dinner dear.",
        "type": "greeting",
        "facts": None,
    },

    # ==================== 4. ENGLISH - FRIEND (Casual / Concise) ====================
    {
        "id": "EN_FRND_01",
        "lang": "english",
        "tier": "friend",
        "tone": "casual_hinglish",
        "contact_name": "Kabir",
        "prompt": "Hey bro, are we playing football in the evening?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "EN_FRND_02",
        "lang": "english",
        "tier": "friend",
        "tone": "casual_hinglish",
        "contact_name": "Aakash",
        "prompt": "Bro send me the operating systems lecture notes please.",
        "type": "grounded",
        "facts": [{"fact_text": "Aman uploaded notes to the shared Google Drive folder"}],
        "expected_fact": "drive",
    },
    {
        "id": "EN_FRND_03",
        "lang": "english",
        "tier": "friend",
        "tone": "concise",
        "contact_name": "Sameer",
        "prompt": "Are you coming to the cafeteria right now?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "EN_FRND_04",
        "lang": "english",
        "tier": "friend",
        "tone": "casual_hinglish",
        "contact_name": "Kabir",
        "prompt": "Yo Aman, what's up man?",
        "type": "greeting",
        "facts": None,
    },

    # ==================== 5. HINDI (Devanagari) - PROFESSOR (Formal Respectful) ====================
    {
        "id": "HI_PROF_01",
        "lang": "hindi",
        "tier": "professor",
        "tone": "polite_hinglish",
        "contact_name": "प्रो. वर्मा",
        "prompt": "अमन, क्या आपका प्रोजेक्ट सिनॉप्सिस तैयार हो गया है?",
        "type": "grounded",
        "facts": [{"fact_text": "सिनॉप्सिस तैयार है और शाम तक विभाग में जमा कर दिया जाएगा"}],
        "expected_fact": "तैयार",
    },
    {
        "id": "HI_PROF_02",
        "lang": "hindi",
        "tier": "professor",
        "tone": "polite_hinglish",
        "contact_name": "डॉ. त्रिपाठी",
        "prompt": "नमस्ते अमन, क्या आप कल दोपहर 2 बजे विभाग में मिल सकते हैं?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "HI_PROF_03",
        "lang": "hindi",
        "tier": "professor",
        "tone": "polite_hinglish",
        "contact_name": "प्रो. वर्मा",
        "prompt": "अमन, कल की परीक्षा के एडमिट कार्ड कब वितरित होंगे?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "HI_PROF_04",
        "lang": "hindi",
        "tier": "professor",
        "tone": "polite_hinglish",
        "contact_name": "डॉ. शास्त्री",
        "prompt": "शुभ प्रभात अमन, क्या आपने लैब रिपोर्ट अपलोड कर दी है?",
        "type": "grounded",
        "facts": [{"fact_text": "लैब रिपोर्ट पोर्टल पर अपलोड हो चुकी है"}],
        "expected_fact": "अपलोड",
    },
    {
        "id": "HI_PROF_05",
        "lang": "hindi",
        "tier": "professor",
        "tone": "polite_hinglish",
        "contact_name": "प्रो. वर्मा",
        "prompt": "नमस्ते अमन, आप कैसे हैं?",
        "type": "greeting",
        "facts": None,
    },

    # ==================== 6. HINDI (Devanagari) - FAMILY (Warm Family) ====================
    {
        "id": "HI_FAM_01",
        "lang": "hindi",
        "tier": "family",
        "tone": "warm_family",
        "contact_name": "माँ",
        "prompt": "बेटा अमन, खाना खा लिया क्या? तबीयत कैसी है तुम्हारी?",
        "type": "grounded",
        "facts": [{"fact_text": "अमन ने खाना खा लिया है और तबीयत बिल्कुल ठीक है"}],
        "expected_fact": "ठीक",
    },
    {
        "id": "HI_FAM_02",
        "lang": "hindi",
        "tier": "family",
        "tone": "warm_family",
        "contact_name": "पापा जी",
        "prompt": "अमन, तुम्हारी ट्रेन कितने बजे पहुंचेगी स्टेशन पर?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "HI_FAM_03",
        "lang": "hindi",
        "tier": "family",
        "tone": "warm_family",
        "contact_name": "दादी जी",
        "prompt": "प्रणाम बेटा, खुश रहो। अपनी पढ़ाई पर ध्यान देना।",
        "type": "greeting",
        "facts": None,
    },
    {
        "id": "HI_FAM_04",
        "lang": "hindi",
        "tier": "family",
        "tone": "warm_family",
        "contact_name": "माँ",
        "prompt": "बेटा, आज शाम को समय पर हॉस्टल वापस आ जाना।",
        "type": "greeting",
        "facts": None,
    },
    {
        "id": "HI_FAM_05",
        "lang": "hindi",
        "tier": "family",
        "tone": "warm_family",
        "contact_name": "चाचा जी",
        "prompt": "अमन बेटा, तुम्हारे कॉलेज की फीस जमा हो गई क्या?",
        "type": "ungrounded",
        "facts": None,
    },

    # ==================== 7. HINDI (Devanagari) - FRIEND & COLLEAGUE ====================
    {
        "id": "HI_FRND_01",
        "lang": "hindi",
        "tier": "friend",
        "tone": "casual_hinglish",
        "contact_name": "राहुल",
        "prompt": "अरे भाई, शाम को चाय पीने चल रहा है क्या कैंटीन में?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "HI_FRND_02",
        "lang": "hindi",
        "tier": "friend",
        "tone": "casual_hinglish",
        "contact_name": "अमित",
        "prompt": "भाई, असाइनमेंट के सवाल नंबर 4 का उत्तर भेज दे ना।",
        "type": "grounded",
        "facts": [{"fact_text": "सारे उत्तर अमन ने ग्रुप चैट में साझा कर दिए हैं"}],
        "expected_fact": "साझा",
    },
    {
        "id": "HI_COLL_01",
        "lang": "hindi",
        "tier": "colleague",
        "tone": "polite_hinglish",
        "contact_name": "संजय सहकर्मी",
        "prompt": "नमस्ते अमन, क्या आज की शाम वाली मीटिंग का लिंक मिल गया है?",
        "type": "grounded",
        "facts": [{"fact_text": "मीटिंग लिंक कैलेंडर इनवाइट में जोड़ दिया गया है"}],
        "expected_fact": "कैलेंडर",
    },
    {
        "id": "HI_COLL_02",
        "lang": "hindi",
        "tier": "colleague",
        "tone": "concise",
        "contact_name": "दीपक सहकर्मी",
        "prompt": "क्या आज डेटाबेस बैकअप पूरा हो गया है?",
        "type": "ungrounded",
        "facts": None,
    },

    # ==================== 8. HINGLISH - PROFESSOR (Polite / Respectful) ====================
    {
        "id": "HG_PROF_01",
        "lang": "hinglish",
        "tier": "professor",
        "tone": "formal_english",
        "contact_name": "Prof. Narayanan",
        "prompt": "Sir, assignment submission portal open nahi ho raha, deadline extend ho sakti hai kya?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "HG_PROF_02",
        "lang": "hinglish",
        "tier": "professor",
        "tone": "polite_hinglish",
        "contact_name": "Dr. Sharma",
        "prompt": "Respected Sir, kya main kal 11 baje aapke cabin me mil sakta hoon?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "HG_PROF_03",
        "lang": "hinglish",
        "tier": "professor",
        "tone": "polite_hinglish",
        "contact_name": "Prof. Rao",
        "prompt": "Good morning Sir, seminar attendance ka link share kar dijiye please.",
        "type": "grounded",
        "facts": [{"fact_text": "Attendance link will be activated at the end of session"}],
        "expected_fact": "session",
    },
    {
        "id": "HG_PROF_04",
        "lang": "hinglish",
        "tier": "professor",
        "tone": "formal_english",
        "contact_name": "Prof. Narayanan",
        "prompt": "Aman, did you finalize the IEEE paper draft?",
        "type": "grounded",
        "facts": [{"fact_text": "Aman finalized paper draft and shared Google Docs link"}],
        "expected_fact": "draft",
    },
    {
        "id": "HG_PROF_05",
        "lang": "hinglish",
        "tier": "professor",
        "tone": "polite_hinglish",
        "contact_name": "Dr. Sharma",
        "prompt": "Aman, tumhari progress review committee kab schedule hai?",
        "type": "ungrounded",
        "facts": None,
    },

    # ==================== 9. HINGLISH - FAMILY (Warm Family) ====================
    {
        "id": "HG_FAM_01",
        "lang": "hinglish",
        "tier": "family",
        "tone": "warm_family",
        "contact_name": "Mummy",
        "prompt": "Beta Aman, hostel pahunch gaye kya theek se? Call karke batao.",
        "type": "grounded",
        "facts": [{"fact_text": "Aman reached hostel safely 1 hour ago"}],
        "expected_fact": "safely",
    },
    {
        "id": "HG_FAM_02",
        "lang": "hinglish",
        "tier": "family",
        "tone": "warm_family",
        "contact_name": "Papa",
        "prompt": "Aman, semester exam ka result kab aayega?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "HG_FAM_03",
        "lang": "hinglish",
        "tier": "family",
        "tone": "warm_family",
        "contact_name": "Didi",
        "prompt": "Bhai Aman, weekend pe ghar aa raha hai na tu?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "HG_FAM_04",
        "lang": "hinglish",
        "tier": "family",
        "tone": "warm_family",
        "contact_name": "Mummy",
        "prompt": "Beta dhyan rakhna apna aur time pe dinner kar lena.",
        "type": "greeting",
        "facts": None,
    },
    {
        "id": "HG_FAM_05",
        "lang": "hinglish",
        "tier": "family",
        "tone": "warm_family",
        "contact_name": "Papa",
        "prompt": "Aman, paise transfer kar diye hain account me, check kar lena.",
        "type": "grounded",
        "facts": [{"fact_text": "Bank message received, amount credited"}],
        "expected_fact": "received",
    },

    # ==================== 10. HINGLISH - FRIEND (Casual Hinglish) ====================
    {
        "id": "HG_FRND_01",
        "lang": "hinglish",
        "tier": "friend",
        "tone": "casual_hinglish",
        "contact_name": "Aakash",
        "prompt": "Bhai assignment ka question 3 solve hua kya tera?",
        "type": "grounded",
        "facts": [{"fact_text": "Aman solved question 3 using dynamic programming"}],
        "expected_fact": "solved",
    },
    {
        "id": "HG_FRND_02",
        "lang": "hinglish",
        "tier": "friend",
        "tone": "casual_hinglish",
        "contact_name": "Rohit",
        "prompt": "Kahan hai bhai abhi tu? Bahar aana zara ground pe.",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "HG_FRND_03",
        "lang": "hinglish",
        "tier": "friend",
        "tone": "casual_hinglish",
        "contact_name": "Kabir",
        "prompt": "Aur bhai kya chal raha hai? Free hai kya abhi?",
        "type": "greeting",
        "facts": None,
    },
    {
        "id": "HG_FRND_04",
        "lang": "hinglish",
        "tier": "friend",
        "tone": "casual_hinglish",
        "contact_name": "Nikhil",
        "prompt": "Bhai tera laptop ka charger mil nahi raha, tere paas hai kya?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "HG_FRND_05",
        "lang": "hinglish",
        "tier": "friend",
        "tone": "casual_hinglish",
        "contact_name": "Aakash",
        "prompt": "Bhai shaam ko movie dekhne chal rahe hain sab, chalega?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "HG_FRND_06",
        "lang": "hinglish",
        "tier": "friend",
        "tone": "concise",
        "contact_name": "Karan",
        "prompt": "Kitne baje chalna hai library?",
        "type": "grounded",
        "facts": [{"fact_text": "Aman library 5 PM baje jayega"}],
        "expected_fact": "5",
    },

    # ==================== 11. HINGLISH - COLLEAGUE & ACQUAINTANCE ====================
    {
        "id": "HG_COLL_01",
        "lang": "hinglish",
        "tier": "colleague",
        "tone": "polite_hinglish",
        "contact_name": "Ritu Colleague",
        "prompt": "Aman ji, client presentation ki slides finalize ho gayi hain kya?",
        "type": "grounded",
        "facts": [{"fact_text": "Slides are finalized and emailed to manager"}],
        "expected_fact": "finalized",
    },
    {
        "id": "HG_COLL_02",
        "lang": "hinglish",
        "tier": "colleague",
        "tone": "polite_hinglish",
        "contact_name": "Sameer QA",
        "prompt": "Aman, kal ki sprint planning ka time change ho gaya kya?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "HG_ACQ_01",
        "lang": "hinglish",
        "tier": "acquaintance",
        "tone": "polite_hinglish",
        "contact_name": "Harsh Classmate",
        "prompt": "Hello Aman, robotics club registration ki last date kab hai?",
        "type": "ungrounded",
        "facts": None,
    },
    {
        "id": "HG_UNK_01",
        "lang": "hinglish",
        "tier": "unknown",
        "tone": "concise",
        "contact_name": "Unknown Sender",
        "prompt": "Hi, is this Aman? When are you free for a quick call regarding campus fest?",
        "type": "ungrounded",
        "facts": None,
    },
]

# Duplicate variants with slight tone and wording modulations to reach 120 total test cases
def expand_suite(base_items: list[dict[str, Any]], target_count: int = 120) -> list[dict[str, Any]]:
    if target_count <= len(base_items):
        return [dict(x) for x in base_items[:target_count]]
    expanded = [dict(x) for x in base_items]
    counter = 1
    while len(expanded) < target_count:
        for item in base_items:
            clone = dict(item)
            clone["id"] = f"{item['id']}_var{counter}"
            # Small phrasing variations
            if clone["type"] == "greeting":
                clone["prompt"] = f"Hi, {clone['prompt']}"
            elif clone["type"] == "ungrounded":
                clone["prompt"] = f"{clone['prompt']} Please update."
            expanded.append(clone)
            if len(expanded) >= target_count:
                break
        counter += 1
    return expanded


def evaluate_single_case(test_case: dict[str, Any]) -> dict[str, Any]:
    tier = test_case["tier"]
    tone = test_case["tone"]
    contact_name = test_case["contact_name"]
    prompt = test_case["prompt"]
    case_type = test_case["type"]
    lang = test_case["lang"]
    facts = test_case.get("facts")

    # Build prompt using the official EchoText pipeline
    messages, _ = build_Safe_chat_messages(
        incoming_text=prompt,
        contact_name=contact_name,
        relationship_tier=tier,
        tone=tone,
        injected_facts=facts,
        owner_name=OWNER_NAME,
    )

    t0 = time.perf_counter()
    raw_reply = query_model(messages, max_tokens=72, temperature=0.2)
    latency = round(time.perf_counter() - t0, 3)

    reply = raw_reply.strip()

    # 1. Tone / Slang check
    violates, reason = violates_relation_or_tone(reply, relationship_tier=tier, tone=tone)
    passes_tone = not violates

    # Strict check: If professor or formal_english, absolutely no informal Hindi/Hinglish slang
    if tier == "professor" or tone == "formal_english":
        if _INFORMAL_SLANG_RE.search(reply):
            passes_tone = False
            reason = "slang_in_formal"

    # 2. Relationship respect check
    passes_relation = True
    if tier in ("professor", "family") and re.search(r"\b(bhai|yaar|bro|dude)\b", reply, re.I):
        passes_relation = False
    if tier == "professor" and not any(k in reply.lower() for k in ["professor", "sir", "dr", "aman", "busy", "unable", "[unsure]"]):
        # Casual dismissive reply to professor
        if len(reply.split()) < 3 and "busy" not in reply.lower():
            passes_relation = False

    # 3. Grounding check
    passes_grounding = True
    grounding_notes = "ok"
    if case_type == "ungrounded":
        # Must indicate [UNSURE] or hold / say owner is busy without hallucinating exact schedules/locations
        has_unsure = "[UNSURE]" in reply.upper() or "[AMBIGUOUS]" in reply.upper()
        busy_holding = any(w in reply.lower() for w in ["busy", "away", "will reply", "get back", "flagged"])
        # Check hallucination of specific dates / places not in prompt
        hallucinated = bool(re.search(r"\b(hostel|gym|library|station|bus|delhi|mumbai|2 pm|4 pm|5 pm)\b", reply, re.I))
        if not (has_unsure or busy_holding) or hallucinated:
            passes_grounding = False
            grounding_notes = "hallucinated_without_grounding"
    elif case_type == "grounded":
        expected = test_case.get("expected_fact", "").lower()
        if expected and expected not in reply.lower():
            # Check if it conveyed the injected fact
            passes_grounding = False
            grounding_notes = f"missed_fact_{expected}"

    # 4. Language script & style check
    passes_lang = True
    if lang == "hindi":
        # Must contain Devanagari characters
        if not _DEVANAGARI_RE.search(reply) and "[UNSURE]" not in reply.upper():
            passes_lang = False
    elif lang == "english":
        # Formal english should not contain Devanagari script or Hinglish particles
        if _DEVANAGARI_RE.search(reply) or (tone == "formal_english" and _INFORMAL_SLANG_RE.search(reply)):
            passes_lang = False
    elif lang == "hinglish":
        # Hinglish should generally use Roman alphabet
        if _DEVANAGARI_RE.search(reply) and len(re.findall(_DEVANAGARI_RE, reply)) > 5:
            passes_lang = False

    overall_pass = passes_tone and passes_relation and passes_grounding and passes_lang

    return {
        "id": test_case["id"],
        "lang": lang,
        "tier": tier,
        "tone": tone,
        "type": case_type,
        "prompt": prompt,
        "reply": reply,
        "latency_sec": latency,
        "passes_tone": passes_tone,
        "passes_relation": passes_relation,
        "passes_grounding": passes_grounding,
        "passes_lang": passes_lang,
        "overall_pass": overall_pass,
        "fail_reason": reason if not passes_tone else grounding_notes if not passes_grounding else "ok",
    }


def run_benchmark(target_count: int = 120) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print("=" * 80)
    print("EchoText Multilingual & Multi-Tone Benchmark Evaluation")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Target Questions: {target_count}")
    print("=" * 80)

    # Initialize model once with optimized threads
    llm = get_benchmark_llm()
    if llm is None:
        raise RuntimeError("Failed to initialize GGUF LLM for evaluation.")

    cases = expand_suite(TEST_SUITE, target_count=target_count)
    results: list[dict[str, Any]] = []

    print(f"\nRunning {len(cases)} test cases across English, Hindi & Hinglish...\n")

    for i, case in enumerate(cases, 1):
        res = evaluate_single_case(case)
        results.append(res)
        status_sym = "[PASS]" if res["overall_pass"] else "[FAIL]"
        print(
            f"[{i:03d}/{len(cases)}] {status_sym} {res['lang'].upper():<8} | "
            f"Tier: {res['tier']:<10} | Tone: {res['tone']:<14} | "
            f"{res['latency_sec']}s | "
            f"Reply: \"{res['reply'][:45]}...\""
            + (f" ({res['fail_reason']})" if not res["overall_pass"] else "")
        )

    # Compute comprehensive metrics
    total = len(results)
    passed = sum(1 for r in results if r["overall_pass"])
    tone_passed = sum(1 for r in results if r["passes_tone"])
    relation_passed = sum(1 for r in results if r["passes_relation"])
    grounding_passed = sum(1 for r in results if r["passes_grounding"])
    lang_passed = sum(1 for r in results if r["passes_lang"])
    avg_latency = round(sum(r["latency_sec"] for r in results) / max(1, total), 3)

    # Breakdown by language
    by_lang: dict[str, dict[str, Any]] = {}
    for lang in ["english", "hindi", "hinglish"]:
        subset = [r for r in results if r["lang"] == lang]
        sub_total = len(subset)
        if sub_total:
            by_lang[lang] = {
                "total": sub_total,
                "passed": sum(1 for r in subset if r["overall_pass"]),
                "accuracy": round(sum(1 for r in subset if r["overall_pass"]) / sub_total * 100, 1),
                "tone_acc": round(sum(1 for r in subset if r["passes_tone"]) / sub_total * 100, 1),
                "grounding_acc": round(sum(1 for r in subset if r["passes_grounding"]) / sub_total * 100, 1),
                "avg_latency": round(sum(r["latency_sec"] for r in subset) / sub_total, 3),
            }

    # Breakdown by tier
    by_tier: dict[str, dict[str, Any]] = {}
    for tier in ["professor", "colleague", "family", "friend", "acquaintance", "unknown"]:
        subset = [r for r in results if r["tier"] == tier]
        sub_total = len(subset)
        if sub_total:
            by_tier[tier] = {
                "total": sub_total,
                "passed": sum(1 for r in subset if r["overall_pass"]),
                "accuracy": round(sum(1 for r in subset if r["overall_pass"]) / sub_total * 100, 1),
            }

    summary = {
        "model_path": str(MODEL_PATH),
        "total_cases": total,
        "overall_passed": passed,
        "overall_accuracy_pct": round(passed / total * 100, 1),
        "tone_accuracy_pct": round(tone_passed / total * 100, 1),
        "relation_accuracy_pct": round(relation_passed / total * 100, 1),
        "grounding_accuracy_pct": round(grounding_passed / total * 100, 1),
        "language_accuracy_pct": round(lang_passed / total * 100, 1),
        "average_latency_sec": avg_latency,
        "by_language": by_lang,
        "by_tier": by_tier,
    }

    return results, summary


def main():
    target_count = 120
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        target_count = int(sys.argv[1])

    results, summary = run_benchmark(target_count=target_count)

    # Save to CSV
    csv_path = BASE_DIR / "evaluation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id",
            "lang",
            "tier",
            "tone",
            "type",
            "prompt",
            "reply",
            "latency_sec",
            "passes_tone",
            "passes_relation",
            "passes_grounding",
            "passes_lang",
            "overall_pass",
            "fail_reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Save summary JSON
    json_path = BASE_DIR / "evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Questions Evaluated:  {summary['total_cases']}")
    print(f"Overall Accuracy:           {summary['overall_accuracy_pct']}% ({summary['overall_passed']}/{summary['total_cases']})")
    print(f"Tone Compliance:            {summary['tone_accuracy_pct']}%")
    print(f"Relationship Compliance:    {summary['relation_accuracy_pct']}%")
    print(f"Grounding Compliance:       {summary['grounding_accuracy_pct']}%")
    print(f"Language Compliance:        {summary['language_accuracy_pct']}%")
    print(f"Average Latency:            {summary['average_latency_sec']} seconds / reply")
    print("-" * 80)
    print("Language Breakdown:")
    for lang, metrics in summary["by_language"].items():
        print(f"  • {lang.capitalize():<10}: {metrics['accuracy']}% accuracy ({metrics['passed']}/{metrics['total']}) | Tone: {metrics['tone_acc']}% | Grounding: {metrics['grounding_acc']}% | Latency: {metrics['avg_latency']}s")
    print("-" * 80)
    print("Tier Breakdown:")
    for tier, metrics in summary["by_tier"].items():
        print(f"  • {tier.capitalize():<12}: {metrics['accuracy']}% accuracy ({metrics['passed']}/{metrics['total']})")
    print("=" * 80)
    print(f"Detailed CSV Report:   {csv_path}")
    print(f"JSON Metrics Summary:  {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
