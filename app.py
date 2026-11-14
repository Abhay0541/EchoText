"""Streamlit Modern Control Center & Dashboard for EchoText.

Key Highlights:
- Dynamic Real-Time UI: Live auto-refreshing KPI metrics and draft inbox via ``st.fragment(run_every=3)``.
- User Registration & Login Portal: Complete self-service account registration and authentication.
- Comprehensive Settings Section: Interactive parameter controls (intervals, cooldowns, delays, AI persona) with 1-click restore defaults.
- Modern Executive Theme: High-contrast Dark Slate palette, clear status badges, and refined chat bubbles.
- Self-Explanatory Human Names: Clear terminology, helpful tooltips, and guided 3-step approval workflows.
"""

from __future__ import annotations

from datetime import datetime, timezone
import html
import os
from pathlib import Path
import time
from typing import Any

import httpx
import streamlit as st

from auth import AuthTracker, hash_password, verify_password
from config import (
    ACTIVE_HOURS_ENABLED,
    ACTIVE_HOURS_END,
    ACTIVE_HOURS_START,
    ACTIVE_PROFILE,
    API_SECRET_KEY,
    AUTO_SEND_DELAY_SECONDS,
    BASE_DIR,
    COOLDOWN_SECONDS,
    DEFAULT_FOOTER_TEXT,
    HARDWARE_PROFILE,
    KILL_SWITCH_PAUSE_HOURS,
    MAX_AUTO_REPLIES_PER_HOUR,
    MODEL_PATH,
    MODEL_PROFILES,
    MODELS_DIR,
    OWNER_NAME,
    RAG_DIR,
    RETENTION_DAYS,
    SENSITIVITY_THRESHOLD,
    SERVER_URL,
    TIMEZONE_NAME,
    VALID_MODES,
    VALID_PLATFORMS,
    VALID_TIERS,
    VALID_TONES,
    escape_markdown,
    get_available_models,
    get_proxy_status,
    set_proxy_status,
)
from database import (
    DEFAULT_SYSTEM_SETTINGS,
    add_injected_fact,
    approve_draft,
    authenticate_app_user,
    cancel_pending_auto_send,
    clear_contact_review_flag,
    clear_kill_switch,
    count_registered_users,
    delete_injected_fact,
    edit_and_approve_draft,
    enqueue_outbox_event,
    forget_contact,
    get_active_injected_facts,
    get_all_system_settings,
    get_chat_history,
    get_contacts_eligible_for_graduated_autonomy,
    get_or_create_contact,
    get_sensitivity_threshold,
    init_db,
    link_contact_handle,
    list_app_users,
    list_audit_logs,
    list_contacts,
    list_drafts,
    promote_due_pending_sends,
    purge_old_records,
    register_app_user,
    reject_draft,
    request_delete_sent_reply,
    reset_system_settings_to_defaults,
    save_system_settings,
    set_sensitivity_threshold,
    set_setting,
    update_contact_settings,
    update_draft_options,
    update_user_password,
)
from intelligence import (
    build_Safe_chat_messages,
    format_reply_with_footer,
    generate_catchup_digest,
    generate_reply_cards_detailed,
    is_within_active_hours,
    split_reply_into_parts,
)
from rag import list_all_note_files, retrieve_top_chunks, save_uploaded_note


TIER_LABELS: dict[str, str] = {
    "friend": "🤝 Friend (Peer / Casual)",
    "family": "🏠 Family (Parents / Relatives)",
    "colleague": "💼 Colleague (Work / Project)",
    "professor": "🎓 Professor / Mentor / Teacher",
    "acquaintance": "👋 Acquaintance (Classmate)",
    "unknown": "❓ Unknown / Stranger",
}
TONE_LABELS: dict[str, str] = {
    "casual_hinglish": "😎 Casual Hinglish (Friendly peer Hinglish)",
    "polite_hinglish": "🙏 Polite Hinglish (Respectful 'aap' & 'ji')",
    "formal_english": "👔 Formal English (Strictly professional English)",
    "warm_family": "❤️ Warm Family (Affectionate family tone)",
    "concise": "⚡ Concise (Short 1-sentence factual)",
}
TIER_RECOMMENDED_TONE: dict[str, str] = {
    "professor": "formal_english",
    "colleague": "formal_english",
    "family": "warm_family",
    "friend": "casual_hinglish",
    "acquaintance": "polite_hinglish",
    "unknown": "polite_hinglish",
}


# =============================================================================
# Modern Executive Design System (CSS)
# =============================================================================

def _inject_custom_css() -> None:
    """Inject a clean, modern, professional executive theme with Dark/White mode switching."""
    theme = st.session_state.get("app_theme", "dark")
    if theme == "light":
        css = """
        <style>
        /* Base page canvas (Clean White Theme) */
        .stApp {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: #0F172A !important;
        }
        p, span, label, div[data-testid="stMarkdownContainer"] p {
            color: #334155 !important;
        }
        .stCaption, small, p.ec-desc {
            color: #64748B !important;
        }

        /* Top header container */
        .ec-header {
            background: linear-gradient(135deg, #FFFFFF, #F1F5F9);
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.06);
        }

        /* Title */
        .ec-app-title {
            color: #0F172A !important;
        }

        /* KPI Metric Cards */
        .ec-kpi-container {
            display: flex;
            gap: 12px;
            margin-bottom: 18px;
        }
        .ec-kpi-box {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 14px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: all 0.2s ease-in-out;
        }
        .ec-kpi-box:hover {
            border-color: #6366F1 !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.12);
        }
        .ec-kpi-title {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748B !important;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .ec-kpi-val {
            font-size: 1.4rem;
            font-weight: 700;
            color: #0F172A;
        }

        /* Clean pill badges */
        .ec-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            margin-right: 6px;
            margin-bottom: 4px;
        }
        .badge-green { background: #DCFCE7 !important; color: #166534 !important; border: 1px solid #22C55E !important; }
        .badge-red { background: #FEE2E2 !important; color: #991B1B !important; border: 1px solid #EF4444 !important; }
        .badge-amber { background: #FEF3C7 !important; color: #92400E !important; border: 1px solid #F59E0B !important; }
        .badge-indigo { background: #EEF2FF !important; color: #3730A3 !important; border: 1px solid #6366F1 !important; }
        .badge-cyan { background: #CFFAFE !important; color: #155E75 !important; border: 1px solid #06B6D4 !important; }
        .badge-gray { background: #F1F5F9 !important; color: #334155 !important; border: 1px solid #94A3B8 !important; }

        /* Standard Cards */
        .ec-card {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            color: #0F172A !important;
        }

        /* High-readability Chat Bubbles */
        .ec-bubble-in {
            background: #F1F5F9 !important;
            border-left: 4px solid #0284C7 !important;
            padding: 12px 16px;
            border-radius: 10px;
            margin: 10px 0;
            color: #0F172A !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }
        .ec-bubble-out {
            background: #F0FDF4 !important;
            border-left: 4px solid #16A34A !important;
            padding: 10px 14px;
            border-radius: 10px;
            margin: 8px 0;
            color: #14532D !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }

        /* Attention / Flag Alert Banner */
        .ec-flag-banner {
            background: #FFFBEB !important;
            border: 1px solid #F59E0B !important;
            border-left: 5px solid #F59E0B !important;
            color: #92400E !important;
            padding: 12px 16px;
            border-radius: 10px;
            margin: 10px 0;
            font-size: 0.92rem;
        }

        /* Section descriptions */
        .ec-desc {
            color: #64748B !important;
            font-size: 0.88rem;
            margin-bottom: 16px;
        }

        /* Streamlit Tab Customization */
        button[data-baseweb="tab"] {
            font-size: 0.95rem;
            font-weight: 600;
            padding: 10px 18px;
            color: #64748B !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #4F46E5 !important;
            border-bottom-color: #4F46E5 !important;
        }

        /* Container overrides for light theme */
        div[data-testid="stExpander"], div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important;
            border-color: #E2E8F0 !important;
        }
        div[data-testid="stExpander"] summary {
            color: #0F172A !important;
            font-weight: 600;
        }

        /* Inputs, textareas, and select boxes */
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border-color: #CBD5E1 !important;
        }
        input, textarea {
            color: #0F172A !important;
            background-color: #FFFFFF !important;
        }

        /* Secondary and standard buttons */
        button[kind="secondary"] {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
            border: 1px solid #CBD5E1 !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        }
        button[kind="secondary"]:hover {
            background-color: #F8FAFC !important;
            border-color: #6366F1 !important;
            color: #4F46E5 !important;
        }

        .ec-test-box {
            background: #F1F5F9;
            border: 1px solid rgba(99, 102, 241, 0.35);
            border-radius: 10px;
            padding: 12px 16px;
            margin: 8px 0;
            color: #0F172A;
        }
        </style>
        """
    else:
        css = """
        <style>
        /* Base page canvas (Dark Slate Theme) */
        .stApp {
            background-color: #0A0E17 !important;
            color: #E2E8F0 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: #F8FAFC !important;
        }
        p, span, label, div[data-testid="stMarkdownContainer"] p {
            color: #CBD5E1;
        }
        .stCaption, small, p.ec-desc {
            color: #94A3B8 !important;
        }

        /* Top header container */
        .ec-header {
            background: linear-gradient(135deg, rgba(19, 27, 46, 0.95), rgba(15, 23, 42, 0.95));
            border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.4);
        }

        /* Title */
        .ec-app-title {
            color: #F8FAFC !important;
        }

        /* KPI Metric Cards */
        .ec-kpi-container {
            display: flex;
            gap: 12px;
            margin-bottom: 18px;
        }
        .ec-kpi-box {
            background: linear-gradient(145deg, #131B2E, #111827);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transition: all 0.2s ease-in-out;
        }
        .ec-kpi-box:hover {
            border-color: rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
        }
        .ec-kpi-title {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94A3B8;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .ec-kpi-val {
            font-size: 1.4rem;
            font-weight: 700;
            color: #F8FAFC;
        }

        /* Clean pill badges */
        .ec-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            margin-right: 6px;
            margin-bottom: 4px;
        }
        .badge-green { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid #10B981; }
        .badge-red { background: rgba(239, 68, 68, 0.18); color: #FCA5A5; border: 1px solid #EF4444; }
        .badge-amber { background: rgba(245, 158, 11, 0.18); color: #FCD34D; border: 1px solid #F59E0B; }
        .badge-indigo { background: rgba(99, 102, 241, 0.18); color: #A5B4FC; border: 1px solid #6366F1; }
        .badge-cyan { background: rgba(6, 182, 212, 0.18); color: #67E8F9; border: 1px solid #06B6D4; }
        .badge-gray { background: rgba(100, 116, 139, 0.18); color: #CBD5E1; border: 1px solid #64748B; }

        /* Standard Cards */
        .ec-card {
            background: #131B2E;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            color: #E2E8F0;
        }

        /* High-readability Chat Bubbles */
        .ec-bubble-in {
            background: #1E293B;
            border-left: 4px solid #38BDF8;
            padding: 12px 16px;
            border-radius: 10px;
            margin: 10px 0;
            color: #F1F5F9;
        }
        .ec-bubble-out {
            background: rgba(16, 185, 129, 0.1);
            border-left: 4px solid #10B981;
            padding: 10px 14px;
            border-radius: 10px;
            margin: 8px 0;
            color: #ECFDF5;
        }

        /* Attention / Flag Alert Banner */
        .ec-flag-banner {
            background: rgba(245, 158, 11, 0.12);
            border: 1px solid #F59E0B;
            border-left: 5px solid #F59E0B;
            color: #FEF3C7;
            padding: 12px 16px;
            border-radius: 10px;
            margin: 10px 0;
            font-size: 0.92rem;
        }

        /* Section descriptions */
        .ec-desc {
            color: #94A3B8;
            font-size: 0.88rem;
            margin-bottom: 16px;
        }

        /* Streamlit Tab Customization */
        button[data-baseweb="tab"] {
            font-size: 0.95rem;
            font-weight: 600;
            padding: 10px 18px;
            color: #94A3B8;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #6366F1 !important;
            border-bottom-color: #6366F1 !important;
        }

        .ec-test-box {
            background: rgba(30, 41, 59, 0.9);
            border: 1px solid rgba(99, 102, 241, 0.4);
            border-radius: 10px;
            padding: 12px 16px;
            margin: 8px 0;
            color: #F1F5F9;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


# =============================================================================
# Helper REST API Caller
# =============================================================================

def _api_call(
    method: str,
    path: str,
    json_body: dict[str, Any] | None = None,
    timeout: float = 4.0,
) -> dict[str, Any] | None:
    """Call the local Brain REST API with mandatory X-API-Key header; fallback to direct DB if offline."""
    headers = {"Content-Type": "application/json", "X-API-Key": API_SECRET_KEY}
    url = f"{SERVER_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.request(method, url, json=json_body, headers=headers)
            if resp.status_code < 400:
                data: dict[str, Any] = resp.json()
                return data
    except Exception:
        return None
    return None


def _init_session_state() -> None:
    """Ensure database schema is created and reactive session trackers exist."""
    init_db()
    if "app_theme" not in st.session_state:
        stored_theme = str(get_all_system_settings().get("theme", "dark")).lower().strip()
        st.session_state.app_theme = "light" if stored_theme == "light" else "dark"
    if "auth_tracker" not in st.session_state:
        st.session_state.auth_tracker = AuthTracker()
    if "last_catchup_digest" not in st.session_state:
        st.session_state.last_catchup_digest = ""
    if "auth_mode" not in st.session_state:
        # Default to login if users exist, otherwise registration
        users_count = count_registered_users()
        st.session_state.auth_mode = "login" if users_count > 0 or st.session_state.auth_tracker.is_configured() else "register"


# =============================================================================
# User Registration & Login Portal
# =============================================================================

def _render_auth_portal(tracker: AuthTracker) -> bool:
    """Render the user registration or login portal. Returns True once authenticated."""
    if tracker.check_session_active():
        return True

    # Check for active query_params session token (persists across browser reloads)
    param_token = st.query_params.get("session_token", "")
    if param_token and tracker.restore_session(param_token):
        return True

    user_count = count_registered_users()
    has_env_password = bool(tracker.password_hash)
    can_switch_mode = user_count > 0 or has_env_password

    is_dark = st.session_state.get("app_theme", "dark") == "dark"
    title_color = "#F8FAFC" if is_dark else "#0F172A"
    sub_color = "#94A3B8" if is_dark else "#64748B"

    # Subtle top header bar with mode changer switch
    _, top_theme_col = st.columns([6, 1])
    with top_theme_col:
        auth_theme_lbl = "☀️ Light" if is_dark else "🌙 Dark"
        if st.button(
            auth_theme_lbl,
            key="auth_theme_toggle_btn",
            help="Switch between Dark Slate and Clean White theme",
            use_container_width=True,
        ):
            new_theme = "light" if is_dark else "dark"
            st.session_state.app_theme = new_theme
            set_setting("theme", new_theme)
            st.rerun()

    # Centered modern card
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.write("")
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 24px;">
                <h1 style="font-size: 2.2rem; font-weight: 800; color: {title_color}; margin-bottom: 4px;">
                    💬 EchoText
                </h1>
                <p style="color: #818CF8; font-size: 1.0rem; font-weight: 500;">
                    Autonomous AI Communication Assistant
                </p>
                <p style="color: {sub_color}; font-size: 0.85rem;">
                    Localhost 127.0.0.1 • PBKDF2-HMAC-SHA256 • Full Local Control
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if tracker.is_locked_out():
            rem = tracker.remaining_lockout_seconds()
            st.error(f"🔒 Account locked due to failed attempts. Please wait {rem} seconds.")
            return False

        if can_switch_mode:
            mode_choice = st.radio(
                "Access Mode",
                ["Sign In to Dashboard", "Register New User"],
                index=0 if st.session_state.auth_mode == "login" else 1,
                horizontal=True,
                label_visibility="collapsed",
            )
            st.session_state.auth_mode = "login" if mode_choice == "Sign In to Dashboard" else "register"

        # --- View 1: User Registration ---
        if st.session_state.auth_mode == "register" or (user_count == 0 and not has_env_password):
            st.markdown(
                f"""
                <div class="ec-card" style="border-top: 4px solid #6366F1;">
                    <h3 style="margin-top:0; color: {title_color};">👤 Set Up Owner Account</h3>
                    <p style="color: {sub_color}; font-size: 0.88rem;">
                        Create your administrator credentials to manage your AI assistant securely.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.form("register_form"):
                new_username = st.text_input("Username (e.g. admin or your nickname)", placeholder="e.g. abhay")
                new_display_name = st.text_input("Your Display Name", placeholder="e.g. Abhay Kumar")
                new_password = st.text_input("Choose Strong Password (min 6 characters)", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")

                reg_submitted = st.form_submit_button("✨ Create Account & Enter Dashboard", type="primary", use_container_width=True)
                if reg_submitted:
                    if not new_username.strip():
                        st.error("Please enter a username.")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match!")
                    else:
                        ok, msg = tracker.attempt_register(
                            new_username.strip(),
                            new_password,
                            display_name=new_display_name.strip() or new_username.strip(),
                        )
                        if ok:
                            tok = tracker.create_session()
                            st.query_params["session_token"] = tok
                            st.success(msg)
                            time.sleep(0.3)
                            st.rerun()
                        else:
                            st.error(f"Registration failed: {msg}")

        # --- View 2: User Sign In ---
        else:
            st.markdown(
                f"""
                <div class="ec-card" style="border-top: 4px solid #10B981;">
                    <h3 style="margin-top:0; color: {title_color};">🔐 Dashboard Sign In</h3>
                    <p style="color: {sub_color}; font-size: 0.88rem;">
                        Enter your username & password to access the command center.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.form("login_form"):
                login_user = st.text_input("Username", placeholder="e.g. abhay (or leave blank if using .env password)")
                login_pwd = st.text_input("Password", type="password", placeholder="Enter your password")

                login_submitted = st.form_submit_button("Sign In →", type="primary", use_container_width=True)
                if login_submitted:
                    if not login_pwd:
                        st.error("Please enter your password.")
                    else:
                        ok, msg = tracker.attempt_login(login_pwd, username=login_user.strip())
                        if ok:
                            tok = tracker.create_session()
                            st.query_params["session_token"] = tok
                            st.success(msg)
                            time.sleep(0.3)
                            st.rerun()
                        else:
                            st.error(msg)

    return False


# =============================================================================
# Live Command Header & KPIs (Dynamic Fragment)
# =============================================================================

@st.fragment(run_every=3)
def _render_live_header_and_kpis(tracker: AuthTracker) -> None:
    """Reactive fragment that updates KPIs and status in real-time every 3s."""
    promote_due_pending_sends()

    current_status = get_proxy_status()
    active_ok = is_within_active_hours()
    contacts = list_contacts()
    pending_drafts = list_drafts(statuses=["pending_approval", "failed"], limit=100)
    sent_drafts = list_drafts(statuses=["sent"], limit=100)
    settings = get_all_system_settings()

    flagged_contacts = [
        c for c in contacts if bool(c.get("flagged_for_review")) or bool(c.get("paused_until"))
    ]

    # Status summary banner
    top_col1, top_col2, top_col3 = st.columns([3.2, 1.8, 1.4])
    with top_col1:
        if current_status == "BUSY" and flagged_contacts:
            st.markdown("### 🟡 EchoText Assistant — **BUSY (Review Needed on Flagged Chat)**")
        elif current_status == "BUSY":
            st.markdown("### 🔴 EchoText Assistant — **BUSY (Active Interception & Answering)**")
        else:
            st.markdown("### 🟢 EchoText Assistant — **AVAILABLE (Passive / Gate Open)**")

        schedule_label = (
            f"Schedule ({settings.get('active_hours_start', 8):02d}:00–{settings.get('active_hours_end', 23):02d}:00): "
            + ("✅ Open" if active_ok else "🌙 Quiet Hours")
            if bool(settings.get("active_hours_enabled"))
            else "Schedule: 24/7 Monitoring Active"
        )
        st.caption(f"Owner: **{settings.get('owner_name', OWNER_NAME)}** • Profile: `{ACTIVE_PROFILE['description']}` • {schedule_label}")

    with top_col2:
        if current_status == "BUSY":
            if st.button("⏹ Switch to AVAILABLE & Build Catch-Up Digest", use_container_width=True):
                api_res = _api_call("POST", "/status", {"status": "AVAILABLE"})
                if api_res is not None:
                    st.session_state.last_catchup_digest = api_res.get("catchup_digest") or ""
                else:
                    set_proxy_status("AVAILABLE")
                    recent = list_drafts(limit=30)
                    digest = generate_catchup_digest(recent)
                    st.session_state.last_catchup_digest = digest
                    enqueue_outbox_event(
                        "notification",
                        {
                            "type": "catchup_digest",
                            "digest_text": digest,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                st.rerun()
        else:
            if st.button("🔴 Activate BUSY Mode (Start Auto-Reply)", type="primary", use_container_width=True):
                if _api_call("POST", "/status", {"status": "BUSY"}) is None:
                    set_proxy_status("BUSY")
                st.session_state.last_catchup_digest = ""
                st.rerun()

    with top_col3:
        logged_name = tracker.current_display_name or tracker.current_username or "Admin"
        st.caption(f"👤 **{logged_name}**")
        h_btn_c1, h_btn_c2, h_btn_c3 = st.columns(3)
        is_dark = st.session_state.get("app_theme", "dark") == "dark"
        theme_btn_label = "☀️ Light" if is_dark else "🌙 Dark"
        with h_btn_c1:
            if st.button(
                theme_btn_label,
                key="header_theme_toggle_btn",
                help="Switch between Dark Slate and Clean White theme",
                use_container_width=True,
            ):
                new_theme = "light" if is_dark else "dark"
                st.session_state.app_theme = new_theme
                set_setting("theme", new_theme)
                st.rerun()
        with h_btn_c2:
            if st.button(
                "🔄 Sync",
                key="top_refresh_btn",
                help="Refresh all database feeds and queues immediately without logging out",
                use_container_width=True,
            ):
                tracker.last_activity_at = time.time()
                st.rerun()
        with h_btn_c3:
            if st.button("🚪 Exit", key="logout_btn", help="Sign out of dashboard", use_container_width=True):
                st.query_params.clear()
                tracker.logout()
                st.rerun()

    # 4 High-Readability Metric Boxes
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="ec-kpi-box">
                <div class="ec-kpi-title">Current Gate</div>
                <div class="ec-kpi-val" style="color:{'#EF4444' if current_status == 'BUSY' else '#10B981'}">
                    {'🔴 BUSY' if current_status == 'BUSY' else '🟢 AVAILABLE'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="ec-kpi-box">
                <div class="ec-kpi-title">Pending Review</div>
                <div class="ec-kpi-val" style="color:{'#F59E0B' if pending_drafts else '#94A3B8'}">
                    📥 {len(pending_drafts)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="ec-kpi-box">
                <div class="ec-kpi-title">Flagged / Paused</div>
                <div class="ec-kpi-val" style="color:{'#F59E0B' if flagged_contacts else '#94A3B8'}">
                    🟡 {len(flagged_contacts)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="ec-kpi-box">
                <div class="ec-kpi-title">Dispatched Replies</div>
                <div class="ec-kpi-val" style="color:#6366F1;">
                    📤 {len(sent_drafts)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.last_catchup_digest:
        with st.expander("📋 Session Catch-Up Digest", expanded=True):
            st.text(st.session_state.last_catchup_digest)


# =============================================================================
# Tab 1: Approval Inbox & Outgoing Countdown Queue
# =============================================================================

def _render_approval_inbox_tab() -> None:
    """Render the human-in-the-loop review inbox, candidate reply cards, and outgoing queue."""
    settings = get_all_system_settings()
    auto_delay = int(settings.get("auto_send_delay_seconds", AUTO_SEND_DELAY_SECONDS))

    # Outgoing Countdown Queue
    pending_sends = list_drafts(statuses=["pending_send"], limit=20)
    if pending_sends:
        st.subheader(f"⏳ Outgoing Countdown Queue ({auto_delay}s Auto-Send Grace Window)")
        st.markdown(
            '<p class="ec-desc">These auto-replies are counting down before being sent on Telegram. '
            'You can cancel or hold them anytime before the timer reaches zero.</p>',
            unsafe_allow_html=True,
        )
        for ps in pending_sends:
            did = int(ps["id"])
            parts = ps.get("reply_parts") or [ps.get("draft_text", "")]
            disp = html.escape(str(ps.get("display_name") or ps.get("canonical_name") or ""))
            canon = html.escape(str(ps.get("canonical_name") or ""))

            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(
                        f"**To:** {disp} (`@{canon}`) • **Scheduled Send:** `{ps.get('scheduled_send_at')}` • "
                        f"**Parts:** `{len(parts)} bubble(s)`"
                    )
                    for p_i, p_txt in enumerate(parts, start=1):
                        st.markdown(
                            f'<div class="ec-bubble-out"><b>Part {p_i}/{len(parts)}:</b> {html.escape(str(p_txt))}</div>',
                            unsafe_allow_html=True,
                        )
                with c2:
                    if st.button("🛑 Cancel & Hold", key=f"cancel_queue_{did}", use_container_width=True):
                        if _api_call("POST", f"/drafts/{did}/cancel_auto") is None:
                            cancel_pending_auto_send(did)
                        st.warning(f"Draft #{did} cancelled from queue and moved to Approval Inbox.")
                        st.rerun()

    st.subheader("📥 Human-in-the-Loop Message Review Inbox")
    pending_drafts = list_drafts(statuses=["pending_approval", "failed"], limit=50)

    if not pending_drafts:
        st.success("✨ All clear! No pending drafts or flagged conversations waiting for approval.")
    else:
        st.markdown(
            f'<p class="ec-desc">There are <b>{len(pending_drafts)} message(s)</b> requiring your review. '
            'Follow the simple 3-step confirmation below to send, edit, or reject.</p>',
            unsafe_allow_html=True,
        )

        for draft in pending_drafts:
            did = int(draft["id"])
            options: list[str] = draft.get("options") or [draft.get("draft_text", "")]
            chunks: list[dict[str, str]] = draft.get("chunks") or []
            sens = int(draft.get("sensitivity_score", 0))
            triage = str(draft.get("triage_label", "chitchat"))
            is_flagged = bool(draft.get("flagged_for_review"))
            flag_reason = str(draft.get("flag_reason") or "")
            disp_html = html.escape(str(draft.get("display_name") or ""))
            canon_html = html.escape(str(draft.get("canonical_name") or ""))
            in_html = html.escape(str(draft.get("incoming_text") or "")).replace("\n", "<br>")

            with st.container(border=True):
                # Header badges
                t_badge = f'<span class="ec-badge badge-indigo">🏷 {triage.upper()}</span>'
                s_cls = "badge-red" if sens >= 3 else ("badge-amber" if sens == 2 else "badge-gray")
                s_badge = f'<span class="ec-badge {s_cls}">🔒 Sensitivity {sens}/3</span>'
                f_badge = '<span class="ec-badge badge-amber">🟡 CONTACT FLAGGED (AI PAUSED)</span>' if is_flagged else ""

                st.markdown(
                    f"**Draft #{did}** • **{disp_html}** (<code>@{canon_html}</code>) • "
                    f"<small style='color:#94A3B8;'>{draft.get('created_at')}</small><br>"
                    f"{t_badge}{s_badge}{f_badge}",
                    unsafe_allow_html=True,
                )

                needs_context = is_flagged or triage in ("unsure", "ambiguous")
                if is_flagged and flag_reason:
                    st.markdown(
                        f'<div class="ec-flag-banner">⚠️ <b>Missing Context / Owner Attention Needed:</b> {html.escape(flag_reason)}<br>'
                        '<small>EchoText did not guess or invent facts. Please either <b>give context below to regenerate a reply</b> OR <b>write/send your own reply</b>.</small></div>',
                        unsafe_allow_html=True,
                    )
                elif needs_context:
                    st.markdown(
                        '<div class="ec-flag-banner">⚠️ <b>No Matching Context Found:</b> EchoText held this message instead of inventing details. '
                        'Please supply context below so AI can draft accurately, or reply yourself.</div>',
                        unsafe_allow_html=True,
                    )

                if draft.get("status") == "failed":
                    st.error(f"⚠️ Delivery failed on Telegram ({draft.get('send_attempts')} attempts): {draft.get('last_error')}")

                # Step 1: Review Incoming
                st.markdown(
                    f'<div class="ec-bubble-in"><b>Step 1 — Incoming Message:</b><br>{in_html}</div>',
                    unsafe_allow_html=True,
                )

                if chunks:
                    with st.expander(f"📚 Grounded in {len(chunks)} Reference Note Chunks"):
                        for ch in chunks:
                            st.markdown(f"- **`{ch.get('chunk_id')}`** (BM25 score: `{ch.get('score')}`): {ch.get('text')}")

                # Give Context & Regenerate AI Reply
                with st.expander("🧠 Give Context & Re-Generate AI Reply", expanded=needs_context):
                    ctx_input = st.text_input(
                        "Tell EchoText the context / factual answer for this message:",
                        placeholder="e.g. Busy in DBMS lab till 6 PM; we can talk about the project at 7 PM.",
                        key=f"ctx_inp_{did}",
                    )
                    ctx_col1, ctx_col2 = st.columns([2, 1])
                    with ctx_col1:
                        save_as_fact = st.checkbox(
                            "Remember this context for 2 hours (helps future questions)",
                            value=True,
                            key=f"ctx_save_{did}",
                        )
                    with ctx_col2:
                        if st.button("✨ Regenerate with Context", key=f"btn_regen_{did}", type="primary", use_container_width=True):
                            if not ctx_input.strip():
                                st.warning("Please enter context first (or write your own reply below).")
                            else:
                                regen_res = _api_call(
                                    "POST",
                                    f"/drafts/{did}/regenerate",
                                    {
                                        "context_text": ctx_input.strip(),
                                        "save_as_fact": save_as_fact,
                                        "ttl_minutes": 120,
                                    },
                                    timeout=35.0,
                                )
                                if regen_res is None:
                                    # Offline fallback if Brain API is not reachable
                                    d_cid = int(draft["contact_id"])
                                    d_tier = str(draft.get("relationship_tier") or "unknown")
                                    d_tone = str(draft.get("tone") or "casual_hinglish")
                                    if save_as_fact:
                                        add_injected_fact(
                                            ctx_input.strip(),
                                            ttl_minutes=120,
                                            scope_contact_id=d_cid,
                                            scope_tier=d_tier if d_tier != "unknown" else "all_trusted",
                                        )
                                    facts_now = get_active_injected_facts(contact_id=d_cid, relationship_tier=d_tier)
                                    if not any(ctx_input.strip() in str(f.get("fact_text", "")) for f in facts_now):
                                        facts_now.insert(0, {"fact_text": ctx_input.strip(), "expires_at": "owner_supplied"})
                                    chunks_now = retrieve_top_chunks(
                                        str(draft.get("incoming_text") or ""),
                                        contact_handle=str(draft.get("canonical_name") or ""),
                                        relationship_tier=d_tier,
                                        top_k=3,
                                    )
                                    p_msgs, p_meta = build_Safe_chat_messages(
                                        str(draft.get("incoming_text") or ""),
                                        contact_name=str(draft.get("display_name") or draft.get("canonical_name") or ""),
                                        relationship_tier=d_tier,
                                        tone=d_tone,
                                        owner_name=str(settings.get("owner_name") or OWNER_NAME),
                                        retrieved_chunks=chunks_now,
                                        injected_facts=facts_now,
                                    )
                                    gen_out = generate_reply_cards_detailed(
                                        p_msgs,
                                        num_cards=int(settings.get("num_draft_options", 3)),
                                        triage_label="facts",
                                        language_style=p_meta["language_style"],
                                        tone=d_tone,
                                        relationship_tier=d_tier,
                                        owner_name=str(settings.get("owner_name") or OWNER_NAME),
                                        injected_facts=facts_now,
                                        retrieved_chunks=chunks_now,
                                    )
                                    f_cards = [
                                        format_reply_with_footer(
                                            c_txt,
                                            footer_enabled=bool(draft.get("footer_enabled")),
                                            footer_text=str(draft.get("footer_text") or ""),
                                        )
                                        for c_txt in gen_out.cards
                                    ]
                                    update_draft_options(
                                        did,
                                        draft_options=f_cards,
                                        reply_parts=split_reply_into_parts(f_cards[0]) if f_cards else [],
                                        triage_label="facts",
                                    )
                                st.success(f"Draft #{did} regenerated using your context!")
                                st.rerun()

                # Step 2 & 3: Candidate Cards
                st.markdown("**Step 2 & 3 — Pick a Candidate Reply to Confirm & Send:**")
                card_cols = st.columns(max(1, len(options)))
                for idx, opt_text in enumerate(options):
                    preview_parts = split_reply_into_parts(str(opt_text))
                    with card_cols[idx]:
                        st.markdown(f"**Card #{idx + 1}** <small>({len(preview_parts)} message bubble(s))</small>", unsafe_allow_html=True)
                        for p_i, p_txt in enumerate(preview_parts, start=1):
                            lbl = f"Part {p_i}: " if len(preview_parts) > 1 else ""
                            st.markdown(f'<div class="ec-bubble-out"><small><b>{lbl}</b></small>{html.escape(p_txt)}</div>', unsafe_allow_html=True)

                        if st.button(f"✅ Send Card #{idx + 1}", key=f"send_opt_{did}_{idx}", use_container_width=True):
                            if _api_call("POST", f"/drafts/{did}/approve", {"option_index": idx}) is None:
                                approve_draft(did, option_index=idx, reply_parts=preview_parts)
                            st.success(f"Approved Draft #{did} and dispatched to Telegram!")
                            st.rerun()

                # Manual Edit & Reject Section
                with st.expander("✍️ Reply Yourself (Write Custom Reply), or Reject / Reply Manually on Telegram", expanded=needs_context):
                    custom_text = st.text_area(
                        "Write your own reply text (long replies split automatically into Part 1 & Part 2):",
                        value=options[0] if options else "",
                        key=f"custom_edit_box_{did}",
                    )
                    custom_parts = split_reply_into_parts(custom_text)
                    if len(custom_parts) > 1:
                        st.caption(f"✂️ Will be dispatched as **{len(custom_parts)} sequential bubbles** (Gap: {settings.get('multipart_gap_seconds', 0.35)}s):")
                        for cp_i, cp_txt in enumerate(custom_parts, start=1):
                            st.markdown(f'<div class="ec-bubble-out"><b>Part {cp_i}:</b> {html.escape(cp_txt)}</div>', unsafe_allow_html=True)

                    btn_c1, btn_c2 = st.columns(2)
                    with btn_c1:
                        if st.button("💾 Send My Custom Reply", key=f"btn_save_send_{did}", type="primary", use_container_width=True):
                            if _api_call("POST", f"/drafts/{did}/edit", {"new_text": custom_text}) is None:
                                edit_and_approve_draft(did, custom_text, reply_parts=custom_parts)
                            st.success(f"Custom reply approved for Draft #{did}!")
                            st.rerun()
                    with btn_c2:
                        if st.button("✋ Unflag & I'll Reply Manually on Telegram", key=f"btn_reject_{did}", use_container_width=True):
                            if _api_call("POST", f"/drafts/{did}/reject") is None:
                                reject_draft(did)
                            st.warning(f"Draft #{did} dismissed and contact unflagged so you can reply manually.")
                            st.rerun()

    # Recent Database Visualizer & Conversation Feed
    st.divider()
    st.subheader("💬 Recent Database Feed & Conversation Visualizer")
    st.markdown(
        '<p class="ec-desc">Live chronological visualizer showing both the <b>incoming sender message</b> '
        'and the corresponding <b>AI-generated reply / draft</b> across all recent interactions.</p>',
        unsafe_allow_html=True,
    )

    feed_filter = st.radio(
        "Feed Filter",
        ["All Recent Records (Latest 30)", "Sent / Dispatched", "Pending Approval", "Cancelled / Other"],
        horizontal=True,
        label_visibility="collapsed",
        key="recent_feed_filter",
    )
    if feed_filter == "Sent / Dispatched":
        target_statuses = ["sent", "approved", "sending"]
    elif feed_filter == "Pending Approval":
        target_statuses = ["pending_approval", "pending_send", "failed"]
    elif feed_filter == "Cancelled / Other":
        target_statuses = ["cancelled", "rejected", "shadow", "deleted_sent"]
    else:
        target_statuses = None

    recent_records = list_drafts(statuses=target_statuses, limit=30)
    if not recent_records:
        st.caption("No recent chat interactions found in the database for this filter.")
    else:
        for rd in recent_records:
            rd_id = int(rd["id"])
            rd_status = str(rd.get("status") or "")
            rd_canon = html.escape(str(rd.get("canonical_name") or ""))
            rd_disp = html.escape(str(rd.get("display_name") or rd.get("canonical_name") or ""))
            rd_incoming = html.escape(str(rd.get("incoming_text") or "")).replace("\n", "<br>")
            rd_reply = html.escape(str(rd.get("draft_text") or (rd.get("options") or [""])[0])).replace("\n", "<br>")
            rd_parts = rd.get("reply_parts") or [rd.get("draft_text", "")]
            rd_ts = str(rd.get("created_at") or "")[:19]
            rd_sens = int(rd.get("sensitivity_score", 0))
            rd_triage = str(rd.get("triage_label", "chitchat"))

            # Status styling
            if rd_status in ("sent", "approved"):
                st_badge = f'<span class="ec-badge badge-green">✅ {rd_status.upper()}</span>'
            elif rd_status in ("pending_approval", "pending_send"):
                st_badge = f'<span class="ec-badge badge-amber">⏳ {rd_status.upper()}</span>'
            elif rd_status == "shadow":
                st_badge = f'<span class="ec-badge badge-cyan">👁️ SHADOW</span>'
            elif rd_status == "failed":
                st_badge = f'<span class="ec-badge badge-red">⚠️ FAILED</span>'
            else:
                st_badge = f'<span class="ec-badge badge-gray">{rd_status.upper()}</span>'

            sens_cls = "badge-red" if rd_sens >= 3 else ("badge-amber" if rd_sens == 2 else "badge-gray")
            sens_badge = f'<span class="ec-badge {sens_cls}">Score {rd_sens}/3</span>'
            triage_badge = f'<span class="ec-badge badge-indigo">🏷 {rd_triage}</span>'

            with st.container(border=True):
                head_c1, head_c2 = st.columns([4, 1])
                with head_c1:
                    st.markdown(
                        f"**Draft #{rd_id}** • <b>{rd_disp}</b> (<code>@{rd_canon}</code>) • "
                        f"{st_badge} {triage_badge} {sens_badge} • <small style='color:#94A3B8;'>{rd_ts}</small>",
                        unsafe_allow_html=True,
                    )
                with head_c2:
                    if rd_status in ("sent", "approved", "sending"):
                        if st.button("🗑 Unsend Telegram", key=f"unsend_db_{rd_id}", use_container_width=True):
                            if _api_call("POST", f"/drafts/{rd_id}/delete_sent") is None:
                                request_delete_sent_reply(rd_id)
                            st.warning(f"Deletion requested for Draft #{rd_id} on Telegram.")
                            st.rerun()

                # Conversation Pair: Incoming Message Box + AI Generated Reply Box
                pair_c1, pair_c2 = st.columns(2)
                with pair_c1:
                    st.markdown(
                        f'<div class="ec-bubble-in" style="min-height: 90px;">'
                        f'<div style="font-size:0.8rem; font-weight:700; color:#818CF8; margin-bottom:4px;">'
                        f'👤 Incoming from @{rd_canon} ({rd_disp}):</div>'
                        f'{rd_incoming or "<i>[Empty incoming text]</i>"}</div>',
                        unsafe_allow_html=True,
                    )
                with pair_c2:
                    st.markdown(
                        f'<div class="ec-bubble-out" style="min-height: 90px;">'
                        f'<div style="font-size:0.8rem; font-weight:700; color:#10B981; margin-bottom:4px;">'
                        f'🤖 EchoText AI Reply ({rd_status}):</div>'
                        f'{rd_reply or "<i>[No reply generated]</i>"}'
                        f'<div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">Parts: {len(rd_parts)} bubble(s)</div></div>',
                        unsafe_allow_html=True,
                    )


# =============================================================================
# Tab 2: Contacts & Autonomy Controls
# =============================================================================

def _on_contact_tier_changed(contact_id: int) -> None:
    """Auto-sync Reply Tone Style to the recommended tone when Relationship Tier changes."""
    new_tier = str(st.session_state.get(f"sel_t_{contact_id}") or "").strip().lower()
    if new_tier in TIER_RECOMMENDED_TONE:
        st.session_state[f"sel_tn_{contact_id}"] = TIER_RECOMMENDED_TONE[new_tier]


def _render_contacts_tab() -> None:
    """Render Contacts, Autonomy Mode assignments, and Conversation History."""
    st.subheader("👥 Contact Manager & Autonomy Controls")
    st.markdown(
        '<p class="ec-desc">Configure each contact\'s <b>Autonomy Mode</b>, <b>Relationship Tier</b>, and <b>Reply Tone Style</b>. '
        'Changing a contact\'s Relationship Tier automatically recommends the matching Reply Tone Style (which you can customize anytime).</p>',
        unsafe_allow_html=True,
    )

    live_settings = get_all_system_settings()
    default_tone = str(live_settings.get("default_tone") or "casual_hinglish")

    # Graduated autonomy promotion banners
    eligible = get_contacts_eligible_for_graduated_autonomy()
    for el in eligible:
        st.info(
            f"🎓 **Smart Trust Promotion:** `{el['canonical_name']}` has "
            f"**{el['unchanged_approvals_count']} consecutive unchanged draft approvals**. "
            "You can confidently promote them to `auto` mode below."
        )

    with st.expander("➕ Add New Contact / Pre-Authorize Handle"):
        with st.form("new_contact_form"):
            ac1, ac2 = st.columns(2)
            with ac1:
                new_h = st.text_input("Handle / Username (without @)", placeholder="e.g. aman99")
                new_dn = st.text_input("Display Name", placeholder="e.g. Aman Verma")
                new_pl = st.selectbox("Platform", list(VALID_PLATFORMS), index=0)
                new_md = st.selectbox("Autonomy Mode", list(VALID_MODES), index=0)
            with ac2:
                new_tr = st.selectbox(
                    "Relationship Tier",
                    list(VALID_TIERS),
                    index=0,
                    format_func=lambda t: TIER_LABELS.get(t, t),
                )
                def_tn_idx = list(VALID_TONES).index(default_tone) if default_tone in VALID_TONES else 0
                new_tn = st.selectbox(
                    "Reply Tone Style",
                    list(VALID_TONES),
                    index=def_tn_idx,
                    format_func=lambda tn: TONE_LABELS.get(tn, tn),
                )
                new_pr = st.text_input("Custom Persona Directive (optional)", placeholder="e.g. Keep answers under 10 words")
                new_ft = st.checkbox("Include Disclosure Footer", value=(new_tr == "unknown"))

            if st.form_submit_button("💾 Save Contact", type="primary"):
                if new_h.strip():
                    payload = {
                        "handle": new_h.strip().lower(),
                        "app_source": new_pl,
                        "display_name": new_dn.strip() or new_h.strip(),
                        "mode": new_md,
                        "relationship_tier": new_tr,
                        "tone": new_tn,
                        "custom_persona": new_pr.strip(),
                        "footer_enabled": new_ft,
                    }
                    if _api_call("POST", "/contacts", payload) is None:
                        c_obj = get_or_create_contact(handle=new_h, app_source=new_pl, display_name=new_dn or new_h)
                        update_contact_settings(
                            int(c_obj["id"]),
                            display_name=new_dn or c_obj["display_name"],
                            mode=new_md,
                            relationship_tier=new_tr,
                            tone=new_tn,
                            custom_persona=new_pr,
                            footer_enabled=new_ft,
                        )
                    st.success(f"Contact @{new_h.strip().lower()} saved ({new_tr} / {new_tn}) in '{new_md}' mode!")
                    st.rerun()

    contacts = list_contacts()
    if not contacts:
        st.caption("No contacts registered yet. Add one above or receive a Telegram message to auto-register.")
        return

    for c in contacts:
        cid = int(c["id"])
        disp = html.escape(str(c.get("display_name") or ""))
        canon = html.escape(str(c.get("canonical_name") or ""))
        mode_val = str(c.get("mode") or "off")
        tier_val = str(c.get("relationship_tier") or "unknown")
        tone_val = str(c.get("tone") or "casual_hinglish")
        is_flagged = bool(c.get("flagged_for_review"))
        is_paused = bool(c.get("paused_until"))

        # Keep widget session state aligned with DB whenever the DB row updates
        db_ver = str(c.get("updated_at") or "")
        if st.session_state.get(f"c_ver_{cid}") != db_ver:
            st.session_state[f"sel_m_{cid}"] = mode_val if mode_val in VALID_MODES else VALID_MODES[0]
            st.session_state[f"sel_t_{cid}"] = tier_val if tier_val in VALID_TIERS else VALID_TIERS[0]
            st.session_state[f"sel_tn_{cid}"] = tone_val if tone_val in VALID_TONES else VALID_TONES[0]
            st.session_state[f"chk_f_{cid}"] = bool(c.get("footer_enabled"))
            st.session_state[f"inp_p_{cid}"] = str(c.get("custom_persona") or "")
            st.session_state[f"c_ver_{cid}"] = db_ver

        mode_badge = (
            '<span class="ec-badge badge-green">MODE: AUTO</span>' if mode_val == "auto"
            else ('<span class="ec-badge badge-indigo">MODE: DRAFT</span>' if mode_val == "draft"
            else ('<span class="ec-badge badge-cyan">MODE: SHADOW</span>' if mode_val == "shadow"
            else '<span class="ec-badge badge-gray">MODE: OFF</span>'))
        )
        tier_badge = f'<span class="ec-badge badge-indigo">RELATION: {html.escape(tier_val.upper())}</span>'
        tone_badge = f'<span class="ec-badge badge-cyan">TONE: {html.escape(tone_val.upper())}</span>'
        flag_badge = '<span class="ec-badge badge-amber">🟡 FLAGGED (AI PAUSED)</span>' if is_flagged else ""
        pause_badge = '<span class="ec-badge badge-amber">🟡 MANUAL TAKEOVER ACTIVE</span>' if is_paused else ""

        with st.container(border=True):
            st.markdown(
                f"**{disp}** (<code>@{canon}</code>) • {mode_badge}{tier_badge}{tone_badge}{flag_badge}{pause_badge}",
                unsafe_allow_html=True,
            )

            if is_flagged:
                f_col1, f_col2 = st.columns([4, 1])
                with f_col1:
                    st.markdown(f'<div class="ec-flag-banner">🟡 <b>Ambiguity / Context Hold Active:</b> {html.escape(str(c.get("flag_reason") or "Needs review"))}</div>', unsafe_allow_html=True)
                with f_col2:
                    if st.button("✓ Unflag Now", key=f"btn_unflag_{cid}", use_container_width=True):
                        if _api_call("POST", f"/contacts/{cid}/unflag") is None:
                            clear_contact_review_flag(cid)
                        st.rerun()

            if is_paused:
                p_col1, p_col2 = st.columns([4, 1])
                with p_col1:
                    st.warning(f"🟡 **Manual Takeover Active:** Auto-replies paused until `{c.get('paused_until')}` because you typed in this chat.")
                with p_col2:
                    if st.button("▶️ Resume AI", key=f"btn_resum_{cid}", use_container_width=True):
                        if _api_call("POST", f"/contacts/{cid}/resume") is None:
                            clear_kill_switch(cid)
                        st.rerun()

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                c_mode = st.selectbox("Autonomy Mode", list(VALID_MODES), key=f"sel_m_{cid}")
            with sc2:
                c_tier = st.selectbox(
                    "Relationship Tier",
                    list(VALID_TIERS),
                    key=f"sel_t_{cid}",
                    format_func=lambda t: TIER_LABELS.get(t, t),
                    on_change=_on_contact_tier_changed,
                    args=(cid,),
                )
            with sc3:
                c_tone = st.selectbox(
                    "Reply Tone Style",
                    list(VALID_TONES),
                    key=f"sel_tn_{cid}",
                    format_func=lambda tn: TONE_LABELS.get(tn, tn),
                )
            with sc4:
                c_foot = st.checkbox("Include Disclosure Footer", key=f"chk_f_{cid}")

            c_pers = st.text_input("Custom Persona / Behaviour Rules", key=f"inp_p_{cid}")

            has_unsaved = (
                c_mode != mode_val
                or c_tier != tier_val
                or c_tone != tone_val
                or bool(c_foot) != bool(c.get("footer_enabled"))
                or c_pers.strip() != str(c.get("custom_persona") or "").strip()
            )
            if has_unsaved:
                st.info("📝 Unsaved relation/tone/mode changes — click **Save Settings** below to apply.")

            b_c1, b_c2 = st.columns([3, 1])
            with b_c1:
                if st.button("💾 Save Settings for @" + c['canonical_name'], key=f"btn_save_c_{cid}", type="primary", use_container_width=True):
                    upd_payload = {
                        "mode": c_mode,
                        "relationship_tier": c_tier,
                        "tone": c_tone,
                        "custom_persona": c_pers,
                        "footer_enabled": c_foot,
                    }
                    if _api_call("PATCH", f"/contacts/{cid}", upd_payload) is None:
                        update_contact_settings(cid, mode=c_mode, relationship_tier=c_tier, tone=c_tone, custom_persona=c_pers, footer_enabled=c_foot)
                    st.session_state.pop(f"c_ver_{cid}", None)
                    st.success(f"Saved @{c['canonical_name']}: Mode={c_mode}, Relation={c_tier}, Tone={c_tone}!")
                    st.rerun()
            with b_c2:
                if st.button("🧹 Forget Contact", key=f"btn_forget_{cid}", use_container_width=True):
                    if _api_call("DELETE", f"/contacts/{cid}") is None:
                        forget_contact(cid, rag_dir=RAG_DIR)
                    st.warning(f"Contact @{c['canonical_name']} and all history deleted.")
                    st.rerun()

            # Chat History Explorer (Newest to Oldest)
            history_rows = get_chat_history(cid, limit=25)
            with st.expander(f"💬 View Chat History with @{c['canonical_name']} ({len(history_rows)} messages — Newest on top)"):
                if not history_rows:
                    st.caption("No saved chat messages for this contact yet.")
                else:
                    for hm in reversed(history_rows):
                        h_role = str(hm.get("sender_role"))
                        h_ts = str(hm.get("timestamp") or "")[:19]
                        h_txt = html.escape(str(hm.get("content") or "")).replace("\n", "<br>")
                        if h_role == "user":
                            st.markdown(f'<div class="ec-bubble-in"><b>👤 {disp}:</b> <small style="color:#94A3B8;">({h_ts})</small><br>{h_txt}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="ec-bubble-out"><b>🤖 EchoText:</b> <small style="color:#94A3B8;">({h_ts})</small><br>{h_txt}</div>', unsafe_allow_html=True)


# =============================================================================
# Tab 3: Knowledge Notes (RAG) & Expiring Facts
# =============================================================================

def _render_rag_tab() -> None:
    """Render File Uploads (.txt/.pdf), Visibility Scopes, and Expiring Status Facts."""
    st.subheader("📚 Knowledge Notes (RAG) & Expiring Status Facts")
    st.markdown(
        '<p class="ec-desc">Upload documents (class syllabus, schedule, project details) so the AI assistant '
        'can answer specific questions factually instead of guessing.</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### 📤 Upload Reference File (`.txt` or `.pdf`)")
            up_file = st.file_uploader("Select file (max 5 MB)", type=["txt", "pdf"])
            up_scope = st.selectbox(
                "Visibility Scope",
                ["trusted", "contact", "tier", "public"],
                help="'trusted' is hidden from strangers; 'contact' is restricted to one person only.",
            )
            up_target = st.text_input("Target Handle or Tier (if scoped to contact/tier)", placeholder="e.g. aman99 or friend")

            if st.button("📤 Upload & Index Document", type="primary", use_container_width=True) and up_file:
                try:
                    p = save_uploaded_note(up_file.name, up_file.getvalue(), scope_type=up_scope, scope_target=up_target)
                    st.success(f"Indexed note `{p.name}` (Scope: `{up_scope}`)!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Upload rejected: {ex}")

    with col2:
        with st.container(border=True):
            st.markdown("#### ⚡ Inject Temporary Expiring Fact")
            st.caption("Tell EchoText temporary information (e.g. where you are right now) that automatically expires.")
            fact_str = st.text_input("Temporary Status", placeholder="e.g. In DBMS exam till 5 PM; emergency calls only.")
            fact_mins = st.slider("Auto-Expire After (minutes)", min_value=15, max_value=720, value=120)
            fact_tier = st.selectbox("Visible To", ["all_trusted", "public", "friend", "family"])

            if st.button("⚡ Inject Temporary Status", type="primary", use_container_width=True) and fact_str.strip():
                add_injected_fact(fact_str, ttl_minutes=fact_mins, scope_tier=fact_tier)
                st.success("Temporary fact injected and active!")
                st.rerun()

            active_facts = get_active_injected_facts(relationship_tier="friend")
            if active_facts:
                st.markdown("**Currently Active Facts:**")
                for af in active_facts:
                    fc1, fc2 = st.columns([4, 1])
                    with fc1:
                        st.markdown(f'<div class="ec-card"><b>{html.escape(str(af["fact_text"]))}</b><br><small style="color:#94A3B8;">Expires: {af["expires_at"]}</small></div>', unsafe_allow_html=True)
                    with fc2:
                        if st.button("🗑", key=f"del_fact_btn_{af['id']}"):
                            delete_injected_fact(int(af["id"]))
                            st.rerun()

    st.divider()
    st.markdown("#### 📄 Uploaded Files on Disk")
    all_files = list_all_note_files()
    if all_files:
        st.dataframe(all_files, use_container_width=True, hide_index=True)
    else:
        st.caption("No reference documents uploaded yet.")


# =============================================================================
# Tab 4: System Settings & Parameter Tuning
# =============================================================================

def _render_settings_tab() -> None:
    """Render the full parameter control section with defaults, descriptions, and 1-click restore."""
    st.subheader("⚙️ System Parameters & Operational Settings")
    st.markdown(
        '<p class="ec-desc">Fine-tune all response intervals, cooldowns, AI personality rules, and privacy filters. '
        'All changes are saved persistently in SQLite and take effect immediately without server restarts.</p>',
        unsafe_allow_html=True,
    )

    settings = get_all_system_settings()

    # Category 0: UI Theme & Visual Mode
    with st.container(border=True):
        st.markdown("#### 🎨 Dashboard Theme & Visual Mode")
        st.caption("Toggle between executive Dark Slate theme and high-contrast Clean White mode.")
        current_theme = st.session_state.get("app_theme", settings.get("theme", "dark"))
        selected_theme = st.radio(
            "Color Theme Mode",
            options=["Dark Slate (Executive Dark)", "Clean White (High-Contrast Light)"],
            index=0 if current_theme == "dark" else 1,
            horizontal=True,
            key="settings_theme_radio",
        )
        theme_code = "dark" if "Dark" in selected_theme else "light"
        if theme_code != current_theme:
            st.session_state.app_theme = theme_code
            set_setting("theme", theme_code)
            st.rerun()

    # Category 1: Timing & Response Cadence
    with st.container(border=True):
        st.markdown("#### ⏱️ Timing, Intervals & Auto-Reply Cadence")
        st.caption("Controls how quickly EchoText checks Telegram, types, and spaces out replies.")

        t_col1, t_col2 = st.columns(2)
        with t_col1:
            val_poll = st.number_input(
                "Listener Polling Interval (seconds)",
                min_value=0.5,
                max_value=15.0,
                value=float(settings.get("poll_interval_seconds", 2.0)),
                step=0.5,
                help="How frequently the background worker checks for new incoming Telegram messages. Default: 2.0 seconds.",
            )
            val_auto_delay = st.number_input(
                "Auto-Send Grace Delay (seconds)",
                min_value=0,
                max_value=300,
                value=int(settings.get("auto_send_delay_seconds", 10)),
                step=1,
                help="Grace window before an auto-reply is sent. Set to 10s to allow cancelling. Set to 0 for instant send. Default: 10s.",
            )
            val_cooldown = st.number_input(
                "Contact Cooldown Period (seconds)",
                min_value=0,
                max_value=3600,
                value=int(settings.get("cooldown_seconds", 60)),
                step=5,
                help="Minimum rest time before EchoText will auto-reply to the same contact again. Default: 60s.",
            )

        with t_col2:
            val_multipart_gap = st.number_input(
                "Multi-Part Reply Bubble Gap (seconds)",
                min_value=0.1,
                max_value=5.0,
                value=float(settings.get("multipart_gap_seconds", 0.35)),
                step=0.05,
                help="Pause duration between sending Part 1 and Part 2 of long messages. Default: 0.35s.",
            )
            current_ks_sec = int(settings.get("kill_switch_seconds") or round(float(settings.get("kill_switch_hours", 2.0)) * 3600))
            val_kill_switch_sec = st.number_input(
                "Manual Takeover Pause (seconds)",
                min_value=5,
                max_value=86400,
                value=current_ks_sec,
                step=10,
                help="Duration in seconds auto-replies are paused when you manually reply to a contact from your phone (e.g. 7200s = 2 hrs, 60s = 1 min). Default: 7200 seconds.",
            )
            st.caption(f"⏱️ Pause duration: **{val_kill_switch_sec}s** (~{val_kill_switch_sec / 60:.1f} mins / {val_kill_switch_sec / 3600:.2f} hrs)")
            val_max_replies = st.number_input(
                "Hourly Rate Limit (replies / hour / contact)",
                min_value=1,
                max_value=500,
                value=int(settings.get("max_replies_per_hour", 5)),
                step=1,
                help="Burst protection cap per contact to prevent runaway loops. Default: 5 replies/hour.",
            )

    # Category 2: AI Persona & Personality
    with st.container(border=True):
        st.markdown("#### 🤖 AI Assistant Persona, Tone & Options")
        st.caption("Customize how the AI introduces itself and sounds to your contacts.")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            val_owner_name = st.text_input(
                "Owner / Your Name",
                value=str(settings.get("owner_name", OWNER_NAME)),
                help="Your name as referenced in system prompts and default disclosure footers.",
            )
            val_timezone = st.text_input(
                "Timezone",
                value=str(settings.get("timezone_name", TIMEZONE_NAME)),
                help="System timezone name (e.g. Asia/Kolkata).",
            )
            val_num_drafts = st.slider(
                "Candidate Reply Options Generated",
                min_value=1,
                max_value=5,
                value=int(settings.get("num_draft_options", 3)),
                help="Number of candidate responses generated for each draft review. Default: 3.",
            )

        with p_col2:
            val_footer = st.text_input(
                "Disclosure Footer Text",
                value=str(settings.get("default_footer_text", DEFAULT_FOOTER_TEXT)),
                help="Text appended to messages sent to strangers or contacts with footers enabled.",
            )
            cur_def_tone = str(settings.get("default_tone", "casual_hinglish"))
            tone_idx = list(VALID_TONES).index(cur_def_tone) if cur_def_tone in VALID_TONES else 0
            val_default_tone = st.selectbox("Default Reply Tone Style", list(VALID_TONES), index=tone_idx)

    # Category 3: Privacy, Safety Shield & Active Hours
    with st.container(border=True):
        st.markdown("#### 🛡️ Privacy Shield, Safety Gate & Active Hours")
        st.caption("Protect sensitive information and limit replies to working hours.")

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            val_sens = st.slider(
                "Privacy Shield Sensitivity Threshold (1–3)",
                min_value=1,
                max_value=3,
                value=int(settings.get("sensitivity_threshold", SENSITIVITY_THRESHOLD)),
                help="1: Strict (holds financial/schedule questions). 2: Balanced (holds money/transfers). 3: Lenient (holds passwords/OTPs/card numbers only). Default: 2.",
            )
            val_retention = st.number_input(
                "Data Retention Limit (days)",
                min_value=1,
                max_value=365,
                value=int(settings.get("retention_days", RETENTION_DAYS)),
                help="Days to keep chat history, audit logs, and old drafts before purge. Default: 30 days.",
            )
            val_redact = st.checkbox(
                "Redact Message Text in Console & Server Logs",
                value=bool(settings.get("hide_message_text", True)),
                help="Mask message content in logs for privacy. Default: True.",
            )

        with s_col2:
            val_active_hours_enabled = st.checkbox(
                "Enable Active Working Hours Filter",
                value=bool(settings.get("active_hours_enabled", False)),
                help="When enabled, auto-replies only dispatch during the hours specified below.",
            )
            ah_c1, ah_c2 = st.columns(2)
            with ah_c1:
                val_ah_start = st.number_input("Start Hour (0–23)", min_value=0, max_value=23, value=int(settings.get("active_hours_start", 8)))
            with ah_c2:
                val_ah_end = st.number_input("End Hour (0–23)", min_value=0, max_value=23, value=int(settings.get("active_hours_end", 23)))

    # Category 4: Hardware & Model Profile
    with st.container(border=True):
        st.markdown("#### ⚡ Hardware Profile & Local AI Engine")
        st.caption("Select and verify your local GGUF inference model. Models are automatically discovered from `EchoText/models/`.")

        available_models = get_available_models()
        model_filenames = [m["filename"] for m in available_models]

        cur_stored_path = str(settings.get("model_path", "")).strip()
        cur_active_fname = Path(cur_stored_path).name if cur_stored_path else MODEL_PATH.name

        if cur_active_fname in model_filenames:
            default_model_idx = model_filenames.index(cur_active_fname)
        elif "echotext-qwen3-8b-Q4_K_M.gguf" in model_filenames:
            default_model_idx = model_filenames.index("echotext-qwen3-8b-Q4_K_M.gguf")
        else:
            default_model_idx = 0

        h_col1, h_col2 = st.columns(2)
        with h_col1:
            if available_models:
                selected_model_idx = st.selectbox(
                    "🤖 Active AI Model (GGUF)",
                    range(len(available_models)),
                    format_func=lambda idx: available_models[idx]["label"],
                    index=default_model_idx,
                    help="Choose which model EchoText uses for drafting and auto-replies.",
                )
                chosen_model = available_models[selected_model_idx]
                val_model_path = f"models/{chosen_model['filename']}"
                st.caption(f"📁 Path: `{chosen_model['path']}` • Size: **{chosen_model['size_gb']:.1f} GB**")
            else:
                st.warning("⚠️ No `.gguf` models found in `EchoText/models/`.")
                val_model_path = st.text_input("Custom GGUF Model Path", value=cur_stored_path)

            prof_options = list(MODEL_PROFILES.keys())
            # Auto-suggest profile based on model size
            suggested_profile = "pc_8b" if (available_models and "qwen3" in chosen_model["filename"].lower()) else HARDWARE_PROFILE
            cur_prof = str(settings.get("hardware_profile", suggested_profile))
            prof_idx = prof_options.index(cur_prof) if cur_prof in prof_options else 0
            val_profile = st.selectbox("Hardware Optimization Profile", prof_options, index=prof_idx)
            st.caption(f"Profile: `{MODEL_PROFILES[val_profile]['description']}`")

            val_gpu_layers = st.number_input(
                "GPU Offload Layers (0 for CPU only)",
                min_value=0,
                max_value=60,
                value=int(settings.get("gpu_layers", 0)),
                help="Set to 20-33 if you have an NVIDIA GPU to accelerate inference. Default: 0.",
            )

        with h_col2:
            st.markdown("##### 🧪 Test Model Live Inference")
            st.caption("Verify how your selected model replies to different tones before saving.")
            test_preset = st.selectbox(
                "Quick Test Preset",
                [
                    "Friend Casual Hinglish: 'bhai kya scene hai aaj shaam ko?'",
                    "Professor Formal English: 'Dear Aman, please share your progress report by 5 PM.'",
                    "Family Warm Hindi: 'beta khana kha liya? kab tak ghar aaoge?'",
                    "Colleague Semi-Formal: 'Can we sync up at 4 PM for the code review?'",
                    "Custom Message...",
                ],
                index=0,
            )
            if test_preset == "Custom Message...":
                custom_test_msg = st.text_input("Enter incoming message to test", value="bhai kal movie chalte hai?")
            else:
                custom_test_msg = test_preset.split(": ", 1)[-1].strip("'\"")

            if st.button("🚀 Test Selected Model Reply", use_container_width=True):
                with st.spinner("Generating test reply with local model..."):
                    # Check via server API first
                    test_res = _call_server_api(
                        "POST",
                        "/api/model/test",
                        {"prompt": custom_test_msg},
                        timeout=30.0,
                    )
                    if test_res and test_res.get("status") == "ok":
                        st.success(f"✅ Generated in {test_res.get('latency_ms', 0):.0f} ms via `{test_res.get('model_name')}`:")
                        st.markdown(
                            f"""
                            <div style="background: rgba(30, 41, 59, 0.9); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 10px; padding: 12px 16px; margin: 8px 0;">
                                <span style="font-size: 0.85rem; color: #94A3B8;">Incoming:</span> <i>{html.escape(custom_test_msg)}</i><br>
                                <span style="font-size: 0.85rem; color: #818CF8; font-weight: 600;">EchoText Reply:</span><br>
                                <span style="font-size: 1.05rem; color: #F1F5F9;">{html.escape(test_res.get('reply', ''))}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        # Direct local inference fallback if Brain server is currently stopped
                        try:
                            from llama_cpp import Llama
                            import time
                            target_p = Path(val_model_path)
                            if not target_p.is_absolute():
                                target_p = (BASE_DIR / target_p).resolve()
                            t0 = time.perf_counter()
                            llm_temp = Llama(
                                model_path=str(target_p),
                                n_ctx=1024,
                                n_threads=int(MODEL_PROFILES[val_profile]["n_threads"]),
                                n_gpu_layers=int(val_gpu_layers),
                                verbose=False,
                            )
                            resp = llm_temp.create_chat_completion(
                                messages=[
                                    {"role": "system", "content": f"You are EchoText, {OWNER_NAME}'s reply assistant."},
                                    {"role": "user", "content": custom_test_msg},
                                ],
                                max_tokens=60,
                                stop=["<|im_end|>", "<|im_start|>", "<|eot_id|>", "<|end_of_text|>"],
                            )
                            lat_ms = round((time.perf_counter() - t0) * 1000, 1)
                            raw_reply = resp["choices"][0]["message"]["content"]
                            from intelligence import clean_llm_output
                            clean_rep = clean_llm_output(raw_reply)
                            st.success(f"✅ Generated in {lat_ms:.0f} ms via `{target_p.name}`:")
                            st.markdown(
                                f"""
                                <div style="background: rgba(30, 41, 59, 0.9); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 10px; padding: 12px 16px; margin: 8px 0;">
                                    <span style="font-size: 0.85rem; color: #94A3B8;">Incoming:</span> <i>{html.escape(custom_test_msg)}</i><br>
                                    <span style="font-size: 0.85rem; color: #818CF8; font-weight: 600;">EchoText Reply:</span><br>
                                    <span style="font-size: 1.05rem; color: #F1F5F9;">{html.escape(clean_rep)}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        except Exception as exc:
                            st.error(f"❌ Test inference failed: {exc}")

    # Category 5: Mobile Approval Bot (Telegram Bot API)
    with st.container(border=True):
        st.markdown("#### 📱 Mobile Approval Bot (Remote Phone Control)")
        st.caption("Receive approval buttons directly on your personal phone via Telegram.")

        mb_c1, mb_c2 = st.columns(2)
        with mb_c1:
            val_bot_token = st.text_input(
                "Telegram Approval Bot Token",
                value=str(settings.get("approval_bot_token", "")),
                type="password",
                placeholder="e.g. 123456789:ABCdefGhIJKlmNoPQRstuv",
            )
        with mb_c2:
            val_owner_tg_id = st.text_input(
                "Owner Telegram Numeric ID",
                value=str(settings.get("owner_telegram_id", "")),
                placeholder="e.g. 987654321",
            )

    # Action Buttons: Save & Reset
    act_c1, act_c2 = st.columns([3, 1])
    with act_c1:
        if st.button("💾 Save All Parameter Changes", type="primary", use_container_width=True):
            updated_dict = {
                "poll_interval_seconds": float(val_poll),
                "auto_send_delay_seconds": int(val_auto_delay),
                "cooldown_seconds": int(val_cooldown),
                "multipart_gap_seconds": float(val_multipart_gap),
                "kill_switch_seconds": int(val_kill_switch_sec),
                "kill_switch_hours": round(val_kill_switch_sec / 3600.0, 4),
                "max_replies_per_hour": int(val_max_replies),
                "owner_name": str(val_owner_name).strip() or "Abhay",
                "timezone_name": str(val_timezone).strip() or "Asia/Kolkata",
                "default_footer_text": str(val_footer).strip(),
                "num_draft_options": int(val_num_drafts),
                "default_tone": str(val_default_tone),
                "sensitivity_threshold": int(val_sens),
                "retention_days": int(val_retention),
                "hide_message_text": bool(val_redact),
                "active_hours_enabled": bool(val_active_hours_enabled),
                "active_hours_start": int(val_ah_start),
                "active_hours_end": int(val_ah_end),
                "hardware_profile": str(val_profile),
                "gpu_layers": int(val_gpu_layers),
                "model_path": str(val_model_path).strip(),
                "approval_bot_token": str(val_bot_token).strip(),
                "owner_telegram_id": str(val_owner_tg_id).strip(),
                "theme": str(st.session_state.get("app_theme", "dark")),
            }
            save_system_settings(updated_dict)
            set_sensitivity_threshold(int(val_sens))
            # Notify live backend server to switch model immediately
            if val_model_path:
                _call_server_api("POST", "/api/model/switch", {"model_filename": str(val_model_path).strip()})
            st.success("✅ All system parameters updated and applied immediately!")
            st.rerun()

    with act_c2:
        if st.button("🔄 Restore Recommended Defaults", use_container_width=True):
            reset_system_settings_to_defaults()
            st.warning("Restored all parameters to factory defaults (e.g. 2.0s poll interval, 10s auto-delay, 60s cooldown).")
            st.rerun()


# =============================================================================
# Tab 5: Privacy, Shadow Mode & Security Audit
# =============================================================================

def _render_privacy_and_audit_tab() -> None:
    """Render Shadow Mode Evaluation, Data Purge, and the Security Audit Trail."""
    st.subheader("🛡️ Safety Filters, Shadow Mode & Decision Audit Trail")
    st.markdown(
        '<p class="ec-desc">Inspect every autonomous action, evaluate silent shadow drafts, and enforce data privacy.</p>',
        unsafe_allow_html=True,
    )

    settings = get_all_system_settings()
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### 🔒 Privacy Shield Summary")
            current_thresh = get_sensitivity_threshold()
            st.markdown(
                f"**Current Sensitivity Threshold:** `{current_thresh}/3`<br>"
                "• **Score 1:** Mild schedule / timing questions<br>"
                "• **Score 2:** Financial references, payment requests<br>"
                "• **Score 3:** Passwords, OTPs, Card numbers (Always blocked automatically)",
                unsafe_allow_html=True,
            )
    with c2:
        with st.container(border=True):
            st.markdown("#### 🧹 Manual Data Retention Purge")
            ret_days = int(settings.get("retention_days", 30))
            if st.button(f"🧹 Purge Records Older Than {ret_days} Days", use_container_width=True):
                res = purge_old_records(retention_days=ret_days)
                st.success(f"Purge complete: {res.get('messages_deleted', 0)} messages, {res.get('drafts_deleted', 0)} drafts deleted.")
                st.rerun()

    st.markdown("#### 👤 Shadow Mode Evaluation (Generated Silently, Never Sent)")
    shadows = list_drafts(statuses=["shadow"], limit=20)
    if shadows:
        st.dataframe(
            [
                {
                    "Draft ID": f"#{s['id']}",
                    "Contact": f"@{s.get('canonical_name')}",
                    "Triage": str(s.get("triage_label") or "").upper(),
                    "Incoming Message": s.get("incoming_text"),
                    "Shadow Reply": s.get("draft_text"),
                    "Timestamp": str(s.get("created_at") or "")[:19],
                }
                for s in shadows
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No shadow mode logs recorded yet.")

    st.markdown("#### 📜 Security Decision Audit Trail")
    audit_logs = list_audit_logs(limit=40)
    if audit_logs:
        st.dataframe(
            [
                {
                    "Time (UTC)": str(lg.get("created_at") or "")[:19],
                    "Event": lg.get("event_type"),
                    "Contact": f"@{lg['canonical_name']}" if lg.get("canonical_name") else "System",
                    "Mode": lg.get("mode_used") or "-",
                    "Triage": lg.get("triage_label") or "-",
                    "Sensitivity": f"{lg.get('sensitivity_score', 0)}/3",
                    "Outcome": lg.get("outcome"),
                    "Details": lg.get("details"),
                }
                for lg in audit_logs
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Audit log is currently empty.")


# =============================================================================
# Main Dashboard Entrypoint
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title="EchoText Command Center",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _init_session_state()
    _inject_custom_css()
    tracker: AuthTracker = st.session_state.auth_tracker

    # 1. Gate with Registration & Login portal
    if not _render_auth_portal(tracker):
        return

    # 2. Dynamic Real-Time Live Header (KPIs + Notifications)
    _render_live_header_and_kpis(tracker)
    st.divider()

    # 3. Clean, Professional, Intuitive Tabs
    tab_inbox, tab_contacts, tab_rag, tab_settings, tab_privacy = st.tabs(
        [
            "📥 Message Approvals",
            "👥 Contact Manager",
            "📚 Knowledge & Notes",
            "⚙️ Settings & Timing",
            "🛡️ Privacy & Audit",
        ]
    )

    with tab_inbox:
        _render_approval_inbox_tab()
    with tab_contacts:
        _render_contacts_tab()
    with tab_rag:
        _render_rag_tab()
    with tab_settings:
        _render_settings_tab()
    with tab_privacy:
        _render_privacy_and_audit_tab()


if __name__ == "__main__":
    main()
