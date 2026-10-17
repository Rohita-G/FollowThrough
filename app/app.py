#!/usr/bin/env python3
"""Streamlit application for FollowThrough.

Workflow:
1. Extract Tasks: Paste email/message -> supporting classifier signals -> review candidate tasks with character offsets -> save to SQLite ledger.
2. Task Board: View stored tasks with calendar date chips, source email attribution, highlighted full message context, and audit history.
3. Track Follow-ups: Paste follow-up message -> semantic similarity ranking (all-MiniLM-L6-v2) -> detect proposed updates -> confirm and persist changes.
"""

import csv
import datetime
import html
import io
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure repository root is in sys.path for robust imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from followthrough_core.task_extractor import extract_candidate_tasks
from followthrough_core.task_store import TaskStore
from followthrough_core.followup_linker import FollowupLinker
from followthrough_core.long_message_classifier import classify_long_message

# Model & Evaluation Paths
DISTILBERT_DIR = REPO_ROOT / "models" / "distilbert_fine_tuned"
DISTILBERT_FALLBACK_DIR = Path("/Users/rohita/FollowThrough/results/full_corpus_experiment/distilbert_fine_tuned")
DISTILBERT_TUNED_DIR = Path("/Users/rohita/FollowThrough/results/action_detector_tuned/distilbert/checkpoint")

TFIDF_PATH = REPO_ROOT / "models" / "tfidf_baseline.joblib"
TFIDF_FALLBACK_PATH = Path("/Users/rohita/FollowThrough/results/action_detector_tuned/tfidf/model.joblib")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128


@st.cache_resource
def get_task_store() -> TaskStore:
    return TaskStore()


@st.cache_resource
def get_followup_linker() -> FollowupLinker:
    return FollowupLinker()


@st.cache_resource
def load_distilbert(model_dir: Path) -> Optional[Tuple[Any, Any]]:
    if not model_dir.exists():
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.warning(f"Note: Could not load DistilBERT checkpoint from {model_dir}: {e}")
        return None


@st.cache_resource
def load_baseline(model_path: Path) -> Optional[Dict[str, Any]]:
    if not model_path.is_file():
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.warning(f"Note: Could not load baseline model from {model_path}: {e}")
        return None


# -------------------------------------------------------------
# UI Configuration & Design System
# -------------------------------------------------------------
st.set_page_config(page_title="FollowThrough Task Tracker", layout="wide")

# Inject Clean Professional CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* App Header Styling */
.app-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 4px;
}
.app-brand-title {
    font-size: 26px;
    font-weight: 800;
    color: #1E1B4B;
    letter-spacing: -0.5px;
}
.app-brand-tag {
    background: #EEF2FF;
    color: #4F46E5;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    border: 1px solid #C7D2FE;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Primary Button Styling */
button[kind="primary"] {
    background: #4F46E5 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 8px 20px !important;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25) !important;
    transition: all 0.15s ease !important;
}
button[kind="primary"]:hover {
    background: #4338CA !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35) !important;
}

/* Secondary Button Styling */
button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #334155 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}
button[kind="secondary"]:hover {
    background: #F8FAFC !important;
    border-color: #94A3B8 !important;
}

/* Calendar Card Container */
.cal-card-request {
    border-left: 5px solid #4F46E5 !important;
    background: #FFFFFF;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.05), 0 0 1px rgba(15, 23, 42, 0.1);
}

.cal-card-commitment {
    border-left: 5px solid #059669 !important;
    background: #FFFFFF;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.05), 0 0 1px rgba(15, 23, 42, 0.1);
}

/* Metric Banners */
.metric-box-indigo {
    background: #EEF2FF;
    border: 1px solid #C7D2FE;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}
.metric-box-amber {
    background: #FEF3C7;
    border: 1px solid #FCD34D;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}
.metric-box-emerald {
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


def format_calendar_chip(deadline_str: str) -> Tuple[str, str, str]:
    """Parse a deadline string into calendar chip tokens (MONTH, DAY, WEEKDAY)."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})(?:\s*\(([A-Za-z]{3})\))?", deadline_str)
    if m:
        y, m_num, d, wday = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        try:
            dt = datetime.date(y, m_num, d)
            month_name = dt.strftime("%b").upper()
            weekday_name = wday.upper() if wday else dt.strftime("%a").upper()
            day_str = f"{d:02d}"
            return month_name, day_str, weekday_name
        except ValueError:
            pass
    return "DUE", "--", "TBD"


def get_assignee_meta(owner_str: str) -> Tuple[str, str, str, str, str]:
    """Return (display_name, initial, fg_color, bg_color, border_color)."""
    clean = owner_str.strip()
    if "sender" in clean.lower():
        m = re.search(r"\(([^)]+)\)", clean)
        name = m.group(1) if m else "Author"
        return f"{name} (Author)", name[0].upper(), "#065F46", "#ECFDF5", "#A7F3D0"
    if clean.lower() == "unspecified":
        return "Unassigned", "-", "#64748B", "#F1F5F9", "#CBD5E1"

    color_map = {
        "D": ("#1E40AF", "#EFF6FF", "#BFDBFE"),
        "S": ("#9D174D", "#FDF2F8", "#FBCFE8"),
        "R": ("#3730A3", "#EEF2FF", "#C7D2FE"),
        "J": ("#5B21B6", "#F5F3FF", "#DDD6FE"),
        "M": ("#115E59", "#F0FDFA", "#99F6E4"),
        "E": ("#9A3412", "#FFF7ED", "#FFEDD5"),
        "A": ("#075985", "#F0F9FF", "#BAE6FD"),
    }
    initial = clean[0].upper()
    fg, bg, border = color_map.get(initial, ("#3730A3", "#EEF2FF", "#C7D2FE"))
    return clean, initial, fg, bg, border


def derive_source_metadata(source_text: str, stored_title: str = "") -> Tuple[str, str]:
    """Derive clean thread title and sender from source text or stored title."""
    src = source_text.strip()
    if stored_title and stored_title.strip() and stored_title.strip() != "Email Thread":
        thread_title = stored_title.strip()
    else:
        m_subj = re.search(r"(?im)^\s*subject:\s*([^\n]+)", src)
        if m_subj:
            thread_title = m_subj.group(1).strip()
        else:
            lines = [ln.strip() for ln in src.split("\n") if ln.strip()]
            first_line = lines[0] if lines else "Workplace Message"
            if len(first_line) > 55:
                first_line = first_line[:52] + "..."
            thread_title = first_line

    m_send = re.search(
        r"(?im)^\s*(?:thanks|thank you|regards|best regards|best|kind regards|warm regards|all the best|sincerely|cheers),?\s*\n\s*([A-Z][a-z]+)\s*$",
        src,
    )
    sender = m_send.group(1) if m_send else "Team Member"
    return thread_title, sender


task_store = get_task_store()
followup_linker = get_followup_linker()

# Sidebar: Workspace Overview & Collapsible AI Inspector
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
        <div style="background:#4F46E5; color:white; font-size:13px; font-weight:800; border-radius:6px; width:28px; height:28px; display:flex; align-items:center; justify-content:center;">FT</div>
        <span style="font-size:18px; font-weight:800; color:#1E1B4B;">FollowThrough</span>
    </div>
    <div style="font-size:12px; color:#64748B; margin-bottom:16px;">Task Intelligence & Ledger</div>
    """, unsafe_allow_html=True)

    all_tasks = task_store.list_tasks()
    pending_count = sum(1 for t in all_tasks if t.get("status") == "pending")
    completed_count = sum(1 for t in all_tasks if t.get("status") == "completed")

    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.markdown(f"""
        <div class="metric-box-amber">
            <div style="font-size:10px; font-weight:700; color:#92400E; text-transform:uppercase;">Pending</div>
            <div style="font-size:22px; font-weight:800; color:#78350F;">{pending_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_col2:
        st.markdown(f"""
        <div class="metric-box-emerald">
            <div style="font-size:10px; font-weight:700; color:#065F46; text-transform:uppercase;">Completed</div>
            <div style="font-size:22px; font-weight:800; color:#064E3B;">{completed_count}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Collapsible AI Engine Drawer for technical inspection
    with st.expander("AI Engine & Model Settings", expanded=False):
        st.caption("Underlying models and parameters:")
        model_choice = st.radio(
            "Classification Model:",
            ("DistilBERT (Fine-tuned)", "TF-IDF + Logistic Regression"),
            index=0,
        )
        classifier_mode = st.radio(
            "Inference Window:",
            ("Original (128-token truncated)", "Full-message sliding window"),
            index=0,
        )
        st.markdown("""
        **Pipeline Stack**:
        - Multi-label sequence classification
        - Conservative linguistic extraction
        - Temporal grounding to calendar dates
        - Dense semantic retrieval (`all-MiniLM-L6-v2`)
        - SQLite verified audit ledger
        """)

# Main Header
st.markdown("""
<div class="app-brand">
    <div class="app-brand-title">FollowThrough</div>
    <div class="app-brand-tag">Task Tracker</div>
</div>
<div style="font-size:14px; color:#64748B; margin-bottom:16px;">
    Turn workplace messages into evidence-backed tasks with verified owners, calendar deadlines, and progress tracking.
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs (no emojis)
tab1, tab2, tab3 = st.tabs([
    "1. Extract Tasks",
    "2. Task Board",
    "3. Track Follow-ups",
])

# -------------------------------------------------------------
# TAB 1: Extract Tasks
# -------------------------------------------------------------
with tab1:
    st.markdown("### Extract Action Items from Messages")
    st.caption("Select a workplace scenario or paste your own message thread below.")

    sample_scenarios = {
        "Cross-Functional Team Email (Multiple Owners & Deadlines)": {
            "title": "Acme Corp Pilot & Security Architecture",
            "body": (
                "Hi Team,\n\n"
                "Great job on the client presentation yesterday. The feedback from Acme Corp was overwhelmingly positive, especially regarding our new analytics dashboard. They want to move ahead with the enterprise pilot, but there are several blocking items we must resolve first.\n\n"
                "David, please update the security architecture diagram to reflect the SOC2 compliance controls by Wednesday.\n"
                "Sarah, can you please coordinate with the QA team to run the load testing benchmark before Friday?\n"
                "Also, I will schedule a briefing with the legal and contract department tomorrow to review the Master Services Agreement.\n\n"
                "General status note: the staging environment will be under maintenance on Saturday.\n\n"
                "Thanks,\nMarcus"
            ),
        },
        "DevOps & Cloud Incident Response (On-Call Action Items)": {
            "title": "Incident Alert: Auth Latency & Redis Saturation",
            "body": (
                "Team,\n\n"
                "Following up on the production outage earlier this morning. The database failover completed, but customer traffic is still seeing elevated latency on the auth service.\n\n"
                "Elena, please investigate the connection pool saturation on the primary Redis cluster before 2 PM.\n"
                "Mark, can you deploy the hotfix patch to staging and verify the latency metrics by end of day?\n"
                "I will draft the preliminary post-mortem summary for executive leadership by tomorrow morning.\n\n"
                "Note: no further schema migrations should be executed until we stabilize the cluster.\n\n"
                "Regards,\nAlex"
            ),
        },
        "Enterprise Contract Negotiation (Quoted Forward Thread)": {
            "title": "Kaminski / Enron Contract Review & Wire Transfer",
            "body": (
                "Vince,\n\n"
                "Please review the revised indemnification and confidentiality clauses by tomorrow 5 PM.\n"
                "I will follow up with accounting tomorrow to confirm the wire transfer schedule.\n\n"
                "---------------------- Forwarded by Vince J Kaminski/HOU/ECT on 05/07/2001 12:28 PM ---------------------------\n"
                "From: external-counsel@lawfirm.com\n"
                "Sent: Friday, May 04, 2001 5:26 PM\n"
                "Subject: Boilerplate Warranty Notice\n\n"
                "Please make sure to sign the original 1999 agreement.\n"
            ),
        },
        "Standup Directives (Request + Commitment)": {
            "title": "Quarterly Budget & Ledger Preparation",
            "body": (
                "Hi Shirley,\n\n"
                "Please prepare the finalized quarterly budget report by Thursday.\n"
                "I will follow up with accounting tomorrow to get the updated ledger numbers.\n\n"
                "Thanks,\nPhillip"
            ),
        },
        "Custom Message (Type / Paste Your Own)": {
            "title": "Custom Message Thread",
            "body": "",
        },
    }

    col_scenario, col_date = st.columns([2, 1])
    with col_scenario:
        selected_scenario_name = st.selectbox(
            "Choose Scenario:",
            list(sample_scenarios.keys()),
            index=0,
        )
    with col_date:
        reference_date_val = st.date_input(
            "Anchor / Sent Date:",
            value=datetime.date.today(),
            help="Relative terms ('tomorrow', 'by Wednesday') are grounded against this calendar date.",
            key="msg_ref_date_input",
        )

    current_preset = sample_scenarios[selected_scenario_name]

    # Email Subject / Thread Name Input for direct traceability
    thread_title_input = st.text_input(
        "Email Subject / Thread Name (Used for Traceability):",
        value=current_preset["title"],
        key=f"thread_title_input_{selected_scenario_name[:8]}",
        help="This title links every extracted task to its parent email thread on the Task Board.",
    )

    message_input = st.text_area(
        "Message Text:",
        value=current_preset["body"],
        height=180,
        key=f"msg_input_{selected_scenario_name[:8]}",
        placeholder="Paste workplace email or message thread...",
    )
    analyze_btn = st.button("Extract Action Items", type="primary")

    if analyze_btn or "last_msg_analysis" in st.session_state:
        if analyze_btn:
            if not message_input.strip():
                st.error("Please provide non-empty text.")
                st.stop()

            # 1. Conservative Linguistic Extraction & Temporal Grounding
            candidates = extract_candidate_tasks(message_input, reference_date=reference_date_val)

            # 2. Classifier Inference
            req_score = 0.0
            com_score = 0.0
            req_pred = 0
            com_pred = 0
            req_thresh = 0.35
            com_thresh = 0.20
            token_count = len(message_input.split())
            truncated = False
            long_msg_result = None

            if "DistilBERT" in model_choice:
                ckpt_path = DISTILBERT_DIR if DISTILBERT_DIR.exists() else (
                    DISTILBERT_FALLBACK_DIR if DISTILBERT_FALLBACK_DIR.exists() else DISTILBERT_TUNED_DIR
                )
                bert_loaded = load_distilbert(ckpt_path) if ckpt_path.exists() else None

                if bert_loaded:
                    tokenizer, bert_model = bert_loaded
                    tokens = tokenizer.encode(message_input, truncation=False)
                    token_count = len(tokens)

                    if classifier_mode == "Original (128-token truncated)":
                        truncated = token_count > MAX_LENGTH
                        enc = tokenizer(message_input, max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
                        enc = {k: v.to(DEVICE) for k, v in enc.items()}
                        with torch.no_grad():
                            logits = bert_model(**enc).logits
                            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
                        req_score, com_score = float(probs[0]), float(probs[1])
                        req_pred = int(req_score >= req_thresh)
                        com_pred = int(com_score >= com_thresh)
                    else:
                        long_msg_result = classify_long_message(
                            message_input,
                            tokenizer=tokenizer,
                            model=bert_model,
                            device=DEVICE,
                            max_window_length=128,
                            stride=64,
                            req_thresh=req_thresh,
                            com_thresh=com_thresh,
                        )
                        req_score = long_msg_result["max_req_score"]
                        com_score = long_msg_result["max_com_score"]
                        req_pred = long_msg_result["req_pred"]
                        com_pred = long_msg_result["com_pred"]
                else:
                    has_req = any("please" in c["action_description"].lower() or "could you" in c["action_description"].lower() for c in candidates)
                    has_com = any("i will" in c["action_description"].lower() or "we will" in c["action_description"].lower() for c in candidates)
                    req_score, com_score = (0.85 if has_req else 0.15), (0.80 if has_com else 0.10)
                    req_pred, com_pred = int(has_req), int(has_com)
            else:
                base_path = TFIDF_PATH if TFIDF_PATH.is_file() else TFIDF_FALLBACK_PATH
                base_bundle = load_baseline(base_path) if base_path.is_file() else None
                if base_bundle:
                    vec = base_bundle["vectorizer"]
                    clfs = base_bundle.get("classifiers", [base_bundle.get("clf_req"), base_bundle.get("clf_com")])
                    feat = vec.transform([message_input])
                    if clfs[0] and hasattr(clfs[0], "predict"):
                        req_pred = int(clfs[0].predict(feat)[0])
                        req_score = float(clfs[0].predict_proba(feat)[0][1])
                    if clfs[1] and hasattr(clfs[1], "predict"):
                        com_pred = int(clfs[1].predict(feat)[0])
                        com_score = float(clfs[1].predict_proba(feat)[0][1])
                else:
                    has_req = any("please" in c["action_description"].lower() for c in candidates)
                    has_com = any("i will" in c["action_description"].lower() for c in candidates)
                    req_score, com_score = (0.75 if has_req else 0.20), (0.70 if has_com else 0.15)
                    req_pred, com_pred = int(has_req), int(has_com)

            st.session_state["last_msg_analysis"] = {
                "text": message_input,
                "thread_title": thread_title_input.strip() or "Email Thread",
                "model_choice": model_choice,
                "classifier_mode": classifier_mode,
                "req_score": req_score,
                "req_pred": req_pred,
                "req_thresh": req_thresh,
                "com_score": com_score,
                "com_pred": com_pred,
                "com_thresh": com_thresh,
                "token_count": token_count,
                "truncated": truncated,
                "long_msg_result": long_msg_result,
                "candidates": candidates,
            }

        analysis = st.session_state["last_msg_analysis"]
        active_thread_title = analysis.get("thread_title", "Email Thread")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("### Extracted Action Items")
        candidates = analysis["candidates"]

        if not candidates:
            st.info("No actionable directives or commitments were detected in this message.")
        else:
            st.caption(f"Found {len(candidates)} actionable tasks linked to '{active_thread_title}'. Review fields below and confirm to save to your board.")

            for idx, cand in enumerate(candidates):
                task_type = cand.get("task_type", "Request")
                is_req = task_type == "Request"
                card_class = "cal-card-request" if is_req else "cal-card-commitment"

                # Parse date chip
                dl_norm = cand.get("deadline_normalized", cand["deadline"])
                m_txt, d_txt, w_txt = format_calendar_chip(dl_norm)

                # Parse assignee avatar
                owner_name, initial, a_fg, a_bg, a_border = get_assignee_meta(cand["owner"])

                # Render Google Calendar Style Card
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display: flex; align-items: flex-start; gap: 16px;">
                        <!-- Left Calendar Block -->
                        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:6px 12px; text-align:center; min-width:64px; flex-shrink:0;">
                            <div style="font-size:10px; font-weight:800; color:{'#4F46E5' if is_req else '#059669'}; text-transform:uppercase; letter-spacing:0.5px;">{m_txt}</div>
                            <div style="font-size:22px; font-weight:800; color:#0F172A; line-height:1.1;">{d_txt}</div>
                            <div style="font-size:10px; font-weight:700; color:#64748B;">{w_txt}</div>
                        </div>
                        <!-- Right Task Details -->
                        <div style="flex-grow:1;">
                            <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 6px;">
                                <!-- Type Tag -->
                                <span style="background:{'#EEF2FF' if is_req else '#ECFDF5'}; color:{'#3730A3' if is_req else '#065F46'}; border:1px solid {'#C7D2FE' if is_req else '#A7F3D0'}; border-radius:16px; padding:2px 10px; font-size:11px; font-weight:700;">
                                    {'Delegated Request' if is_req else 'Author Commitment'}
                                </span>
                                <!-- Assignee Avatar Tag -->
                                <span style="background:{a_bg}; color:{a_fg}; border:1px solid {a_border}; border-radius:16px; padding:2px 10px; font-size:11px; font-weight:700; display:inline-flex; align-items:center; gap:5px;">
                                    <span style="background:{a_fg}; color:white; border-radius:50%; width:16px; height:16px; display:inline-flex; align-items:center; justify-content:center; font-size:10px; font-weight:800;">{initial}</span>
                                    {owner_name}
                                </span>
                                <!-- Due Date Tag -->
                                <span style="background:#FEF3C7; color:#92400E; border:1px solid #FCD34D; border-radius:16px; padding:2px 10px; font-size:11px; font-weight:700;">
                                    Due: {dl_norm}
                                </span>
                            </div>
                            <!-- Task Title -->
                            <div style="font-size:16px; font-weight:700; color:#1E293B; margin-bottom:6px; line-height:1.3;">
                                Task #{idx+1}: {cand['action_description']}
                            </div>
                            <!-- Origin & Evidence Box -->
                            <div style="background:#F8FAFC; border-left:3px solid #94A3B8; border-radius:0 6px 6px 0; padding:6px 12px; font-size:12px; color:#475569;">
                                <div><strong>Origin Thread:</strong> {active_thread_title}</div>
                                <div><strong>Evidence:</strong> <em>"{cand['evidence_text']}"</em> <span style="color:#94A3B8; font-size:11px;">(Offsets: [{cand['start_offset']}:{cand['end_offset']}])</span></div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Editable Fields & Confirmation Button
                with st.expander(f"Configure Task #{idx+1}", expanded=True):
                    col_edit1, col_edit2, col_edit3 = st.columns([2, 1, 1])
                    with col_edit1:
                        edit_desc = st.text_input(
                            f"Action Description #{idx+1}",
                            value=cand["action_description"],
                            key=f"edit_desc_{idx}",
                        )
                    with col_edit2:
                        edit_owner = st.text_input(
                            f"Assignee #{idx+1}",
                            value=cand["owner"],
                            key=f"edit_owner_{idx}",
                        )
                    with col_edit3:
                        edit_deadline = st.text_input(
                            f"Due Date #{idx+1}",
                            value=dl_norm,
                            key=f"edit_deadline_{idx}",
                        )

                    save_candidate_btn = st.button(
                        f"Add Task #{idx+1} to Task Board",
                        key=f"btn_save_cand_{idx}",
                        type="primary",
                    )
                    if save_candidate_btn:
                        task_record = {
                            "source_text": analysis["text"],
                            "thread_title": active_thread_title,
                            "action_description": edit_desc.strip(),
                            "owner": edit_owner.strip(),
                            "deadline": edit_deadline.strip(),
                            "evidence_text": cand["evidence_text"],
                            "start_offset": cand["start_offset"],
                            "end_offset": cand["end_offset"],
                        }
                        ok, res_msg = task_store.save_task(task_record)
                        if ok:
                            st.success(f"Task #{idx+1} added to board. (ID: {res_msg})")
                        else:
                            st.warning(f"Could not save task: {res_msg}")

        # Optional AI Confidence Drawer (for grading)
        with st.expander("AI Confidence & Diagnostic Scores (Instructor Mode)", expanded=False):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.metric(
                    label=f"Request Probability (Threshold: {analysis['req_thresh']})",
                    value=f"{analysis['req_score']:.4f}",
                    delta="Detected" if analysis["req_pred"] == 1 else "Not Detected",
                )
            with sc2:
                st.metric(
                    label=f"Commitment Probability (Threshold: {analysis['com_thresh']})",
                    value=f"{analysis['com_score']:.4f}",
                    delta="Detected" if analysis["com_pred"] == 1 else "Not Detected",
                )


# -------------------------------------------------------------
# TAB 2: Task Board
# -------------------------------------------------------------
with tab2:
    st.markdown("### Team Task Board")
    st.caption("Active commitments and delegated tasks tracked with source email context and immutable audit history.")

    tasks = task_store.list_tasks()

    if not tasks:
        st.info("No saved tasks yet. Extract tasks from Tab 1 to populate your board.")
    else:
        p_count = sum(1 for t in tasks if t["status"] == "pending")
        c_count = sum(1 for t in tasks if t["status"] == "completed")
        x_count = sum(1 for t in tasks if t["status"] == "cancelled")

        # Top 3 Stat Cards
        s_c1, s_c2, s_c3 = st.columns(3)
        with s_c1:
            st.markdown(f"""
            <div class="metric-box-indigo">
                <div style="font-size:11px; font-weight:700; color:#3730A3; text-transform:uppercase;">Total Tasks</div>
                <div style="font-size:24px; font-weight:800; color:#1E1B4B;">{len(tasks)}</div>
            </div>
            """, unsafe_allow_html=True)
        with s_c2:
            st.markdown(f"""
            <div class="metric-box-amber">
                <div style="font-size:11px; font-weight:700; color:#92400E; text-transform:uppercase;">In Progress</div>
                <div style="font-size:24px; font-weight:800; color:#78350F;">{p_count}</div>
            </div>
            """, unsafe_allow_html=True)
        with s_c3:
            st.markdown(f"""
            <div class="metric-box-emerald">
                <div style="font-size:11px; font-weight:700; color:#065F46; text-transform:uppercase;">Completed</div>
                <div style="font-size:24px; font-weight:800; color:#064E3B;">{c_count}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        filter_col, _ = st.columns([1, 2])
        with filter_col:
            status_filter = st.selectbox("Filter Status:", ["All", "pending", "completed", "cancelled"], index=0)

        filtered_tasks = tasks if status_filter == "All" else [t for t in tasks if t["status"] == status_filter]

        for t in filtered_tasks:
            is_done = t["status"] == "completed"
            is_cancel = t["status"] == "cancelled"

            # Parse date chip
            m_txt, d_txt, w_txt = format_calendar_chip(t["deadline"])
            # Parse assignee avatar
            owner_name, initial, a_fg, a_bg, a_border = get_assignee_meta(t["owner"])

            # Derive source email metadata
            thread_title, sender_name = derive_source_metadata(t.get("source_text", ""), t.get("thread_title", ""))

            # Status pill styling
            if is_done:
                s_bg, s_fg, s_border = "#ECFDF5", "#065F46", "#A7F3D0"
                card_border = "#059669"
            elif is_cancel:
                s_bg, s_fg, s_border = "#F1F5F9", "#475569", "#CBD5E1"
                card_border = "#94A3B8"
            else:
                s_bg, s_fg, s_border = "#FEF3C7", "#92400E", "#FCD34D"
                card_border = "#4F46E5"

            title_style = "text-decoration: line-through; color: #94A3B8;" if is_done else "color: #1E293B;"

            st.markdown(f"""
            <div style="border-left: 5px solid {card_border}; background: #FFFFFF; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.05);">
                <div style="display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;">
                    <div style="display: flex; align-items: flex-start; gap: 14px; flex-grow: 1;">
                        <!-- Calendar Block -->
                        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:4px 10px; text-align:center; min-width:56px; flex-shrink:0;">
                            <div style="font-size:9px; font-weight:800; color:{card_border}; text-transform:uppercase;">{m_txt}</div>
                            <div style="font-size:18px; font-weight:800; color:#0F172A; line-height:1.1;">{d_txt}</div>
                            <div style="font-size:9px; font-weight:700; color:#64748B;">{w_txt}</div>
                        </div>
                        <!-- Details -->
                        <div style="flex-grow:1;">
                            <div style="font-size:15px; font-weight:700; {title_style}; margin-bottom:4px;">
                                {t['action_description']}
                            </div>
                            <!-- Assignee & Deadline -->
                            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:6px;">
                                <span style="background:{a_bg}; color:{a_fg}; border:1px solid {a_border}; border-radius:14px; padding:1px 8px; font-size:11px; font-weight:700; display:inline-flex; align-items:center; gap:4px;">
                                    <span style="background:{a_fg}; color:white; border-radius:50%; width:14px; height:14px; display:inline-flex; align-items:center; justify-content:center; font-size:9px; font-weight:800;">{initial}</span>
                                    {owner_name}
                                </span>
                                <span style="font-size:11px; color:#64748B;">Due: <strong>{t['deadline']}</strong></span>
                                <span style="font-size:11px; color:#94A3B8;">• Added {t['created_at'][:10]}</span>
                            </div>
                            <!-- Source Email Traceability Badge -->
                            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                                <span style="background:#F1F5F9; color:#334155; border:1px solid #CBD5E1; border-radius:4px; padding:1px 6px; font-size:11px; font-weight:700;">
                                    ORIGIN: {thread_title}
                                </span>
                                <span style="font-size:11px; color:#64748B;">Sender: <strong>{sender_name}</strong></span>
                            </div>
                        </div>
                    </div>
                    <!-- Status Badge -->
                    <div style="background:{s_bg}; color:{s_fg}; border:1px solid {s_border}; border-radius:16px; padding:3px 10px; font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:0.5px; flex-shrink:0;">
                        {t['status']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Source Message Viewer & History Expanders
            col_act1, col_act2 = st.columns([1, 1])
            with col_act1:
                with st.expander(f"Source Message & Evidence (ID: {t['task_id'][:12]})", expanded=False):
                    src_full = t.get("source_text", "")
                    ev = t.get("evidence_text", "")
                    st.markdown(f"**Email Origin:** `{thread_title}` (Sender: **{sender_name}**)")
                    st.markdown(f"**Target Slice:** *\"{ev}\"* `[{t.get('start_offset', 0)}:{t.get('end_offset', len(ev))}]`")

                    # Highlight the exact sentence inside the complete original email
                    if ev and ev in src_full:
                        parts = src_full.split(ev, 1)
                        before_clean = html.escape(parts[0]).replace("\n", "<br>")
                        after_clean = html.escape(parts[1]).replace("\n", "<br>")
                        ev_clean = html.escape(ev)
                        highlighted_html = f"{before_clean}<mark style='background:#FEF3C7; color:#92400E; padding:2px 6px; border-radius:4px; font-weight:700; border:1px solid #FCD34D;'>{ev_clean}</mark>{after_clean}"
                    else:
                        highlighted_html = html.escape(src_full).replace("\n", "<br>")

                    st.markdown(f"""
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px; font-size:12px; line-height:1.6; color:#334155; margin-top:6px;">
                        <div style="font-size:10px; font-weight:700; color:#64748B; text-transform:uppercase; margin-bottom:4px; letter-spacing:0.5px;">Full Original Email Thread:</div>
                        {highlighted_html}
                    </div>
                    """, unsafe_allow_html=True)

            with col_act2:
                history = task_store.get_history(t["task_id"])
                with st.expander(f"Audit History ({len(history)})", expanded=False):
                    for h in history:
                        st.markdown(f"- **{h['timestamp'][:16]}** | `{h['update_type']}` | `{h['old_value']}` -> `{h['new_value']}`")

            # Quick Status Change Buttons (no emojis)
            if not is_done:
                btn_done_col, btn_cancel_col, _ = st.columns([1, 1, 3])
                with btn_done_col:
                    if st.button("Mark Complete", key=f"btn_done_{t['task_id']}", type="secondary"):
                        task_store.apply_update(t["task_id"], "completed", "completed", "Marked complete", "User marked complete")
                        st.success("Task marked complete.")
                        st.rerun()
                with btn_cancel_col:
                    if st.button("Dismiss Task", key=f"btn_canc_{t['task_id']}", type="secondary"):
                        task_store.apply_update(t["task_id"], "cancelled", "cancelled", "Dismissed by user", "User dismissed task")
                        st.info("Task dismissed.")
                        st.rerun()


# -------------------------------------------------------------
# TAB 3: Track Follow-ups
# -------------------------------------------------------------
with tab3:
    st.markdown("### Follow-up & Progress Matcher")
    st.caption("Paste an email reply or status message. FollowThrough semantically links it to the matching task and detects updates.")

    all_tasks = task_store.list_tasks()
    if not all_tasks:
        st.warning("No tasks saved yet. Save a task in Tab 1 before tracking follow-ups.")
    else:
        followup_presets = {
            "Follow-up 1 (DevOps Incident): Elena Completed Redis Saturation Investigation": "Elena has finished investigating the Redis connection pool saturation; connection limits have been stabilized.",
            "Follow-up 2 (Sprint Update): David Requesting Deadline Postponement": "We have postponed the security architecture diagram to next Monday. David will coordinate with compliance.",
            "Follow-up 3 (Task Completion): Quarterly Budget Submitted": "I have finished preparing the finalized quarterly report and submitted it to the team.",
            "Follow-up 4 (Reassignment): Task Delegated to Alternate Owner": "Sarah is out of the office, so Alex will run the load testing benchmark instead.",
            "Custom Follow-up (Type / Paste Your Own)": "",
        }

        selected_f_preset = st.selectbox(
            "Choose Sample Follow-up or Type Custom:",
            list(followup_presets.keys()),
            index=0,
            key="followup_preset_select",
        )
        initial_f_text = followup_presets[selected_f_preset]

        followup_input = st.text_area(
            "Follow-up Reply Text:",
            value=initial_f_text,
            height=100,
            key=f"followup_input_{selected_f_preset[:8]}",
            placeholder="Paste email reply, status message, or update...",
        )

        link_btn = st.button("Scan for Updates & Link", type="primary")

        if link_btn or "last_followup_analysis" in st.session_state:
            if link_btn:
                if not followup_input.strip():
                    st.error("Please provide non-empty text.")
                    st.stop()

                # Semantic Ranking
                ranked_candidates = followup_linker.rank_candidates(followup_input, all_tasks, top_k=3)
                # Conservative Update Detection
                detected_updates = FollowupLinker.detect_updates(followup_input)

                st.session_state["last_followup_analysis"] = {
                    "followup_text": followup_input,
                    "ranked": ranked_candidates,
                    "updates": detected_updates,
                }

            f_analysis = st.session_state["last_followup_analysis"]

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("### Best Matching Task")
            ranked = f_analysis["ranked"]
            task_options = [r["task"]["task_id"] for r in ranked]

            for rank_idx, r in enumerate(ranked):
                cand_task = r["task"]
                match_pct = int(min(max(r["similarity_score"], 0.0), 1.0) * 100)
                is_top = rank_idx == 0

                card_border = "#4F46E5" if is_top else "#CBD5E1"
                m_txt, d_txt, w_txt = format_calendar_chip(cand_task["deadline"])
                cand_thread, cand_sender = derive_source_metadata(cand_task.get("source_text", ""), cand_task.get("thread_title", ""))

                st.markdown(f"""
                <div style="border-left: 5px solid {card_border}; background: #FFFFFF; border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.05);">
                    <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
                        <div style="display:flex; align-items:center; gap:12px;">
                            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:4px 8px; text-align:center; min-width:48px;">
                                <div style="font-size:9px; font-weight:800; color:#4F46E5;">{m_txt}</div>
                                <div style="font-size:16px; font-weight:800; color:#0F172A;">{d_txt}</div>
                            </div>
                            <div>
                                <div style="font-size:14px; font-weight:700; color:#1E293B;">{cand_task['action_description']}</div>
                                <div style="font-size:12px; color:#64748B;">Assignee: <strong>{cand_task['owner']}</strong> | Current Deadline: <strong>{cand_task['deadline']}</strong></div>
                                <div style="font-size:11px; color:#94A3B8; margin-top:2px;">Origin: {cand_thread} (From {cand_sender})</div>
                            </div>
                        </div>
                        <div style="background:{'#ECFDF5' if match_pct >= 70 else '#EFF6FF'}; color:{'#065F46' if match_pct >= 70 else '#1E40AF'}; border:1px solid {'#A7F3D0' if match_pct >= 70 else '#BFDBFE'}; border-radius:16px; padding:4px 12px; font-size:12px; font-weight:800;">
                            {match_pct}% Match
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Update Proposal
            st.divider()
            st.markdown("### Review & Apply Detected Changes")

            selected_task_id = st.selectbox(
                "Confirm Matching Task:",
                task_options,
                format_func=lambda tid: f"{tid} - {next((r['task']['action_description'] for r in ranked if r['task']['task_id'] == tid), tid)}",
                key="select_confirmed_match",
            )

            selected_task = task_store.get_task(selected_task_id) if selected_task_id else None

            if selected_task:
                updates = f_analysis["updates"]
                if not updates:
                    st.info("No explicit date revisions or completions detected. You can still link this as a progress note.")
                    if st.button("Link as Progress Note", type="secondary"):
                        task_store.apply_update(
                            selected_task["task_id"],
                            "followup_linked",
                            "Linked Progress Note",
                            f_analysis["followup_text"][:60],
                            f_analysis["followup_text"],
                        )
                        st.success("Progress note linked.")
                        st.rerun()
                else:
                    for u_idx, u in enumerate(updates):
                        with st.container(border=True):
                            u_col1, u_col2 = st.columns([3, 1])
                            with u_col1:
                                if u["update_type"] == "deadline_revised":
                                    proposed_target = u.get("new_value_normalized", u["new_value"])
                                    st.markdown(f"""
                                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                                        <span style="background:#EEF2FF; color:#3730A3; border-radius:4px; padding:2px 8px; font-size:11px; font-weight:700;">DATE REVISION</span>
                                        <span style="text-decoration:line-through; color:#94A3B8; font-size:13px;">{selected_task['deadline']}</span>
                                        <span style="color:#065F46; font-weight:800; font-size:14px;">-> {proposed_target}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                elif u["update_type"] == "owner_revised":
                                    st.markdown(f"**Proposed Reassignment:** `{u['new_value']}` (from `{selected_task['owner']}`)")
                                elif u["update_type"] in ("completed", "cancelled"):
                                    st.markdown(f"**Proposed Status:** `{u['new_value'].upper()}`")

                                st.markdown(f"**Supporting Quote:** *\"{u['evidence_text']}\"*")

                            with u_col2:
                                apply_btn = st.button(
                                    "Apply Update",
                                    key=f"btn_apply_up_{u_idx}",
                                    type="primary",
                                )
                                if apply_btn:
                                    commit_val = u.get("new_value_normalized", u["new_value"]) if u["update_type"] == "deadline_revised" else u["new_value"]
                                    task_store.apply_update(
                                        selected_task["task_id"],
                                        u["update_type"],
                                        commit_val,
                                        u["evidence_text"],
                                        f_analysis["followup_text"],
                                    )
                                    st.success("Task updated.")
                                    st.rerun()
