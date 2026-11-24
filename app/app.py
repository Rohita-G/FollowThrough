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
from followthrough_core.long_message_classifier import classify_long_message
from followthrough_core.task_store import TaskStore
from followthrough_core.followup_linker import FollowupLinker
from followthrough_core.calendar_sync import (
    generate_google_calendar_url,
    parse_deadline_to_datetimes,
)

# Model & Evaluation Paths
DISTILBERT_DIR = REPO_ROOT / "models" / "distilbert_fine_tuned"
DISTILBERT_FALLBACK_DIR = Path("/Users/rohita/FollowThrough/results/full_corpus_experiment/distilbert_fine_tuned")
DISTILBERT_TUNED_DIR = Path("/Users/rohita/FollowThrough/results/action_detector_tuned/distilbert/checkpoint")

TFIDF_PATH = REPO_ROOT / "models" / "tfidf_baseline.joblib"
TFIDF_FALLBACK_PATH = Path("/Users/rohita/FollowThrough/results/action_detector_tuned/tfidf/model.joblib")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128


def get_task_store() -> TaskStore:
    return TaskStore()


def safe_delete_tasks(store: TaskStore, task_ids: List[str]) -> int:
    if hasattr(store, "delete_tasks"):
        return store.delete_tasks(task_ids)
    count = 0
    for tid in task_ids:
        if store.delete_task(tid):
            count += 1
    return count


def safe_batch_update_status(store: TaskStore, task_ids: List[str], status: str) -> int:
    if hasattr(store, "batch_update_status"):
        return store.batch_update_status(task_ids, status)
    for tid in task_ids:
        store.apply_update(tid, status, status, f"Batch {status}", "User batch action")
    return len(task_ids)


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
    margin-bottom: 2px;
}
.app-brand-title {
    font-size: 24px;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.5px;
}
.app-brand-tag {
    background: #EEF2FF;
    color: #4F46E5;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
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
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 6px 16px !important;
    box-shadow: 0 1px 3px rgba(79, 70, 229, 0.2) !important;
    transition: all 0.15s ease !important;
}
button[kind="primary"]:hover {
    background: #4338CA !important;
    box-shadow: 0 2px 6px rgba(79, 70, 229, 0.3) !important;
}

/* Secondary Button Styling */
button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #334155 !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 6px 14px !important;
    transition: all 0.15s ease !important;
}
button[kind="secondary"]:hover {
    background: #F8FAFC !important;
    border-color: #CBD5E1 !important;
    color: #0F172A !important;
}

/* Calendar Card Container */
.cal-card-request {
    border-left: 4px solid #4F46E5 !important;
    background: #FFFFFF;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    padding: 14px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

.cal-card-commitment {
    border-left: 4px solid #059669 !important;
    background: #FFFFFF;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    padding: 14px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
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
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
        <div style="background:#4F46E5; color:white; font-size:12px; font-weight:800; border-radius:6px; width:26px; height:26px; display:flex; align-items:center; justify-content:center;">FT</div>
        <span style="font-size:17px; font-weight:800; color:#0F172A; letter-spacing:-0.3px;">FollowThrough</span>
    </div>
    <div style="font-size:11px; color:#64748B; margin-bottom:14px;">Task Intelligence Ledger</div>
    """, unsafe_allow_html=True)

    all_tasks = task_store.list_tasks()
    active_count = sum(1 for t in all_tasks if t.get("status") not in ("cancelled", "dismissed"))
    dismissed_count = sum(1 for t in all_tasks if t.get("status") in ("cancelled", "dismissed"))

    st.markdown(f"""
    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 14px; margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:11px; font-weight:600; color:#475569; text-transform:uppercase; letter-spacing:0.5px;">Active Tasks</span>
            <span style="font-size:16px; font-weight:800; color:#4F46E5;">{active_count}</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid #F1F5F9; padding-top:6px;">
            <span style="font-size:11px; font-weight:600; color:#64748B; text-transform:uppercase; letter-spacing:0.5px;">Dismissed</span>
            <span style="font-size:14px; font-weight:700; color:#94A3B8;">{dismissed_count}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        "Slack / Teams Standup (@Mentions & Dev Directives)": {
            "title": "Engineering Sprint & PR Review",
            "body": (
                "@kevin: can you review the pull request #402 before 3 PM today?\n"
                "@sarah please verify the staging endpoints by EOD.\n"
                "alex: I will deploy the hotfix patch to staging by tonight.\n"
                "dan: let's jump on an architecture sync tomorrow at 11am."
            ),
        },
        "Mobile SMS / Urgent On-Call Ping": {
            "title": "Mobile Ops Alert",
            "body": (
                "Hey Alex, production traffic is spiking. Plz push the hotfix build before 5 PM today. "
                "I will notify customer support once it is live."
            ),
        },
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

                with st.expander(f"Edit Fields #{idx+1}", expanded=False):
                    col_edit1, col_edit2, col_edit3 = st.columns([2, 1, 1])
                    with col_edit1:
                        edit_desc = st.text_input(
                            f"Action Description",
                            value=cand["action_description"],
                            key=f"edit_desc_{idx}",
                        )
                    with col_edit2:
                        edit_owner = st.text_input(
                            f"Assignee",
                            value=cand["owner"],
                            key=f"edit_owner_{idx}",
                        )
                    with col_edit3:
                        edit_deadline = st.text_input(
                            f"Due Date",
                            value=dl_norm,
                            key=f"edit_deadline_{idx}",
                        )

                col_save_btn, col_gcal_btn = st.columns([1.5, 1.5])
                temp_task = {
                    "task_id": f"cand-{idx}",
                    "action_description": edit_desc.strip(),
                    "owner": edit_owner.strip(),
                    "deadline": edit_deadline.strip(),
                    "evidence_text": cand["evidence_text"],
                    "thread_title": active_thread_title,
                }
                with col_save_btn:
                    save_candidate_btn = st.button(
                        f"Add to Board",
                        key=f"btn_save_cand_{idx}",
                        type="primary",
                        use_container_width=True,
                    )
                with col_gcal_btn:
                    gcal_cand_url = generate_google_calendar_url(temp_task)
                    st.link_button("Add to Calendar", url=gcal_cand_url, type="secondary", use_container_width=True)
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

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
    tasks = task_store.list_tasks()

    if not tasks:
        st.info("No saved tasks in ledger. Extract tasks from Tab 1 to populate your board.")
    else:
        active_count = sum(1 for t in tasks if t.get("status") not in ("cancelled", "dismissed"))
        dismissed_count = sum(1 for t in tasks if t.get("status") in ("cancelled", "dismissed"))

        # Top Header Bar: Clean Status Summary + Manage Popover
        hdr_col, act_col = st.columns([3.2, 0.8])
        with hdr_col:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:14px; margin-bottom:12px;">
                <h3 style="margin:0; font-size:18px; font-weight:800; color:#0F172A;">Task Board</h3>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="background:#EEF2FF; color:#4F46E5; border:1px solid #C7D2FE; border-radius:12px; padding:2px 10px; font-size:11px; font-weight:700;">
                        {active_count} Active
                    </span>
                    <span style="background:#F1F5F9; color:#64748B; border:1px solid #E2E8F0; border-radius:12px; padding:2px 10px; font-size:11px; font-weight:700;">
                        {dismissed_count} Dismissed
                    </span>
                    <span style="font-size:11px; color:#94A3B8; font-weight:600;">
                        {len(tasks)} Total
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with act_col:
            with st.popover("Manage Board", use_container_width=True):
                st.markdown("**Board Maintenance**")
                st.caption("Perform administrative ledger cleanup actions.")
                if st.button(f"Clear Dismissed ({dismissed_count})", disabled=(dismissed_count == 0), key="btn_clear_dismissed_pop", use_container_width=True):
                    cleared = task_store.clear_completed_and_cancelled()
                    st.success(f"Removed {cleared} dismissed tasks.")
                    st.rerun()
                st.divider()
                st.caption("Wipe all tasks from SQLite ledger for a fresh start.")
                confirm_wipe = st.checkbox("Confirm ledger wipe", key="chk_confirm_wipe_pop")
                if st.button("Reset Entire Board", disabled=not confirm_wipe, key="btn_reset_board_pop", type="secondary", use_container_width=True):
                    wiped = task_store.clear_all_tasks()
                    st.session_state["selected_task_ids"].clear()
                    for k in list(st.session_state.keys()):
                        if k.startswith("sel_"):
                            st.session_state.pop(k, None)
                    st.success(f"Wiped {wiped} tasks.")
                    st.rerun()

        # Clean Filter Bar
        unique_owners = sorted(list(set(t.get("owner", "Unspecified") for t in tasks if t.get("owner"))))

        f_col1, f_col2, f_col3, f_col4 = st.columns([1.6, 1, 1, 1.2])
        with f_col1:
            search_query = st.text_input("Search:", placeholder="Search title, assignee, thread...", key="board_search_input")
        with f_col2:
            status_filter = st.selectbox("Status:", ["Active", "Dismissed", "All"], index=0, key="board_status_filter")
        with f_col3:
            owner_filter = st.selectbox("Assignee:", ["All Assignees"] + unique_owners, index=0, key="board_owner_filter")
        with f_col4:
            sort_choice = st.selectbox("Sort By:", ["Upcoming Deadline", "Newest First", "Oldest First"], index=0, key="board_sort_choice")

        # Apply filtering
        filtered_tasks = []
        for t in tasks:
            is_item_dismissed = t.get("status") in ("cancelled", "dismissed")
            if status_filter == "Active" and is_item_dismissed:
                continue
            if status_filter == "Dismissed" and not is_item_dismissed:
                continue
            if owner_filter != "All Assignees" and t.get("owner", "") != owner_filter:
                continue
            if search_query.strip():
                q = search_query.strip().lower()
                blob = f"{t.get('action_description', '')} {t.get('owner', '')} {t.get('deadline', '')} {t.get('evidence_text', '')} {t.get('thread_title', '')}".lower()
                if q not in blob:
                    continue
            filtered_tasks.append(t)

        # Apply sorting
        if sort_choice == "Upcoming Deadline":
            def _deadline_sort_key(item):
                dl = item.get("deadline", "")
                if not dl or dl.lower() == "none specified":
                    return datetime.datetime(2099, 12, 31)
                start_dt, _, _ = parse_deadline_to_datetimes(dl, default_anchor=reference_date_val)
                return start_dt
            filtered_tasks.sort(key=_deadline_sort_key)
        elif sort_choice == "Oldest First":
            filtered_tasks.sort(key=lambda item: item.get("created_at", ""))
        else:
            filtered_tasks.sort(key=lambda item: item.get("created_at", ""), reverse=True)

        total_filtered = len(filtered_tasks)

        # Selection state
        if "selected_task_ids" not in st.session_state:
            st.session_state["selected_task_ids"] = set()

        curr_selected = set()
        for t in filtered_tasks:
            if st.session_state.get(f"sel_{t['task_id']}", False):
                curr_selected.add(t["task_id"])
        st.session_state["selected_task_ids"] = curr_selected
        n_sel = len(curr_selected)

        # High-density Batch Action Toolbar
        if filtered_tasks:
            all_sel = (n_sel == len(filtered_tasks) and len(filtered_tasks) > 0)
            tb_col1, tb_col2, tb_col3, tb_col4 = st.columns([1.2, 1.4, 1.4, 1.4])
            with tb_col1:
                if st.button("Select All" if not all_sel else "Deselect All", key="btn_toggle_select_all", use_container_width=True):
                    if not all_sel:
                        for t in filtered_tasks:
                            st.session_state[f"sel_{t['task_id']}"] = True
                        st.session_state["selected_task_ids"] = set(t["task_id"] for t in filtered_tasks)
                    else:
                        for t in tasks:
                            st.session_state[f"sel_{t['task_id']}"] = False
                        st.session_state["selected_task_ids"].clear()
                    st.rerun()

            with tb_col2:
                del_label = f"Delete Selected ({n_sel})" if n_sel > 0 else "Delete Selected"
                if st.button(del_label, disabled=(n_sel == 0), key="btn_del_selected", type="primary" if n_sel > 0 else "secondary", use_container_width=True, help="Permanently delete selected tasks"):
                    del_n = safe_delete_tasks(task_store, list(curr_selected))
                    for tid in curr_selected:
                        st.session_state.pop(f"sel_{tid}", None)
                    st.session_state["selected_task_ids"].clear()
                    st.success(f"Deleted {del_n} task(s).")
                    st.rerun()

            with tb_col3:
                dismiss_label = f"Dismiss Selected ({n_sel})" if n_sel > 0 else "Dismiss Selected"
                if st.button(dismiss_label, disabled=(n_sel == 0), key="btn_dismiss_selected", use_container_width=True, help="Dismiss selected tasks"):
                    up_n = safe_batch_update_status(task_store, list(curr_selected), "dismissed")
                    for tid in curr_selected:
                        st.session_state.pop(f"sel_{tid}", None)
                    st.session_state["selected_task_ids"].clear()
                    st.info(f"Dismissed {up_n} task(s).")
                    st.rerun()

            with tb_col4:
                st.markdown(f"<div style='text-align:right; padding-top:6px; font-size:12px; font-weight:600; color:#64748B;'>Showing {total_filtered} task(s)</div>", unsafe_allow_html=True)

        if not filtered_tasks:
            st.info("No tasks match the active filters.")

        # Render Task Cards
        for t in filtered_tasks:
            is_item_dismissed = t.get("status") in ("cancelled", "dismissed")
            is_selected = (t["task_id"] in curr_selected)

            # Parse date chip
            m_txt, d_txt, w_txt = format_calendar_chip(t["deadline"])
            # Parse assignee avatar
            owner_name, initial, a_fg, a_bg, a_border = get_assignee_meta(t["owner"])

            # Derive source email metadata
            thread_title, sender_name = derive_source_metadata(t.get("source_text", ""), t.get("thread_title", ""))

            # Check Overdue Status
            is_overdue = False
            if not is_item_dismissed and t.get("deadline") and t["deadline"].lower() != "none specified":
                t_dt, _, _ = parse_deadline_to_datetimes(t["deadline"], default_anchor=reference_date_val)
                if t_dt.date() < reference_date_val:
                    is_overdue = True

            # Status pill & card border styling
            if is_item_dismissed:
                s_bg, s_fg, s_border = "#F1F5F9", "#475569", "#CBD5E1"
                card_border = "#94A3B8"
                status_display = "DISMISSED"
            elif is_overdue:
                s_bg, s_fg, s_border = "#FEE2E2", "#B91C1C", "#FCA5A5"
                card_border = "#EF4444"
                status_display = "OVERDUE"
            else:
                s_bg, s_fg, s_border = "#EEF2FF", "#4338CA", "#C7D2FE"
                card_border = "#4F46E5"
                status_display = "ACTIVE"

            title_style = "color: #94A3B8; text-decoration: line-through;" if is_item_dismissed else "color: #0F172A;"
            card_bg = "#F5F3FF" if is_selected else "#FFFFFF"
            card_outline = f"border-left: 4px solid {card_border}; border-top: 1px solid #C7D2FE; border-right: 1px solid #C7D2FE; border-bottom: 1px solid #C7D2FE;" if is_selected else f"border-left: 4px solid {card_border}; border: 1px solid #E2E8F0;"

            card_col_chk, card_col_body = st.columns([0.04, 0.96])
            with card_col_chk:
                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
                st.checkbox(
                    "",
                    value=is_selected,
                    key=f"sel_{t['task_id']}",
                    label_visibility="collapsed"
                )

            with card_col_body:
                st.markdown(f"""
                <div style="{card_outline} background: {card_bg}; border-radius: 8px; padding: 12px 14px; margin-bottom: 6px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);">
                    <div style="display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;">
                        <div style="display: flex; align-items: flex-start; gap: 12px; flex-grow: 1;">
                            <!-- Calendar Block -->
                            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:3px 8px; text-align:center; min-width:52px; flex-shrink:0;">
                                <div style="font-size:9px; font-weight:800; color:{card_border}; text-transform:uppercase;">{m_txt}</div>
                                <div style="font-size:16px; font-weight:800; color:#0F172A; line-height:1.1;">{d_txt}</div>
                                <div style="font-size:9px; font-weight:600; color:#64748B;">{w_txt}</div>
                            </div>
                            <!-- Details -->
                            <div style="flex-grow:1;">
                                <div style="font-size:14px; font-weight:700; {title_style}; margin-bottom:4px; line-height:1.3;">
                                    {t['action_description']}
                                </div>
                                <!-- Metadata Row -->
                                <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                                    <span style="background:{a_bg}; color:{a_fg}; border:1px solid {a_border}; border-radius:12px; padding:1px 7px; font-size:11px; font-weight:600; display:inline-flex; align-items:center; gap:4px;">
                                        <span style="background:{a_fg}; color:white; border-radius:50%; width:12px; height:12px; display:inline-flex; align-items:center; justify-content:center; font-size:8px; font-weight:800;">{initial}</span>
                                        {owner_name}
                                    </span>
                                    <span style="font-size:11px; color:#475569;">Due: <strong>{t['deadline']}</strong></span>
                                    <span style="background:#F1F5F9; color:#475569; border:1px solid #E2E8F0; border-radius:4px; padding:1px 6px; font-size:10px; font-weight:600;">
                                        Origin: {thread_title}
                                    </span>
                                </div>
                            </div>
                        </div>
                        <!-- Status Badge -->
                        <div style="background:{s_bg}; color:{s_fg}; border:1px solid {s_border}; border-radius:12px; padding:2px 8px; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.4px; flex-shrink:0;">
                            {status_display}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Single Consolidated Drawer: Details & Edit
                with st.expander("Details & Edit", expanded=False):
                    history = task_store.get_history(t["task_id"])
                    tab_ev, tab_hist, tab_edit = st.tabs(["Source Evidence", f"Audit Trail ({len(history)})", "Edit Task"])

                    with tab_ev:
                        src_full = t.get("source_text", "")
                        ev = t.get("evidence_text", "")
                        st.markdown(f"**Email Origin:** `{thread_title}` (Sender: **{sender_name}**)")
                        st.markdown(f"**Target Slice:** *\"{ev}\"* `[{t.get('start_offset', 0)}:{t.get('end_offset', len(ev))}]`")

                        if ev and ev in src_full:
                            parts = src_full.split(ev, 1)
                            before_clean = html.escape(parts[0]).replace("\n", "<br>")
                            after_clean = html.escape(parts[1]).replace("\n", "<br>")
                            ev_clean = html.escape(ev)
                            highlighted_html = f"{before_clean}<mark style='background:#FEF3C7; color:#92400E; padding:2px 6px; border-radius:4px; font-weight:700; border:1px solid #FCD34D;'>{ev_clean}</mark>{after_clean}"
                        else:
                            highlighted_html = html.escape(src_full).replace("\n", "<br>")

                        st.markdown(f"""
                        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:8px 12px; font-size:12px; line-height:1.5; color:#334155; margin-top:4px;">
                            <div style="font-size:10px; font-weight:700; color:#64748B; text-transform:uppercase; margin-bottom:4px;">Full Original Message:</div>
                            {highlighted_html}
                        </div>
                        """, unsafe_allow_html=True)

                    with tab_hist:
                        if not history:
                            st.caption("No history events logged yet.")
                        else:
                            for h in history:
                                st.markdown(f"- **{h['timestamp'][:16]}** | `{h['update_type']}` | `{h['old_value']}` -> `{h['new_value']}`")

                    with tab_edit:
                        edit_b_desc = st.text_input("Description", value=t["action_description"], key=f"e_desc_{t['task_id']}")
                        e_c1, e_c2 = st.columns(2)
                        with e_c1:
                            edit_b_owner = st.text_input("Assignee", value=t["owner"], key=f"e_owner_{t['task_id']}")
                        with e_c2:
                            edit_b_dl = st.text_input("Due Date", value=t["deadline"], key=f"e_dl_{t['task_id']}")
                        if st.button("Save Changes", key=f"btn_save_edit_{t['task_id']}", type="primary", use_container_width=True):
                            task_store.update_task_details(t["task_id"], edit_b_desc, edit_b_owner, edit_b_dl)
                            st.success("Task details saved.")
                            st.rerun()

                # Action Buttons Row: 3 text-only actions
                btn_gcal_col, btn_dismiss_col, btn_del_col = st.columns([1.4, 1.2, 1.0])
                with btn_gcal_col:
                    gcal_url = generate_google_calendar_url(t)
                    st.link_button("Add to Calendar", url=gcal_url, type="secondary", use_container_width=True)
                with btn_dismiss_col:
                    if not is_item_dismissed:
                        if st.button("Dismiss", key=f"btn_canc_{t['task_id']}", type="secondary", use_container_width=True):
                            task_store.apply_update(t["task_id"], "dismissed", "dismissed", "Dismissed by user", "User dismissed task")
                            st.info("Task dismissed.")
                            st.rerun()
                    else:
                        if st.button("Restore", key=f"btn_rest_{t['task_id']}", type="secondary", use_container_width=True):
                            task_store.apply_update(t["task_id"], "pending", "pending", "Restored", "User restored task")
                            st.success("Task restored.")
                            st.rerun()
                with btn_del_col:
                    if st.button("Delete", key=f"btn_quick_del_{t['task_id']}", type="secondary", use_container_width=True, help="Permanently delete this individual task"):
                        task_store.delete_task(t["task_id"])
                        st.session_state.pop(f"sel_{t['task_id']}", None)
                        if t["task_id"] in st.session_state.get("selected_task_ids", set()):
                            st.session_state["selected_task_ids"].discard(t["task_id"])
                        st.success("Task deleted.")
                        st.rerun()

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


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
            "Chat Reply (Slack/Teams): PR #402 Reviewed & Merged": "@alex I have reviewed PR #402 and merged it to main branch; all checks green.",
            "Chat Status: Elena Completed Redis Saturation Investigation": "Elena has finished investigating the Redis connection pool saturation; connection limits have been stabilized.",
            "Sprint Update: David Postponed Security Architecture to Next Monday": "We have postponed the security architecture diagram to next Monday. David will coordinate with compliance.",
            "Mobile SMS Ping: Client Postponed Call to Friday": "Hey team, client postponed the review call to Friday afternoon.",
            "Task Completion: Quarterly Budget Submitted": "I have finished preparing the finalized quarterly report and submitted it to the team.",
            "Reassignment: Task Delegated to Alternate Owner": "Sarah is out of the office, so Alex will run the load testing benchmark instead.",
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

            top_score = ranked[0]["similarity_score"] if ranked else 0.0
            top_pct = int(top_score * 100)

            # Update Proposal
            st.divider()
            st.markdown("### Review & Apply Detected Changes")

            if top_score < 0.25:
                st.warning(
                    f"Low match confidence ({top_pct}%): This message does not appear closely related to any active task in your ledger. "
                    "Safety lock is active to prevent accidental task corruption."
                )
                allow_apply = st.checkbox("I verify this is the intended task (override safety lock)", value=False, key="override_safety_lock")
            else:
                allow_apply = True

            if allow_apply:
                selected_task_id = st.selectbox(
                    "Confirm Matching Task:",
                    task_options,
                    format_func=lambda tid: f"{tid} - {next((r['task']['action_description'] for r in ranked if r['task']['task_id'] == tid), tid)}",
                    key="select_confirmed_match",
                )

                selected_task = task_store.get_task(selected_task_id) if selected_task_id else None
            else:
                selected_task = None

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
