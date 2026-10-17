#!/usr/bin/env python3
"""Streamlit application for FollowThrough.

Workflow:
1. New Task: Paste email/message → supporting classifier signals → review candidate tasks with character offsets → save to SQLite ledger.
2. My Tasks: View stored tasks, inspect evidence slices, and review immutable change history.
3. Follow-up: Paste follow-up message → semantic similarity ranking (all-MiniLM-L6-v2) → detect proposed updates → confirm and persist changes.
"""

import csv
import datetime
import io
import json
from pathlib import Path
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
# UI Configuration
# -------------------------------------------------------------
st.set_page_config(page_title="FollowThrough", page_icon="✓", layout="wide")

task_store = get_task_store()
followup_linker = get_followup_linker()

# Sidebar: Executive Overview & Collapsible AI Inspector
with st.sidebar:
    st.subheader("FollowThrough")
    st.caption("Intelligent Email Task & Follow-up Tracker")

    all_tasks = task_store.list_tasks()
    pending_count = sum(1 for t in all_tasks if t.get("status") == "pending")
    completed_count = sum(1 for t in all_tasks if t.get("status") == "completed")

    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.metric("Pending Tasks", pending_count)
    with stat_col2:
        st.metric("Completed", completed_count)

    st.divider()

    # Optional Technical Diagnostics drawer
    with st.expander("AI Engine & Diagnostics", expanded=False):
        st.caption("Underlying models and configuration parameters:")
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
        **Pipeline Architecture**:
        - Multi-label sequence classification (Request/Commitment)
        - Rule-based action, owner, and deadline extraction
        - Temporal grounding to ISO calendar dates
        - Pretrained semantic similarity (`all-MiniLM-L6-v2`)
        - SQLite verified audit ledger
        """)

# Header
st.title("FollowThrough")
st.caption("Transform workplace messages into verified tasks, assign owners, ground deadlines, and track follow-ups.")

# -------------------------------------------------------------
# Main Tabs: 1. New Task | 2. My Tasks | 3. Follow-up
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "1. New Task",
    "2. My Tasks",
    "3. Follow-up",
])

# -------------------------------------------------------------
# TAB 1: New Task
# -------------------------------------------------------------
with tab1:
    st.subheader("Extract Action Items")
    st.caption("Paste an email or team update to extract action items, assignees, and calendar deadlines.")

    sample_options = {
        "Realistic Scenario 1: Cross-Functional Team Email (Multiple Owners & Deadlines)": (
            "Hi Team,\n\n"
            "Great job on the client presentation yesterday. The feedback from Acme Corp was overwhelmingly positive, especially regarding our new analytics dashboard. They want to move ahead with the enterprise pilot, but there are several blocking items we must resolve first.\n\n"
            "David, please update the security architecture diagram to reflect the SOC2 compliance controls by Wednesday.\n"
            "Sarah, can you please coordinate with the QA team to run the load testing benchmark before Friday?\n"
            "Also, I will schedule a briefing with the legal and contract department tomorrow to review the Master Services Agreement.\n\n"
            "General status note: the staging environment will be under maintenance on Saturday.\n\n"
            "Thanks,\nMarcus"
        ),
        "Realistic Scenario 2: DevOps & Cloud Incident Response (On-Call Action Items)": (
            "Team,\n\n"
            "Following up on the production outage earlier this morning. The database failover completed, but customer traffic is still seeing elevated latency on the auth service.\n\n"
            "Elena, please investigate the connection pool saturation on the primary Redis cluster before 2 PM.\n"
            "Mark, can you deploy the hotfix patch to staging and verify the latency metrics by end of day?\n"
            "I will draft the preliminary post-mortem summary for executive leadership by tomorrow morning.\n\n"
            "Note: no further schema migrations should be executed until we stabilize the cluster.\n\n"
            "Regards,\nAlex"
        ),
        "Realistic Scenario 3: Enterprise Contract Negotiation (Quoted Forward Thread)": (
            "Vince,\n\n"
            "Please review the revised indemnification and confidentiality clauses by tomorrow 5 PM.\n"
            "I will follow up with accounting tomorrow to confirm the wire transfer schedule.\n\n"
            "---------------------- Forwarded by Vince J Kaminski/HOU/ECT on 05/07/2001 12:28 PM ---------------------------\n"
            "From: external-counsel@lawfirm.com\n"
            "Sent: Friday, May 04, 2001 5:26 PM\n"
            "Subject: Boilerplate Warranty Notice\n\n"
            "Please make sure to sign the original 1999 agreement.\n"
        ),
        "Short Example: Standup Directives (Request + Commitment)": (
            "Hi Shirley,\n\n"
            "Please prepare the finalized quarterly budget report by Thursday.\n"
            "I will follow up with accounting tomorrow to get the updated ledger numbers.\n\n"
            "Thanks,\nPhillip"
        ),
        "Custom Message (Type / Paste Your Own)": "",
    }

    col_sample, col_date = st.columns([2, 1])
    with col_sample:
        selected_sample = st.selectbox(
            "Select Scenario or Enter Custom:",
            list(sample_options.keys()),
            index=0,
        )
    with col_date:
        reference_date_val = st.date_input(
            "Email Sent / Anchor Date:",
            value=datetime.date.today(),
            help="Reference date used to resolve relative deadlines ('tomorrow', 'by Wednesday', 'next Friday') into exact calendar dates.",
            key="msg_ref_date_input",
        )

    initial_text = sample_options[selected_sample]

    message_input = st.text_area(
        "Message Text:",
        value=initial_text,
        height=200,
        key=f"msg_input_{selected_sample[:8]}",
        placeholder="Paste workplace message or email thread...",
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

        # Action Items Display (Immediate, Clean, Professional)
        st.markdown("### Suggested Action Items")
        candidates = analysis["candidates"]
        if not candidates:
            st.info("No actionable directives or commitments were found in this message.")
        else:
            st.caption("Review extracted task parameters below and confirm to save to your board.")
            for idx, cand in enumerate(candidates):
                with st.container(border=True):
                    task_type = cand.get("task_type", "Action Directive")
                    type_label = "Delegated Request" if task_type == "Request" else "Author Commitment"
                    st.markdown(f"#### Task #{idx+1}: {cand['action_description']}")

                    # Temporal Grounding display
                    dl_norm = cand.get("deadline_normalized", cand["deadline"])
                    dl_raw = cand.get("deadline_literal", cand["deadline"])
                    if cand.get("is_relative_deadline") and cand.get("deadline_iso"):
                        dl_caption = f"**Due Date:** `{dl_norm}` *(Grounded from: \"{dl_raw}\")*"
                    else:
                        dl_caption = f"**Due Date:** `{dl_norm}`"

                    st.caption(f"**Type:** `{type_label}` | **Assignee:** `{cand['owner']}` | {dl_caption}")
                    st.markdown(
                        f"**Source Evidence:** *\"{cand['evidence_text']}\"* "
                        f"(Offsets: `[{cand['start_offset']}:{cand['end_offset']}]`)"
                    )

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
                        f"Save Task #{idx+1} to Board",
                        key=f"btn_save_cand_{idx}",
                        type="primary",
                    )
                    if save_candidate_btn:
                        task_record = {
                            "source_text": analysis["text"],
                            "action_description": edit_desc.strip(),
                            "owner": edit_owner.strip(),
                            "deadline": edit_deadline.strip(),
                            "evidence_text": cand["evidence_text"],
                            "start_offset": cand["start_offset"],
                            "end_offset": cand["end_offset"],
                        }
                        ok, res_msg = task_store.save_task(task_record)
                        if ok:
                            st.success(f"Task saved to board! Reference: `{res_msg}`")
                        else:
                            st.warning(f"Could not save task: {res_msg}")

        # Optional Collapsible Diagnostics (For Instructors / Model Auditing)
        with st.expander("AI Confidence & Model Diagnostics", expanded=False):
            st.caption("Inspect underlying classifier predictions and tokenization details:")
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

            if analysis.get("long_msg_result"):
                lm = analysis["long_msg_result"]
                st.caption(f"Sliding window aggregation: {lm['aggregation_label']} across {lm['num_windows']} windows (Total Tokens: {lm['total_tokens']}).")


# -------------------------------------------------------------
# TAB 2: My Tasks
# -------------------------------------------------------------
with tab2:
    st.subheader("My Tasks")
    st.caption("Verified tasks extracted from conversations with full history and evidence.")
    tasks = task_store.list_tasks()

    if not tasks:
        st.info("No saved tasks yet. Extract and save tasks in Tab 1.")
    else:
        status_filter = st.selectbox("Filter Status:", ["All", "pending", "completed", "cancelled"], index=0)
        filtered_tasks = tasks if status_filter == "All" else [t for t in tasks if t["status"] == status_filter]

        for t in filtered_tasks:
            with st.container(border=True):
                head_col1, head_col2 = st.columns([3, 1])
                with head_col1:
                    st.markdown(f"#### {t['action_description']}")
                    st.caption(f"Task ID: `{t['task_id']}`")
                with head_col2:
                    status_badge = t["status"].upper()
                    st.markdown(f"**Status:** `{status_badge}`")

                det_col1, det_col2, det_col3 = st.columns(3)
                with det_col1:
                    st.markdown(f"**Assignee:** `{t['owner']}`")
                with det_col2:
                    st.markdown(f"**Due Date:** `{t['deadline']}`")
                with det_col3:
                    st.markdown(f"**Created:** `{t['created_at'][:10]}`")

                # Expandable Details
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    with st.expander("Source Evidence Quote", expanded=False):
                        st.markdown(f"*\"{t['evidence_text']}\"*")
                        st.caption(f"Character offsets: `[{t['start_offset']}:{t['end_offset']}]`")
                with col_exp2:
                    history = task_store.get_history(t["task_id"])
                    with st.expander(f"Activity History ({len(history)})", expanded=False):
                        for h in history:
                            st.markdown(
                                f"- **{h['timestamp'][:16]}** | `{h['update_type']}` | "
                                f"`{h['old_value']}` ➔ `{h['new_value']}`"
                            )

                # Quick Status Update
                with st.expander("Update Task Status", expanded=False):
                    col_m1, col_m2, col_m3 = st.columns([2, 1, 1])
                    with col_m1:
                        manual_ev = st.text_input(
                            "Status Note / Reason:",
                            key=f"manual_ev_{t['task_id']}",
                            placeholder="e.g., Deliverable completed and verified.",
                        )
                    with col_m2:
                        new_stat = st.selectbox(
                            "Set Status:",
                            ["completed", "cancelled"],
                            key=f"manual_stat_{t['task_id']}",
                        )
                    with col_m3:
                        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                        confirm_stat_btn = st.button(
                            "Update Status",
                            key=f"btn_stat_{t['task_id']}",
                            type="secondary",
                        )
                    if confirm_stat_btn:
                        note = manual_ev.strip() or f"User marked as {new_stat}"
                        ok = task_store.apply_update(
                            t["task_id"],
                            new_stat,
                            new_stat,
                            note,
                            f"Status update: {note}",
                        )
                        if ok:
                            st.success(f"Status updated to '{new_stat}'!")
                            st.rerun()


# -------------------------------------------------------------
# TAB 3: Follow-up
# -------------------------------------------------------------
with tab3:
    st.subheader("Follow-up & State Tracker")
    st.caption("Link replies and follow-ups to existing tasks to automatically update deadlines and track status.")

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
            "Select Follow-up Scenario or Write Custom:",
            list(followup_presets.keys()),
            index=0,
            key="followup_preset_select",
        )
        initial_f_text = followup_presets[selected_f_preset]

        followup_input = st.text_area(
            "Follow-up Message Text:",
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

            st.markdown("### Matching Tasks")
            ranked = f_analysis["ranked"]
            task_options = [r["task"]["task_id"] for r in ranked]

            for rank_idx, r in enumerate(ranked):
                cand_task = r["task"]
                match_pct = int(min(max(r["similarity_score"], 0.0), 1.0) * 100)
                with st.container(border=True):
                    sc_col1, sc_col2 = st.columns([3, 1])
                    with sc_col1:
                        st.markdown(f"**{cand_task['action_description']}**")
                        st.caption(f"Assignee: `{cand_task['owner']}` | Due Date: `{cand_task['deadline']}` | Status: `{cand_task['status'].upper()}`")
                    with sc_col2:
                        st.metric("Match Confidence", f"{match_pct}%")

            # Update Proposal
            st.divider()
            st.markdown("### Review & Apply Updates")

            selected_task_id = st.selectbox(
                "Confirm Matching Task:",
                task_options,
                format_func=lambda tid: f"{tid} — {next((r['task']['action_description'] for r in ranked if r['task']['task_id'] == tid), tid)}",
                key="select_confirmed_match",
            )

            selected_task = task_store.get_task(selected_task_id) if selected_task_id else None

            if selected_task:
                updates = f_analysis["updates"]
                if not updates:
                    st.info("No explicit deadline revisions or status changes detected in this text. You can still link it as a progress note.")
                    confirm_link_note_btn = st.button("Link as Progress Note", type="secondary")
                    if confirm_link_note_btn:
                        ok = task_store.apply_update(
                            selected_task["task_id"],
                            "followup_linked",
                            "Linked Progress Note",
                            f_analysis["followup_text"][:60],
                            f_analysis["followup_text"],
                        )
                        if ok:
                            st.success(f"Progress note linked to task `{selected_task['task_id']}`!")
                            st.rerun()
                else:
                    st.markdown("#### Detected Changes")
                    for u_idx, u in enumerate(updates):
                        with st.container(border=True):
                            u_col1, u_col2 = st.columns([3, 1])
                            with u_col1:
                                if u["update_type"] == "deadline_revised":
                                    proposed_target = u.get("new_value_normalized", u["new_value"])
                                    st.markdown(f"**Proposed Due Date:** `{proposed_target}` *(from \"{u['new_value']}\")*")
                                    st.caption(f"Current Due Date: `{selected_task['deadline']}`")
                                elif u["update_type"] == "owner_revised":
                                    st.markdown(f"**Proposed Reassignment:** `{u['new_value']}`")
                                    st.caption(f"Current Assignee: `{selected_task['owner']}`")
                                elif u["update_type"] in ("completed", "cancelled"):
                                    st.markdown(f"**Proposed Status:** `{u['new_value'].upper()}`")

                                st.markdown(f"**Source Evidence:** *\"{u['evidence_text']}\"*")

                            with u_col2:
                                apply_btn = st.button(
                                    "Apply Update",
                                    key=f"btn_apply_up_{u_idx}",
                                    type="primary",
                                )
                                if apply_btn:
                                    commit_val = u.get("new_value_normalized", u["new_value"]) if u["update_type"] == "deadline_revised" else u["new_value"]
                                    ok = task_store.apply_update(
                                        selected_task["task_id"],
                                        u["update_type"],
                                        commit_val,
                                        u["evidence_text"],
                                        f_analysis["followup_text"],
                                    )
                                    if ok:
                                        st.success(f"Update applied to task `{selected_task['task_id']}`!")
                                        st.rerun()
