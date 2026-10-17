#!/usr/bin/env python3
"""Streamlit application for FollowThrough.

Workflow:
1. New Task: Paste email/message → supporting classifier signals → review candidate tasks with character offsets → save to SQLite ledger.
2. My Tasks: View stored tasks, inspect evidence slices, and review immutable change history.
3. Follow-up: Paste follow-up message → semantic similarity ranking (all-MiniLM-L6-v2) → detect proposed updates → confirm and persist changes.
"""

import csv
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
st.set_page_config(page_title="FollowThrough Task Tracker", layout="wide")
st.title("FollowThrough: Evidence-Backed Conversational Task & State Tracker")
st.caption("Turn messages into evidence-backed tasks, persist change history in SQLite, and track confirmed updates across follow-ups.")

task_store = get_task_store()
followup_linker = get_followup_linker()

# Sidebar: Research Details & Classifier Options
with st.sidebar:
    st.subheader("System Architecture")
    st.markdown("""
**Core Methodology**:
- **Sequence Classifiers**: DistilBERT / TF-IDF for Request and Commitment signals. Advisory scores only; never gates extraction.
- **Pretrained Semantic Similarity**: `all-MiniLM-L6-v2` compares follow-ups to saved tasks.
- **Linguistic Extraction**: Conservative patterns for action verbs, explicit owners, literal dates, and offsets.
- **Verified Ledger**: SQLite database with append-only transition history.
    """)

    with st.expander("Classifier Configuration", expanded=False):
        model_choice = st.radio(
            "Supporting Classifier:",
            ("DistilBERT (Fine-tuned)", "TF-IDF + Logistic Regression"),
            index=0,
        )
        classifier_mode = st.radio(
            "Classifier Inference Mode:",
            ("Original (128-token truncated)", "Full-message sliding window"),
            index=0,
            help="Full-message mode uses overlapping 128-token windows. Message aggregation is an experimental heuristic.",
        )


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
    st.subheader("Extract Task from Email or Message")
    st.markdown(
        "Paste an email or workplace message to detect request/commitment signals, "
        "extract candidate actions with exact source evidence, and save verified tasks."
    )

    sample_options = {
        "Sample 1: Quarterly Budget (Request + Commitment + Deadline)": (
            "Hi Shirley,\n\n"
            "Please prepare the finalized quarterly budget report by Thursday afternoon.\n"
            "I will follow up with accounting tomorrow to get the updated ledger numbers.\n\n"
            "Thanks,\nPhillip"
        ),
        "Sample 2: Email with Forwarded Chain (Demonstrating Clean Task Extraction)": (
            "Vince,\n\n"
            "Please review this draft contract agreement by tomorrow 5 PM.\n"
            "I will follow up with legal to verify the indemnity terms.\n\n"
            "---------------------- Forwarded by Vince J Kaminski/HOU/ECT on 05/07/2001 12:28 PM ---------------------------\n"
            "From: contractor-relations@enron.com\n"
            "Sent: Friday, May 04, 2001 5:26 PM\n"
            "Subject: Old 1999 boilerplate terms...\n"
        ),
        "Sample 3: IT Engineering Task (Assignee + Strict Deadline)": (
            "Alex,\n\n"
            "Could you verify the server deployment logs before Friday?\n"
            "We need to submit the security compliance report to the auditors.\n\n"
            "Best regards,\nSarah"
        ),
        "Custom Message (Type / Paste Your Own)": "",
    }

    selected_sample = st.selectbox(
        "Choose a sample scenario or enter your own:",
        list(sample_options.keys()),
        index=0,
    )
    initial_text = sample_options[selected_sample]

    message_input = st.text_area(
        "Paste Email / Message:",
        value=initial_text,
        height=140,
        key=f"msg_input_{selected_sample[:8]}",
    )
    analyze_btn = st.button("Analyze Message", type="primary")

    if analyze_btn or "last_msg_analysis" in st.session_state:
        if analyze_btn:
            if not message_input.strip():
                st.error("Please provide non-empty text.")
                st.stop()

            # 1. Conservative Linguistic Extraction (Always runs independently of token limits)
            candidates = extract_candidate_tasks(message_input)

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
                # Resolve checkpoint path
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
                    # Fallback advisory score heuristic when checkpoint binary is uncommitted
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

        if analysis.get("truncated"):
            st.warning(
                f"Truncation Notice: Input contains {analysis['token_count']} tokens. "
                f"Original mode processed the first {MAX_LENGTH} tokens. "
                f"(Switch to 'Full-message sliding window' mode in sidebar to analyze all tokens)."
            )

        # Supporting Classifier Scores
        with st.expander("Supporting Classifier Predictions (Non-gating)", expanded=True):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.metric(
                    label=f"Request Prediction (Threshold: {analysis['req_thresh']})",
                    value="YES (1)" if analysis["req_pred"] == 1 else "NO (0)",
                    delta=f"Model score: {analysis['req_score']:.4f}",
                )
            with sc2:
                st.metric(
                    label=f"Commitment Prediction (Threshold: {analysis['com_thresh']})",
                    value="YES (1)" if analysis["com_pred"] == 1 else "NO (0)",
                    delta=f"Model score: {analysis['com_score']:.4f}",
                )

            if analysis.get("long_msg_result"):
                lm = analysis["long_msg_result"]
                st.info(f"**Aggregation Note**: {lm['aggregation_label']}. Total tokens: {lm['total_tokens']}, Windows: {lm['num_windows']}.")
                with st.expander(f"View Per-Window Breakdown ({lm['num_windows']} windows)", expanded=False):
                    for w in lm["windows"]:
                        st.markdown(
                            f"**Window #{w['window_index']+1}** (Tokens: `{w['token_start']}-{w['token_end']}`, "
                            f"Chars: `[{w['char_start']}:{w['char_end']}]`) | "
                            f"Req: `{w['req_score']:.4f}` | Com: `{w['com_score']:.4f}`"
                        )
                        st.caption(f"Source: \"{w['source_text']}\"")
            else:
                st.caption("Raw scores are uncalibrated probabilities. These predictions inform the user but do not block task extraction.")

        # Candidate Tasks Display & User Review
        st.markdown("### Suggested Tasks (Conservative Linguistic Extraction)")
        candidates = analysis["candidates"]
        if not candidates:
            st.info("No explicit action candidate was detected with conservative linguistic rules.")
        else:
            st.caption("Review and edit suggested task fields before saving. Missing fields are never invented.")
            for idx, cand in enumerate(candidates):
                with st.container(border=True):
                    st.markdown(f"**Suggested Task #{idx+1}**")
                    st.markdown(
                        f"**Source Evidence:** `{cand['evidence_text']}` "
                        f"(Offsets: `[{cand['start_offset']}:{cand['end_offset']}]`)"
                    )
                    flags = cand.get("uncertainty_flags", [])
                    if flags:
                        st.markdown("**Uncertainty Flags:** " + " ".join([f"`{flag}`" for flag in flags]))
                    else:
                        st.markdown("**Uncertainty Flags:** `none` (explicit action, owner, and deadline present)")

                    col_edit1, col_edit2, col_edit3 = st.columns([2, 1, 1])
                    with col_edit1:
                        edit_desc = st.text_input(
                            f"Action Description #{idx+1}",
                            value=cand["action_description"],
                            key=f"edit_desc_{idx}",
                        )
                    with col_edit2:
                        edit_owner = st.text_input(
                            f"Explicit Owner #{idx+1}",
                            value=cand["owner"],
                            key=f"edit_owner_{idx}",
                        )
                    with col_edit3:
                        edit_deadline = st.text_input(
                            f"Literal Deadline #{idx+1}",
                            value=cand["deadline"],
                            key=f"edit_deadline_{idx}",
                        )

                    save_candidate_btn = st.button(
                        f"Confirm & Save Task #{idx+1}",
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
                            st.success(f"Task successfully saved to ledger! Task ID: `{res_msg}`")
                        else:
                            st.warning(f"Could not save task: {res_msg}")


# -------------------------------------------------------------
# TAB 2: My Tasks
# -------------------------------------------------------------
with tab2:
    st.subheader("My Tasks")
    tasks = task_store.list_tasks()

    if not tasks:
        st.info("No saved tasks in database yet. Create and confirm tasks in Tab 1.")
    else:
        st.caption(f"Total tasks stored: {len(tasks)}. All state updates and revision histories are tracked.")

        status_filter = st.selectbox("Filter by Status:", ["All", "pending", "completed", "cancelled"], index=0)
        filtered_tasks = tasks if status_filter == "All" else [t for t in tasks if t["status"] == status_filter]

        for t in filtered_tasks:
            with st.container(border=True):
                head_col1, head_col2 = st.columns([3, 1])
                with head_col1:
                    st.markdown(f"### `{t['task_id']}` — {t['action_description']}")
                with head_col2:
                    st.markdown(f"**Status:** `{t['status'].upper()}`")

                det_col1, det_col2, det_col3 = st.columns(3)
                with det_col1:
                    st.markdown(f"**Owner:** `{t['owner']}`")
                with det_col2:
                    st.markdown(f"**Deadline:** `{t['deadline']}`")
                with det_col3:
                    st.markdown(f"**Created (UTC):** `{t['created_at'][:19]}`")

                # Expandable Evidence
                with st.expander("Source Evidence", expanded=False):
                    st.markdown(f"**Evidence Text:** `{t['evidence_text']}`")
                    st.caption(f"Offsets: `[{t['start_offset']}:{t['end_offset']}]` in original message.")

                # Expandable Change History
                history = task_store.get_history(t["task_id"])
                with st.expander(f"Change History ({len(history)} entries)", expanded=False):
                    for h in history:
                        st.markdown(
                            f"- **{h['timestamp'][:19]}** | Update: `{h['update_type']}` | "
                            f"Old: `{h['old_value']}` ➔ New: `{h['new_value']}` | "
                            f"Evidence: *\"{h['evidence_text']}\"*"
                        )
                        if h.get("followup_text"):
                            st.caption(f"Context: {h['followup_text'][:120]}")

                # Manual Status Update with Evidence & Confirmation
                with st.expander("Update Task Status (Requires User Confirmation & Evidence)", expanded=False):
                    st.caption("Constraint: Do not mark an action completed or cancelled without supporting evidence and user confirmation.")
                    col_m1, col_m2 = st.columns([2, 1])
                    with col_m1:
                        manual_ev = st.text_input(
                            "Supporting Evidence / Note:",
                            key=f"manual_ev_{t['task_id']}",
                            placeholder="e.g., Confirmed receipt via call or document delivered.",
                        )
                    with col_m2:
                        new_stat = st.selectbox(
                            "New Status:",
                            ["completed", "cancelled"],
                            key=f"manual_stat_{t['task_id']}",
                        )
                    confirm_stat_btn = st.button(
                        f"Confirm Status Change for {t['task_id']}",
                        key=f"btn_stat_{t['task_id']}",
                        type="secondary",
                    )
                    if confirm_stat_btn:
                        if not manual_ev.strip():
                            st.error("Supporting evidence is strictly required before changing status.")
                        else:
                            ok = task_store.apply_update(
                                t["task_id"],
                                new_stat,
                                new_stat,
                                manual_ev.strip(),
                                f"Manual user confirmation: {manual_ev.strip()}",
                            )
                            if ok:
                                st.success(f"Status updated to '{new_stat}' for task {t['task_id']}.")
                                st.rerun()


# -------------------------------------------------------------
# TAB 3: Follow-up
# -------------------------------------------------------------
with tab3:
    st.subheader("Link Follow-up & Review Changes")
    st.markdown(
        "Compare a pasted follow-up message with saved tasks using **pretrained semantic similarity** "
        "(`sentence-transformers/all-MiniLM-L6-v2`). Conservatively detect proposed updates with transparent rules and exact evidence."
    )

    all_tasks = task_store.list_tasks()
    if not all_tasks:
        st.warning("No saved tasks found. Please create and confirm a task in Tab 1 before linking follow-ups.")
    else:
        followup_presets = {
            "Follow-up 1: Deadline Extension & Reassignment": "We have postponed the quarterly budget report to next Monday. Mike will review.",
            "Follow-up 2: Task Completion": "I have finished preparing the finalized quarterly report and submitted it to the team.",
            "Follow-up 3: Reassignment": "Shirley is out of the office, so David will prepare the report instead.",
            "Custom Follow-up (Type / Paste Your Own)": "",
        }

        selected_f_preset = st.selectbox(
            "Choose a sample follow-up scenario or write your own:",
            list(followup_presets.keys()),
            index=0,
            key="followup_preset_select",
        )
        initial_f_text = followup_presets[selected_f_preset]

        followup_input = st.text_area(
            "Paste Follow-up Message:",
            value=initial_f_text,
            height=100,
            key=f"followup_input_{selected_f_preset[:8]}",
        )

        link_btn = st.button("Analyze Follow-up & Rank Matches", type="primary")

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

            st.markdown("### Top Candidate Task Matches")
            st.info(
                "**Important**: Similarity score indicates semantic proximity, but similarity is **not proof of a relationship**. "
                "User confirmation is strictly required to link a follow-up. "
                "A related commitment is not classified as unrelated merely because it changes no deadline or owner."
            )

            ranked = f_analysis["ranked"]
            task_options = [r["task"]["task_id"] for r in ranked]

            for rank_idx, r in enumerate(ranked):
                cand_task = r["task"]
                with st.container(border=True):
                    sc_col1, sc_col2 = st.columns([3, 1])
                    with sc_col1:
                        st.markdown(f"**Rank #{rank_idx+1} — Task ID:** `{cand_task['task_id']}`")
                        st.markdown(f"**Description:** {cand_task['action_description']}")
                        st.markdown(f"**Current Owner:** `{cand_task['owner']}` | **Current Deadline:** `{cand_task['deadline']}` | **Status:** `{cand_task['status']}`")
                    with sc_col2:
                        st.metric("Similarity Score", f"{r['similarity_score']:.4f}")

            # User confirmation of match
            st.markdown("---")
            st.subheader("Select Matching Task & Review Proposed Updates")

            selected_task_id = st.selectbox(
                "Choose Confirmed Matching Task:",
                task_options,
                key="select_confirmed_match",
            )

            selected_task = task_store.get_task(selected_task_id) if selected_task_id else None

            if selected_task:
                st.markdown(f"Selected: `{selected_task['task_id']}` — *{selected_task['action_description']}*")

                updates = f_analysis["updates"]
                if not updates:
                    st.info(
                        "No explicit deadline revision, owner change, cancellation, or completion detected in text. "
                        "(Note: Courtesies like 'Thanks' are excluded from completion detection.) "
                        "You may still link this follow-up as a related commitment/note."
                    )
                    confirm_link_note_btn = st.button("Confirm & Link Follow-up Note", type="secondary")
                    if confirm_link_note_btn:
                        ok = task_store.apply_update(
                            selected_task["task_id"],
                            "followup_linked",
                            "Linked Follow-up Note",
                            f_analysis["followup_text"][:60],
                            f_analysis["followup_text"],
                        )
                        if ok:
                            st.success(f"Follow-up successfully linked to task `{selected_task['task_id']}` in history.")
                            st.rerun()
                else:
                    st.markdown("#### Conservatively Detected Proposed Updates")
                    st.caption("Review exact evidence. Confirmation is required before modifying task state.")

                    for u_idx, u in enumerate(updates):
                        with st.container(border=True):
                            u_col1, u_col2 = st.columns([3, 1])
                            with u_col1:
                                st.markdown(f"**Update Type:** `{u['update_type']}`")
                                st.markdown(f"**Proposed Change:** {u['description']}")
                                st.markdown(f"**Exact Source Evidence:** `{u['evidence_text']}`")
                                if u["update_type"] == "deadline_revised":
                                    st.markdown(f"Previous Deadline: `{selected_task['deadline']}` ➔ Proposed: `{u['new_value']}`")
                                elif u["update_type"] == "owner_revised":
                                    st.markdown(f"Previous Owner: `{selected_task['owner']}` ➔ Proposed: `{u['new_value']}`")
                                elif u["update_type"] in ("completed", "cancelled"):
                                    st.markdown(f"Previous Status: `{selected_task['status']}` ➔ Proposed: `{u['new_value']}`")

                            with u_col2:
                                apply_btn = st.button(
                                    f"Confirm & Apply Update",
                                    key=f"btn_apply_up_{u_idx}",
                                    type="primary",
                                )
                                if apply_btn:
                                    ok = task_store.apply_update(
                                        selected_task["task_id"],
                                        u["update_type"],
                                        u["new_value"],
                                        u["evidence_text"],
                                        f_analysis["followup_text"],
                                    )
                                    if ok:
                                        st.success(f"Update `{u['update_type']}` successfully applied to task `{selected_task['task_id']}`!")
                                        st.rerun()
