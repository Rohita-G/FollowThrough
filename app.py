#!/usr/bin/env python3
"""Streamlit app for FollowThrough: Task extraction, SQLite persistence, semantic follow-up linking,
sliding-window long-message inference, and Google Calendar sync."""

import csv
import datetime
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from followthrough_core.task_extractor import extract_candidate_tasks
from followthrough_core.task_store import TaskStore
from followthrough_core.followup_linker import FollowupLinker
from followthrough_core.long_message_classifier import classify_long_message
from followthrough_core.calendar_sync import CalendarSyncManager, NOTIFICATION_DISCLAIMER

# Paths
ORIG_DISTILBERT_DIR = Path("results/action_detector_distilbert/distilbert/checkpoint")
TUNED_DISTILBERT_DIR = Path("results/action_detector_tuned/distilbert/checkpoint")
TUNED_THRESHOLDS_PATH = Path("results/action_detector_tuned/thresholds.json")

BASELINE_PATH = Path("results/action_detector_baseline/tfidf/model.joblib")
TUNED_BASELINE_PATH = Path("results/action_detector_tuned/tfidf/model.joblib")

BASELINE_METRICS = Path("results/action_detector_baseline/tfidf/metrics.json")
DISTILBERT_METRICS = Path("results/action_detector_distilbert/distilbert/metrics.json")
COMPARISON_METRICS = Path("results/action_detector_tuned/evaluation_comparison.json")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MAX_LENGTH = 128


@st.cache_resource
def get_task_store() -> TaskStore:
    return TaskStore()


@st.cache_resource
def get_followup_linker() -> FollowupLinker:
    return FollowupLinker()


@st.cache_resource
def get_calendar_manager() -> CalendarSyncManager:
    return CalendarSyncManager()


@st.cache_resource
def load_distilbert(model_dir: Path) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_baseline(model_path: Path) -> Dict[str, Any]:
    return joblib.load(model_path)


def load_json(p: Path) -> Dict[str, Any]:
    if p.is_file():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# UI Setup
st.set_page_config(page_title="FollowThrough Task & Follow-up Tracker", layout="wide")
st.title("FollowThrough: Task Extraction, Follow-up Linking & Calendar Sync")
st.caption("Turn messages into structured tasks, persist change history in SQLite, and link follow-up updates with semantic ranking.")

task_store = get_task_store()
followup_linker = get_followup_linker()
calendar_manager = get_calendar_manager()

# Sidebar: Research Details & Google Calendar OAuth Controls
with st.sidebar:
    st.subheader("Research & Architecture Details")
    st.markdown("""
**Distinction of Methods**:
- **Trained Classifiers**: DistilBERT / TF-IDF trained on small AI-assisted dataset (N=43). Supporting scores only.
- **Pretrained Semantic Similarity**: `all-MiniLM-L6-v2` compares follow-ups to saved tasks. Similarity is *not* proof of a relationship; requires user confirmation.
- **Rule-based Extraction & Update Detection**: Conservative linguistic patterns for action verbs, explicit owners, literal dates, and status changes. Not trained NER.
    """)

    with st.expander("Classifier Configuration & Metrics", expanded=False):
        model_choice = st.radio(
            "Supporting Classifier:",
            ("DistilBERT (Tuned - MPS/CPU)", "DistilBERT (Smoke-run)", "TF-IDF + Logistic Regression"),
            index=0,
        )
        classifier_mode = st.radio(
            "Classifier Inference Mode:",
            ("Original (128-token truncated)", "Full-message sliding window"),
            index=0,
            help="Full-message mode uses overlapping 128-token windows. Message aggregation is experimental, not calibrated confidence.",
        )
        st.markdown("**Dataset**: 43 messages across 12 threads. Test set N=17.")
        comp = load_json(COMPARISON_METRICS)
        if comp:
            st.markdown("**Test Set Comparison (N=17)**:")
            st.write("Baseline:", comp.get("baseline", {}))
            st.write("DistilBERT (Tuned):", comp.get("distilbert_tuned", {}))
        tuned_th = load_json(TUNED_THRESHOLDS_PATH)
        if tuned_th:
            st.markdown(f"**Tuned Thresholds**: Request = `{tuned_th.get('request_threshold')}`, Commitment = `{tuned_th.get('commitment_threshold')}`")
            st.caption(f"Evidence lost to truncation: {tuned_th.get('truncation_lost_evidence')}")

    # Google Calendar OAuth Section
    st.subheader("Google Calendar Integration")
    is_cal_connected = calendar_manager.is_connected()

    if is_cal_connected:
        st.success("Google Calendar: Connected")
        if st.button("Disconnect Calendar", type="secondary"):
            calendar_manager.disconnect()
            st.info("Calendar disconnected. Local token removed.")
            st.rerun()
    else:
        st.info("Google Calendar: Disconnected")
        st.caption("Local desktop OAuth using minimal scope (`calendar.events`). No passwords collected. Credentials & tokens stored locally outside Git.")

        creds_file_exists = calendar_manager.credentials_path.is_file()
        if not creds_file_exists:
            st.caption("Upload `credentials.json` (or place in `data/credentials/credentials.json`):")
            uploaded_creds = st.file_uploader("Upload Google OAuth Client Secrets JSON", type=["json"])
            if uploaded_creds:
                calendar_manager.credentials_path.parent.mkdir(parents=True, exist_ok=True)
                with calendar_manager.credentials_path.open("wb") as f:
                    f.write(uploaded_creds.getvalue())
                st.success("Saved credentials.json successfully.")
                st.rerun()

        if creds_file_exists:
            if st.button("Connect Google Calendar (Desktop OAuth)", type="primary"):
                with st.spinner("Opening browser for local desktop OAuth authorization..."):
                    ok_conn, msg_conn = calendar_manager.start_desktop_oauth()
                    if ok_conn:
                        st.success(msg_conn)
                        st.rerun()
                    else:
                        st.error(f"Connection error: {msg_conn}")

# Main Tabs
tab1, tab2, tab3 = st.tabs([
    "1. Create tasks from a message",
    "2. Saved tasks",
    "3. Link a follow-up and review changes",
])

# -------------------------------------------------------------
# TAB 1: Create tasks from a message
# -------------------------------------------------------------
with tab1:
    st.subheader("Create Tasks from Message")
    st.markdown(
        "Paste an email or message. Candidate tasks are extracted from the **full message** using **conservative linguistic rules** "
        "(exact character offsets and uncertainty flags preserved). Classifier predictions provide supporting information and never gate extraction."
    )

    default_text = (
        "Please submit the quarterly risk report by Friday afternoon. Shirley will assist with the appendices.\n\n"
        "Thanks,\nManagement"
    )
    message_input = st.text_area("Message Text:", value=default_text, height=140, key="input_create_msg")

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        extract_btn = st.button("Extract Tasks", type="primary")

    if extract_btn or "last_msg_analysis" in st.session_state:
        if extract_btn:
            if not message_input.strip():
                st.error("Please provide non-empty text.")
                st.stop()

            # Task extraction always operates on the full message independently
            candidates = extract_candidate_tasks(message_input)

            # Supporting classifier execution
            req_score, com_score = 0.0, 0.0
            req_pred, com_pred = 0, 0
            req_thresh, com_thresh = 0.5, 0.5
            token_count = 0
            truncated = False
            long_msg_result = None

            if "DistilBERT" in model_choice:
                ckpt_path = TUNED_DISTILBERT_DIR if "Tuned" in model_choice else ORIG_DISTILBERT_DIR
                if "Tuned" in model_choice:
                    th_data = load_json(TUNED_THRESHOLDS_PATH)
                    req_thresh = th_data.get("request_threshold", 0.5)
                    com_thresh = th_data.get("commitment_threshold", 0.5)

                tokenizer, bert_model = load_distilbert(ckpt_path)
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
                    # Full-message sliding window mode
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
                base_bundle = load_baseline(TUNED_BASELINE_PATH if TUNED_BASELINE_PATH.is_file() else BASELINE_PATH)
                vec = base_bundle["vectorizer"]
                clfs = base_bundle["classifiers"]
                feat = vec.transform([message_input])
                if hasattr(clfs[0], "predict"):
                    req_pred = int(clfs[0].predict(feat)[0])
                    req_score = float(clfs[0].predict_proba(feat)[0][1])
                if hasattr(clfs[1], "predict"):
                    com_pred = int(clfs[1].predict(feat)[0])
                    com_score = float(clfs[1].predict_proba(feat)[0][1])

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
                f"Truncation Warning: Input contains {analysis['token_count']} tokens. "
                f"Original mode processed only the first {MAX_LENGTH} tokens. "
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
                st.caption("Raw scores are uncalibrated sigmoid probabilities. These predictions inform the user but do not block task extraction.")

        # Candidate Tasks Display & Manual Edit
        st.markdown("### Candidate Tasks (Conservative Linguistic Extraction)")
        candidates = analysis["candidates"]
        if not candidates:
            st.info("No explicit action candidate was detected with conservative linguistic rules.")
        else:
            st.caption("Review and edit candidate task objects before saving to SQLite. Fields are not invented.")
            for idx, cand in enumerate(candidates):
                with st.container(border=True):
                    st.markdown(f"**Candidate #{idx+1}**")
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
                        type="secondary",
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
                            st.success(f"Task successfully saved! Task ID: `{res_msg}`")
                        else:
                            st.warning(f"Could not save task: {res_msg}")

        # Human label correction export
        with st.expander("Export Classifier Labels (Human-Confirmed)", expanded=False):
            col_corr1, col_corr2 = st.columns(2)
            with col_corr1:
                corr_req = st.selectbox("Confirm Request Label:", ["-- Select --", "0 (No)", "1 (Yes)"], index=0, key="corr_req_select")
            with col_corr2:
                corr_com = st.selectbox("Confirm Commitment Label:", ["-- Select --", "0 (No)", "1 (Yes)"], index=0, key="corr_com_select")

            if corr_req != "-- Select --" and corr_com != "-- Select --":
                output_buf = io.StringIO()
                writer = csv.writer(output_buf)
                writer.writerow(["text", "confirmed_request", "confirmed_commitment", "model_used", "pred_request", "pred_commitment"])
                writer.writerow([
                    analysis["text"],
                    corr_req.split()[0],
                    corr_com.split()[0],
                    analysis.get("model_choice", "DistilBERT"),
                    analysis["req_pred"],
                    analysis["com_pred"]
                ])
                st.download_button(
                    label="Download Confirmed Label CSV",
                    data=output_buf.getvalue().encode("utf-8"),
                    file_name="confirmed_action_annotation.csv",
                    mime="text/csv",
                )


# -------------------------------------------------------------
# TAB 2: Saved tasks & Google Calendar Sync
# -------------------------------------------------------------
with tab2:
    st.subheader("Saved Tasks (SQLite Store)")
    tasks = task_store.list_tasks()

    if not tasks:
        st.info("No confirmed tasks in SQLite database yet. Create and confirm tasks in Tab 1.")
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

                st.markdown(f"**Evidence:** `{t['evidence_text']}` (Offsets: `[{t['start_offset']}:{t['end_offset']}]`)")

                # Calendar Status Badge
                cal_id = t.get("calendar_id")
                evt_id = t.get("calendar_event_id")
                if evt_id:
                    st.success(f"Linked to Google Calendar: Event ID `{evt_id}` (Calendar: `{cal_id}`)")
                else:
                    st.caption("Google Calendar: Not synced")

                # Change History
                history = task_store.get_history(t["task_id"])
                with st.expander(f"Change History ({len(history)} entries)", expanded=False):
                    for h in history:
                        st.markdown(
                            f"- **{h['timestamp'][:19]}** | Update: `{h['update_type']}` | "
                            f"Old: `{h['old_value']}` -> New: `{h['new_value']}` | "
                            f"Evidence: *\"{h['evidence_text']}\"*"
                        )
                        if h.get("followup_text"):
                            st.caption(f"Context: {h['followup_text'][:120]}")

                # Google Calendar Sync Controls
                with st.expander("Google Calendar Sync & Preview", expanded=False):
                    if not calendar_manager.is_connected():
                        st.warning("Google Calendar is not connected. Connect in the sidebar first.")
                    else:
                        st.caption("Preview and confirm all parameters before writing. Date-only deadlines remain all-day unless a time is provided.")
                        st.info(NOTIFICATION_DISCLAIMER)

                        # Parse defaults
                        today_str = datetime.date.today().isoformat()
                        default_title = t["action_description"]

                        cal_col1, cal_col2 = st.columns(2)
                        with cal_col1:
                            cal_title = st.text_input("Event Title:", value=default_title, key=f"cal_title_{t['task_id']}")
                            is_all_day = st.checkbox("All-day Event (Date-only deadline)", value=True, key=f"cal_allday_{t['task_id']}")
                            start_date = st.text_input("Date (YYYY-MM-DD):", value=today_str, key=f"cal_sdate_{t['task_id']}")

                        with cal_col2:
                            tz = st.selectbox("Timezone:", ["UTC", "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles", "Europe/London"], index=2, key=f"cal_tz_{t['task_id']}")
                            rem_mins = st.number_input("Reminder (Minutes before):", min_value=0, max_value=1440, value=30, key=f"cal_rem_{t['task_id']}")

                            start_time, end_time = None, None
                            if not is_all_day:
                                st.caption("Timed event: Explicit start and end time are required. Duration is not invented.")
                                c_time1, c_time2 = st.columns(2)
                                with c_time1:
                                    start_time = st.text_input("Start Time (HH:MM):", value="09:00", key=f"cal_stime_{t['task_id']}")
                                with c_time2:
                                    end_time = st.text_input("End Time (HH:MM):", value="10:00", key=f"cal_etime_{t['task_id']}")

                        # Preview Payload
                        try:
                            preview_payload = CalendarSyncManager.prepare_event_payload(
                                title=cal_title,
                                description=f"Task ID: {t['task_id']}\nOwner: {t['owner']}\nSource Evidence: {t['evidence_text']}",
                                start_date=start_date,
                                is_all_day=is_all_day,
                                start_time=start_time,
                                end_time=end_time,
                                timezone=tz,
                                reminder_minutes=rem_mins,
                            )
                            st.json(preview_payload)

                            btn_label = "Confirm & Update Linked Event" if evt_id else "Confirm & Create Event in Google Calendar"
                            if st.button(btn_label, key=f"btn_sync_cal_{t['task_id']}", type="primary"):
                                ok_sync, synced_id, _ = calendar_manager.sync_event(
                                    calendar_id="primary",
                                    payload=preview_payload,
                                    existing_event_id=evt_id,
                                )
                                if ok_sync:
                                    task_store.update_task_calendar(t["task_id"], "primary", synced_id)
                                    st.success(f"Successfully synced event to Google Calendar (ID: `{synced_id}`).")
                                    st.rerun()
                                else:
                                    st.error(f"Sync error: {synced_id}")
                        except ValueError as val_err:
                            st.warning(f"Validation requirement: {val_err}")

                        # Cancellation / Deletion Section
                        if evt_id:
                            st.markdown("---")
                            st.markdown("#### Cancel / Delete Event from Google Calendar")
                            st.caption("Separate confirmation is required before removing this event from Google Calendar.")
                            confirm_del = st.checkbox(
                                "I confirm deletion of this event from Google Calendar",
                                key=f"chk_del_cal_{t['task_id']}",
                            )
                            if st.button("Delete Event from Google Calendar", key=f"btn_del_cal_{t['task_id']}", type="secondary"):
                                if not confirm_del:
                                    st.error("Please check the confirmation box above to proceed with deletion.")
                                else:
                                    ok_del, msg_del = calendar_manager.delete_event("primary", evt_id)
                                    if ok_del:
                                        task_store.clear_task_calendar(t["task_id"])
                                        st.success(msg_del)
                                        st.rerun()
                                    else:
                                        st.error(msg_del)

                # Manual Status Update
                with st.expander("Update Task Status (Requires User Confirmation & Evidence)", expanded=False):
                    st.caption("Constraint: Do not mark an action completed without supporting evidence and user confirmation.")
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
# TAB 3: Link a follow-up and review changes
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
        default_followup = "We have postponed the quarterly risk report to next Monday. Mike will review."
        followup_input = st.text_area("Paste Follow-up Message:", value=default_followup, height=100, key="followup_input")

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
                        if cand_task.get("calendar_event_id"):
                            st.caption(f"Linked Google Calendar Event: `{cand_task['calendar_event_id']}`")
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
                if selected_task.get("calendar_event_id"):
                    st.info(f"Task is linked to Google Calendar Event ID `{selected_task['calendar_event_id']}`. After applying changes, you can sync the revision in Tab 2 without creating a duplicate event.")

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
