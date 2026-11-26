#!/usr/bin/env python3
"""Generates the official FollowThrough 15-minute capstone PowerPoint presentation.

Creates a 16:9 widescreen deck with modern typography, custom cards, tables,
embedded architecture diagram, and presenter speaker notes on every slide.
"""

import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PPTX = REPO_ROOT / "FollowThrough_Presentation.pptx"
ARCH_IMG_PATH = REPO_ROOT / "system_architecture.png"

# Colors
COLOR_BG = RGBColor(248, 250, 252)        # #F8FAFC
COLOR_WHITE = RGBColor(255, 255, 255)     # #FFFFFF
COLOR_NAVY = RGBColor(15, 23, 42)         # #0F172A
COLOR_INDIGO = RGBColor(79, 70, 229)      # #4F46E5
COLOR_EMERALD = RGBColor(5, 150, 105)     # #059669
COLOR_CORAL = RGBColor(239, 68, 68)       # #EF4444
COLOR_SLATE = RGBColor(100, 116, 139)     # #64748B
COLOR_CARD_BORDER = RGBColor(226, 232, 240) # #E2E8F0
COLOR_LIGHT_INDIGO = RGBColor(238, 242, 255) # #EEF2FF
COLOR_LIGHT_EMERALD = RGBColor(236, 253, 245) # #ECFDF5

FONT_FAMILY = "Helvetica"


def create_deck():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_BG
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category_text="FOLLOWTHROUGH • CAPSTONE PRESENTATION"):
        # Category label
        tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        p_cat = tb_cat.text_frame.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.name = FONT_FAMILY
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_INDIGO

        # Main Title
        tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        p_title = tb_title.text_frame.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = FONT_FAMILY
        p_title.font.size = Pt(28)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_NAVY

    def set_speaker_notes(slide, notes_text):
        notes_slide = slide.notes_slide
        text_frame = notes_slide.notes_text_frame
        text_frame.text = notes_text.strip()

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Decorative header bar
    top_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), Inches(1.5), Inches(0.08))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = COLOR_INDIGO
    top_bar.line.fill.background()

    # Title box
    tb = s1.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(2.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "FollowThrough"
    p1.font.name = FONT_FAMILY
    p1.font.size = Pt(48)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_NAVY

    p2 = tf.add_paragraph()
    p2.text = "Evidence-Backed Conversational Task Extraction & State Tracking"
    p2.font.name = FONT_FAMILY
    p2.font.size = Pt(24)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_INDIGO
    p2.space_before = Pt(10)

    p3 = tf.add_paragraph()
    p3.text = "Rohita Gangishetty  |  NLP Capstone Project  |  Fall 2026"
    p3.font.name = FONT_FAMILY
    p3.font.size = Pt(16)
    p3.font.color.rgb = COLOR_SLATE
    p3.space_before = Pt(14)

    # 3 Pill Badges
    badges = [
        ("Deterministic Offset Grounding", COLOR_LIGHT_INDIGO, COLOR_INDIGO),
        ("Zero Unverified Writes", COLOR_LIGHT_EMERALD, COLOR_EMERALD),
        ("Cryptographic Audit Ledger", COLOR_WHITE, COLOR_NAVY),
    ]
    badge_w = Inches(3.6)
    for idx, (b_text, b_bg, b_fg) in enumerate(badges):
        b_x = Inches(0.8) + idx * (badge_w + Inches(0.4))
        card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, b_x, Inches(4.7), badge_w, Inches(1.2))
        card.fill.solid()
        card.fill.fore_color.rgb = b_bg
        card.line.color.rgb = COLOR_CARD_BORDER
        card.line.width = Pt(1.5)
        
        c_tf = card.text_frame
        c_tf.word_wrap = True
        c_p = c_tf.paragraphs[0]
        c_p.text = b_text
        c_p.alignment = PP_ALIGN.CENTER
        c_p.font.name = FONT_FAMILY
        c_p.font.size = Pt(15)
        c_p.font.bold = True
        c_p.font.color.rgb = b_fg

    set_speaker_notes(s1, """
Good morning faculty evaluators, mentors, and peers. Think about your email inbox or workplace Slack right now. How many commitments have you made that slipped past deadlines simply because they were buried in a five-email chain?

In enterprise organizations, critical work obligations don't originate in structured project software like Jira; they happen organically in fluid, messy conversation. Today, I am presenting FollowThrough—an evidence-backed natural language processing system engineered to detect, ground, and track conversational commitments with exact character proof, without generative hallucination, and without premature automation.
""")

    # =========================================================================
    # SLIDE 2: The Real-World Problem
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Where Do Tasks Get Lost in Workplace Communication?")

    # Left Card: The Conversation
    card_l = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    card_l.fill.solid()
    card_l.fill.fore_color.rgb = COLOR_WHITE
    card_l.line.color.rgb = COLOR_CARD_BORDER
    card_l.line.width = Pt(1.5)
    
    tf_l = card_l.text_frame
    tf_l.word_wrap = True
    p = tf_l.paragraphs[0]
    p.text = "REAL WORKPLACE CONVERSATION"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_INDIGO

    convs = [
        "\"I'll finalize the audit report by Friday 5 PM.\"",
        "\"Actually, let's push this to Monday morning.\"",
        "\"David is now handling the executive client review.\"",
        "\"Thanks! Did we verify the staging environment?\""
    ]
    for c in convs:
        cp = tf_l.add_paragraph()
        cp.text = f"•  {c}"
        cp.font.size = Pt(15)
        cp.font.italic = True
        cp.font.color.rgb = COLOR_NAVY
        cp.space_before = Pt(16)

    # Right Card: The Operational Breakdown
    card_r = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    card_r.fill.solid()
    card_r.fill.fore_color.rgb = COLOR_WHITE
    card_r.line.color.rgb = COLOR_CARD_BORDER
    card_r.line.width = Pt(1.5)

    tf_r = card_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "OPERATIONAL BREAKDOWN"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_CORAL

    fails = [
        ("Dispersed Context", "Critical action items are buried in conversational noise across multi-turn threads."),
        ("Continuous Drift", "Deadlines, owners, and scopes shift organically without central tracking."),
        ("Zero State Visibility", "Unstructured chat leaves no auditable trail of who is accountable."),
        ("Manual Friction", "Workers forget commitments before transcribing them into ticketing systems.")
    ]
    for f_title, f_desc in fails:
        fp1 = tf_r.add_paragraph()
        fp1.text = f"❌  {f_title}"
        fp1.font.size = Pt(16)
        fp1.font.bold = True
        fp1.font.color.rgb = COLOR_NAVY
        fp1.space_before = Pt(10)
        
        fp2 = tf_r.add_paragraph()
        fp2.text = f_desc
        fp2.font.size = Pt(13)
        fp2.font.color.rgb = COLOR_SLATE
        fp2.space_before = Pt(2)

    set_speaker_notes(s2, """
Workplace conversation is fundamentally non-linear. In a single multi-turn thread, team members exchange requests, make implicit promises, renegotiate deadlines, and pass ownership back and forth. Because this communication is unstructured, human memory inevitably fails.

Deliverables get buried under follow-up chitchat. Deadlines drift across messages without being recorded, and managers have zero visibility into who is truly accountable for what. Traditional productivity software expects humans to manually transcribe these tasks into tickets, but in high-velocity teams, that manual transcription almost never happens.
""")

    # =========================================================================
    # SLIDE 3: The Failure Modes of Existing NLP
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Why Existing NLP Approaches Fail on Enterprise Tasks")

    card_data = [
        ("Keyword Matching & Regex", "TODO:, Action Item:, Please find attached", [
            "Fails on indirect requests & natural promises.",
            "Captures explicit syntax only (Low Recall).",
            "Misses >90% of real conversational commitments."
        ], COLOR_SLATE),
        ("Isolated Token Classifiers / NER", "Standard SpaCy / BERT token tagging", [
            "Evaluates isolated sentences out of context.",
            "Completely blind to multi-turn thread evolution.",
            "Zero state tracking across message replies."
        ], COLOR_SLATE),
        ("Generative LLMs (GPT-4 / Claude)", "Direct prompt-based task extraction", [
            "Hallucinates phantom deadlines & false owners.",
            "Destroys verbatim character evidence provenance.",
            "Severe enterprise privacy & compliance risk."
        ], COLOR_CORAL),
    ]

    for idx, (title, sub, bullets, border_color) in enumerate(card_data):
        cx = Inches(0.8) + idx * Inches(3.95)
        c = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, Inches(1.8), Inches(3.7), Inches(4.8))
        c.fill.solid()
        c.fill.fore_color.rgb = COLOR_WHITE
        c.line.color.rgb = border_color
        c.line.width = Pt(2.0 if border_color == COLOR_CORAL else 1.5)

        ctf = c.text_frame
        ctf.word_wrap = True
        p = ctf.paragraphs[0]
        p.text = title
        p.font.size = Pt(17)
        p.font.bold = True
        p.font.color.rgb = COLOR_NAVY

        p_sub = ctf.add_paragraph()
        p_sub.text = sub
        p_sub.font.size = Pt(12)
        p_sub.font.italic = True
        p_sub.font.color.rgb = COLOR_SLATE
        p_sub.space_before = Pt(4)

        for b in bullets:
            bp = ctf.add_paragraph()
            bp.text = f"•  {b}"
            bp.font.size = Pt(13)
            bp.font.color.rgb = COLOR_NAVY
            bp.space_before = Pt(14)

    set_speaker_notes(s3, """
Why can't off-the-shelf NLP solve this? Keyword regexes look for explicit tags like 'TODO' or 'Action Item,' but humans don't speak like that—they say 'I’ll take care of this before lunch,' which regex completely misses. Standard Named Entity Recognition can identify a date or person, but has no comprehension of who is obligated to perform the task.

And when organizations deploy generative LLMs like GPT-4, they run into two catastrophic risks: hallucinations—where models invent phantom deadlines—and the loss of verbatim character provenance. In enterprise governance, you cannot legally assign an employee a deliverable based on generative text without proof of where it was said.
""")

    # =========================================================================
    # SLIDE 4: Core Philosophy
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Our Core Philosophy: Bounded, Verifiable NLP")

    pillars = [
        ("1. Deterministic Grounding", "Exact Character Offsets [start:end]", [
            "Every extracted task is strictly anchored to verbatim character offsets [c_start, c_end].",
            "Zero generative fiction or invented text.",
            "Clicking any task on the board highlights the exact source sentence."
        ], COLOR_INDIGO),
        ("2. Advisory Intelligence", "Non-Gating Confidence Signals", [
            "Neural models provide probability scores to prioritize human attention.",
            "Classifiers never unilaterally suppress candidate tasks.",
            "High recall extraction paired with smart focus."
        ], COLOR_INDIGO),
        ("3. The Human Gate", "Zero Unverified Writes", [
            "The AI proposes; the human authorizes.",
            "No state change enters the ledger without explicit user confirmation.",
            "Eliminates silent data corruption in enterprise ledgers."
        ], COLOR_EMERALD),
    ]

    for idx, (title, sub, bullets, accent) in enumerate(pillars):
        cx = Inches(0.8) + idx * Inches(3.95)
        c = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, Inches(1.8), Inches(3.7), Inches(4.8))
        c.fill.solid()
        c.fill.fore_color.rgb = COLOR_WHITE
        c.line.color.rgb = COLOR_CARD_BORDER
        c.line.width = Pt(1.5)

        # Top accent bar on card
        bar = s4.shapes.add_shape(MSO_SHAPE.RECTANGLE, cx + Inches(0.3), Inches(2.1), Inches(3.1), Inches(0.06))
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent
        bar.line.fill.background()

        ctf = c.text_frame
        ctf.word_wrap = True
        p = ctf.paragraphs[0]
        p.text = title
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = COLOR_NAVY
        p.space_before = Pt(20)

        p_sub = ctf.add_paragraph()
        p_sub.text = sub
        p_sub.font.size = Pt(12)
        p_sub.font.bold = True
        p_sub.font.color.rgb = accent
        p_sub.space_before = Pt(4)

        for b in bullets:
            bp = ctf.add_paragraph()
            bp.text = f"•  {b}"
            bp.font.size = Pt(13)
            bp.font.color.rgb = COLOR_NAVY
            bp.space_before = Pt(14)

    set_speaker_notes(s4, """
To solve this safely, FollowThrough adopts three foundational principles. First: Deterministic Evidence Grounding. Every extracted task is anchored to exact character offsets in the raw email text. If you see a task on your board, you can click it and see the exact 12-character sentence span where it came from.

Second: Non-Gating Advisory Models. Deep learning classifiers generate speech act probabilities to help focus user attention, but they never unilaterally suppress candidate tasks. And third: our non-negotiable rule: Zero Unverified Writes. The NLP pipeline acts strictly as an intelligent advisor; the human operator retains final authorization. Nothing enters the production database without human sign-off.
""")

    # =========================================================================
    # SLIDE 5: Architecture
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "End-to-End System Architecture & Workflow")

    # Embed Diagram on left
    if ARCH_IMG_PATH.is_file():
        s5.shapes.add_picture(str(ARCH_IMG_PATH), Inches(0.8), Inches(1.8), width=Inches(5.6))
    
    # Right-side feature breakdown card
    c_arch = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.7), Inches(1.8), Inches(5.8), Inches(5.0))
    c_arch.fill.solid()
    c_arch.fill.fore_color.rgb = COLOR_WHITE
    c_arch.line.color.rgb = COLOR_CARD_BORDER
    c_arch.line.width = Pt(1.5)

    atf = c_arch.text_frame
    atf.word_wrap = True
    ap = atf.paragraphs[0]
    ap.text = "FOUR-STAGE VERIFIABLE PIPELINE"
    ap.font.size = Pt(13)
    ap.font.bold = True
    ap.font.color.rgb = COLOR_INDIGO

    stages = [
        ("1. Dual-Track Ingestion", "Linguistic deontic extraction (modal verbs) runs in parallel with sliding-window DistilBERT sequence classification."),
        ("2. Dense Bi-Encoder Linker", "all-MiniLM-L6-v2 projects incoming replies into 384-d semantic space, calculating cosine similarity over active tasks."),
        ("3. Human Review Gate", "Operator verifies character spans, edits assignees or dates, and confirms tasks with Zero Unverified Writes."),
        ("4. Cryptographic Ledger", "SQLite Write-Ahead Logging backend with SHA-256 payload deduplication and append-only audit history.")
    ]
    for s_title, s_desc in stages:
        sp1 = atf.add_paragraph()
        sp1.text = s_title
        sp1.font.size = Pt(15)
        sp1.font.bold = True
        sp1.font.color.rgb = COLOR_NAVY
        sp1.space_before = Pt(10)

        sp2 = atf.add_paragraph()
        sp2.text = s_desc
        sp2.font.size = Pt(12)
        sp2.font.color.rgb = COLOR_SLATE
        sp2.space_before = Pt(2)

    set_speaker_notes(s5, """
Here is our four-stage architecture. When a message is ingested, it splits into two parallel tracks: our deterministic linguistic extractor detects deontic modal verbs, normalizes relative temporal expressions like 'by Friday 5 PM' into calendar ISO dates, and records exact character offsets. Concurrently, our sliding-window DistilBERT classifier evaluates speech act probabilities for requests and commitments.

When a follow-up message arrives, our sentence-transformers bi-encoder transforms it into a 384-dimensional dense vector and calculates cosine similarity against all active tasks. If a match is detected, the proposed status update is staged at the Human Review Gate. Once confirmed by the user, the record is cryptographically hashed with SHA-256 to prevent duplicates and committed to our append-only SQLite ledger.
""")

    # =========================================================================
    # SLIDE 6: Dataset Rigor
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Dataset Rigor: The Carnegie Mellon Enron Corpus")

    stats = [
        ("97,988", "Cleaned Email Records", "Filtered from 100k raw MIME emails. 1.44M sentences, 29.5M tokens, stripped of forward headers and boilerplates."),
        ("Disjoint-Set", "Graph Partitioning", "Union-Find clustering isolates entire email threads and duplicate messages across splits (Zero Evaluation Leakage)."),
        ("4.08%", "Severe Class Imbalance", "Commitments represent only ~4% of data (1 in 25 sentences). Accurately mirrors real-world conversational scarcity."),
    ]
    for idx, (num, label, desc) in enumerate(stats):
        cx = Inches(0.8) + idx * Inches(3.95)
        c = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, Inches(1.8), Inches(3.7), Inches(3.0))
        c.fill.solid()
        c.fill.fore_color.rgb = COLOR_WHITE
        c.line.color.rgb = COLOR_CARD_BORDER
        c.line.width = Pt(1.5)

        ctf = c.text_frame
        ctf.word_wrap = True
        p = ctf.paragraphs[0]
        p.text = num
        p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(36)
        p.font.bold = True
        p.font.color.rgb = COLOR_INDIGO

        p_lbl = ctf.add_paragraph()
        p_lbl.text = label
        p_lbl.alignment = PP_ALIGN.CENTER
        p_lbl.font.size = Pt(14)
        p_lbl.font.bold = True
        p_lbl.font.color.rgb = COLOR_NAVY
        p_lbl.space_before = Pt(4)

        p_desc = ctf.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = Pt(12)
        p_desc.font.color.rgb = COLOR_SLATE
        p_desc.space_before = Pt(10)

    # Bottom Split Breakdown Bar
    c_split = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.7))
    c_split.fill.solid()
    c_split.fill.fore_color.rgb = COLOR_WHITE
    c_split.line.color.rgb = COLOR_CARD_BORDER
    c_split.line.width = Pt(1.5)

    stf = c_split.text_frame
    stf.word_wrap = True
    sp = stf.paragraphs[0]
    sp.text = "DISJOINT GRAPH SPLIT ALLOCATION (ZERO THREAD LEAKAGE)"
    sp.font.size = Pt(12)
    sp.font.bold = True
    sp.font.color.rgb = COLOR_NAVY

    sp2 = stf.add_paragraph()
    sp2.text = "• Training Set: 78,385 records (80.0%)   • Validation Set: 9,946 records (10.2%)   • Held-out Test Set: 9,657 records (9.8%)"
    sp2.font.size = Pt(14)
    sp2.font.bold = True
    sp2.font.color.rgb = COLOR_INDIGO
    sp2.space_before = Pt(8)

    sp3 = stf.add_paragraph()
    sp3.text = "Guarantees that no subject thread, forwarded reply chain, or exact duplicate exists in both train and test splits."
    sp3.font.size = Pt(12)
    sp3.font.color.rgb = COLOR_SLATE
    sp3.space_before = Pt(4)

    set_speaker_notes(s6, """
Rather than relying on toy synthetic benchmarks, we evaluated FollowThrough on the Carnegie Mellon Enron corpus. We built a data pipeline that preprocessed 100,000 raw RFC 822 MIME emails, stripping forward headers, quoted replies, and boilerplate corporate disclaimers, yielding 98,000 clean records and nearly 30 million tokens.

A major flaw in published NLP literature is random row splitting, which leaks forwarded threads between train and test splits. To guarantee zero evaluation leakage, we developed a Union-Find graph partitioning algorithm that grouped all related messages into disjoint connected components before splitting.

Notice the extreme class imbalance in real-world data: while requests appear in 44% of messages, explicit commitments appear in only 4%—roughly one in every twenty-five sentences.
""")

    # =========================================================================
    # SLIDE 7: Model Benchmark & Collapse
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Model Evaluation on Held-Out Test Set (N = 9,657)")

    # Table Shape
    rows, cols = 5, 4
    t_shape = s7.shapes.add_table(rows, cols, Inches(0.8), Inches(1.8), Inches(11.7), Inches(3.2))
    table = t_shape.table
    table.columns[0].width = Inches(3.0)
    table.columns[1].width = Inches(2.5)
    table.columns[2].width = Inches(2.7)
    table.columns[3].width = Inches(3.5)

    headers = ["Metric", "TF-IDF Baseline", "Default DistilBERT", "Scientific Takeaway"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_LIGHT_INDIGO
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_INDIGO

    table_data = [
        ("Exact Match Accuracy", "88.01%", "71.45%", "Linear model captures lexical markers"),
        ("Macro-F1 Score", "0.6874", "0.3656", "Imbalance triggers neural collapse"),
        ("Commitment Recall", "51.24%", "0.00%", "DistilBERT detected 0 commitments!"),
        ("Inference Latency", "0.02s", "57.57s", "2,878× latency advantage for TF-IDF"),
    ]
    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_WHITE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(13)
            p.font.bold = (c_idx in (1, 2) and r_idx == 2)
            p.font.color.rgb = COLOR_CORAL if (c_idx == 2 and r_idx == 2) else COLOR_NAVY

    # Callout card at bottom
    c_note = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.5))
    c_note.fill.solid()
    c_note.fill.fore_color.rgb = COLOR_WHITE
    c_note.line.color.rgb = COLOR_CORAL
    c_note.line.width = Pt(1.5)
    
    ntf = c_note.text_frame
    ntf.word_wrap = True
    np = ntf.paragraphs[0]
    np.text = "ROOT CAUSE OF NEURAL COLLAPSE:"
    np.font.size = Pt(12)
    np.font.bold = True
    np.font.color.rgb = COLOR_CORAL

    np2 = ntf.add_paragraph()
    np2.text = "96% negative instances overwhelmed unweighted Binary Cross-Entropy gradients during transformer fine-tuning. The neural model learned that predicting zero for all commitments minimized loss, proving why raw transformers fail on real enterprise imbalance."
    np2.font.size = Pt(13)
    np2.font.color.rgb = COLOR_NAVY
    np2.space_before = Pt(4)

    set_speaker_notes(s7, """
Here are our held-out test results across 9,657 unseen emails. Notice something striking: the simple TF-IDF baseline significantly outperformed standard fine-tuned DistilBERT, achieving 88% exact match accuracy versus 71%, and running nearly 3,000 times faster.

Even more dramatic: default DistilBERT suffered complete neural collapse on commitments, producing exactly 0.00% recall. Because 96% of training samples contained no commitment, standard unweighted cross-entropy loss learned to optimize loss by simply predicting zero for every single example. In contrast, n-gram TF-IDF retained strong lexical cues.
""")

    # =========================================================================
    # SLIDE 8: The Reweighting Trade-Off
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Class Balancing & The Engineering Reality")

    c_rec = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(3.2))
    c_rec.fill.solid()
    c_rec.fill.fore_color.rgb = COLOR_WHITE
    c_rec.line.color.rgb = COLOR_EMERALD
    c_rec.line.width = Pt(2.0)

    rtf = c_rec.text_frame
    rtf.word_wrap = True
    p = rtf.paragraphs[0]
    p.text = "RECOVERING COMMITMENT RECALL"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_EMERALD

    rp1 = rtf.add_paragraph()
    rp1.text = "0.00%  ➔  78.79%"
    rp1.font.size = Pt(36)
    rp1.font.bold = True
    rp1.font.color.rgb = COLOR_NAVY
    rp1.space_before = Pt(8)

    rp2 = rtf.add_paragraph()
    rp2.text = "Combined 20.0× Positive Loss Weighting + Batch Oversampling + Validation PR Threshold Tuning (θ = 0.85). Caught 286 / 363 true test commitments."
    rp2.font.size = Pt(13)
    rp2.font.color.rgb = COLOR_SLATE
    rp2.space_before = Pt(8)

    c_prec = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(3.2))
    c_prec.fill.solid()
    c_prec.fill.fore_color.rgb = COLOR_WHITE
    c_prec.line.color.rgb = COLOR_CORAL
    c_prec.line.width = Pt(2.0)

    ptf = c_prec.text_frame
    ptf.word_wrap = True
    p = ptf.paragraphs[0]
    p.text = "THE PRECISION TRADE-OFF"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_CORAL

    pp1 = ptf.add_paragraph()
    pp1.text = "9.55% Precision"
    pp1.font.size = Pt(36)
    pp1.font.bold = True
    pp1.font.color.rgb = COLOR_NAVY
    pp1.space_before = Pt(8)

    pp2 = ptf.add_paragraph()
    pp2.text = "Generated 2,709 false positives against weak labels (~1 real commitment per 10 alerts). A standalone model would flood an enterprise ticketing system."
    pp2.font.size = Pt(13)
    pp2.font.color.rgb = COLOR_SLATE
    pp2.space_before = Pt(8)

    # Bottom scientific conclusion
    c_take = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.5))
    c_take.fill.solid()
    c_take.fill.fore_color.rgb = COLOR_LIGHT_INDIGO
    c_take.line.color.rgb = COLOR_INDIGO
    c_take.line.width = Pt(1.5)

    ttf = c_take.text_frame
    ttf.word_wrap = True
    tp = ttf.paragraphs[0]
    tp.text = "CORE ARCHITECTURAL TAKEAWAY:"
    tp.font.size = Pt(12)
    tp.font.bold = True
    tp.font.color.rgb = COLOR_INDIGO

    tp2 = ttf.add_paragraph()
    tp2.text = "Neural classifiers cannot act as autonomous gates; they must function strictly as non-blocking advisory signals to human operators."
    tp2.font.size = Pt(15)
    tp2.font.bold = True
    tp2.font.color.rgb = COLOR_NAVY
    tp2.space_before = Pt(4)

    set_speaker_notes(s8, """
To solve the neural collapse, we engineered a combined intervention: 20x positive class weighting, batch oversampling, and threshold tuning on precision-recall curves. This successfully recovered commitment recall from 0% up to 78.8%, catching 286 out of 363 commitments.

However, being scientifically rigorous: this recall recovery came at the cost of precision. The model generated 2,709 false alarms, yielding a 9.55% precision. That means roughly 1 out of 10 alerts represents a true commitment against weak labels.

This empirical finding validates our system design: if you let a neural model write directly to a database, you corrupt your enterprise ledger with false positives. The model must remain an advisory signal to guide human attention.
""")

    # =========================================================================
    # SLIDE 9: LIVE DEMO
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Live System Demonstration (Streamlit)", "INTERACTIVE DEMO ROADMAP")

    demo_steps = [
        ("Tab 1: Task Ingestion & Grounding", "Paste workplace message -> review classifier probability signals -> inspect exact character evidence [c_start, c_end] -> confirm task to ledger."),
        ("Tab 2: Task Board & Calendar Grounding", "Filter active tasks -> view calendar date chips -> test 'Add to Google Calendar' sync -> inspect initial audit trail entry."),
        ("Tab 3: Semantic Follow-Up Linking", "Paste renegotiation reply -> MiniLM bi-encoder similarity ranking -> auto-detect status change -> commit immutable audit record.")
    ]
    for idx, (stitle, sdesc) in enumerate(demo_steps):
        cy = Inches(1.8) + idx * Inches(1.5)
        c = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), cy, Inches(11.7), Inches(1.2))
        c.fill.solid()
        c.fill.fore_color.rgb = COLOR_WHITE
        c.line.color.rgb = COLOR_CARD_BORDER
        c.line.width = Pt(1.5)

        dtf = c.text_frame
        dtf.word_wrap = True
        dp = dtf.paragraphs[0]
        dp.text = f"{idx+1}.  {stitle}"
        dp.font.size = Pt(16)
        dp.font.bold = True
        dp.font.color.rgb = COLOR_INDIGO

        dp2 = dtf.add_paragraph()
        dp2.text = sdesc
        dp2.font.size = Pt(13)
        dp2.font.color.rgb = COLOR_NAVY
        dp2.space_before = Pt(4)

    # Link banner
    c_link = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.0), Inches(11.7), Inches(0.8))
    c_link.fill.solid()
    c_link.fill.fore_color.rgb = COLOR_NAVY
    c_link.line.fill.background()

    ltf = c_link.text_frame
    lp = ltf.paragraphs[0]
    lp.text = "LIVE DEPLOYED APPLICATION:  https://followthrough-final-nlp-project.streamlit.app/"
    lp.alignment = PP_ALIGN.CENTER
    lp.font.size = Pt(14)
    lp.font.bold = True
    lp.font.color.rgb = COLOR_WHITE

    set_speaker_notes(s9, """
Now, let's transition directly to our deployed Streamlit application running live at followthrough-final-nlp-project.streamlit.app.

[SWITCH SCREEN TO BROWSER - 3.5 MINUTES]
1. Tab 1: Paste Standup Directives email. Highlight advisory confidence (85% Request, 80% Commitment). Highlight exact character offset slice [10:77]. Click 'Add to Task Board'.
2. Tab 2: Show task card with normalized calendar chip. Show 'Add to Google Calendar' link. Show initial audit log row.
3. Tab 3: Paste follow-up: 'I finished the budget report'. Show MiniLM bi-encoder 0.78 similarity score. Show detected status change 'Completed'. Click Apply. Return to Tab 2 to show COMPLETED badge and new audit log entry!
""")

    # =========================================================================
    # SLIDE 10: Safety & Governance
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Enterprise Safety & Data Governance")

    govs = [
        ("SHA-256 Deduplication", "Content Payload Fingerprinting", [
            "Computes SHA-256 hash over thread title, normalized action, and character span.",
            "Prevents duplicate task creation from forwarded emails and reply chains.",
            "Zero clutter in the operational database."
        ]),
        ("Append-Only Audit Trail", "Immutable State History Ledger", [
            "Every state transition logs ISO timestamp, operator ID, old value, and new value.",
            "Stores verbatim contextual evidence for every status change.",
            "Provides full compliance tracking for audits."
        ]),
        ("100% Local Privacy", "Zero External API Data Egress", [
            "DistilBERT, TF-IDF, and MiniLM execute on-premise without cloud calls.",
            "Zero conversational data leaves the corporate firewall.",
            "Fully compliant with NDA, GDPR, and HIPAA standards."
        ])
    ]
    for idx, (gtitle, gsub, gbullets) in enumerate(govs):
        cx = Inches(0.8) + idx * Inches(3.95)
        c = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, Inches(1.8), Inches(3.7), Inches(4.8))
        c.fill.solid()
        c.fill.fore_color.rgb = COLOR_WHITE
        c.line.color.rgb = COLOR_CARD_BORDER
        c.line.width = Pt(1.5)

        gtf = c.text_frame
        gtf.word_wrap = True
        p = gtf.paragraphs[0]
        p.text = gtitle
        p.font.size = Pt(17)
        p.font.bold = True
        p.font.color.rgb = COLOR_NAVY

        p_sub = gtf.add_paragraph()
        p_sub.text = gsub
        p_sub.font.size = Pt(12)
        p_sub.font.bold = True
        p_sub.font.color.rgb = COLOR_INDIGO
        p_sub.space_before = Pt(4)

        for b in gbullets:
            bp = gtf.add_paragraph()
            bp.text = f"•  {b}"
            bp.font.size = Pt(13)
            bp.font.color.rgb = COLOR_NAVY
            bp.space_before = Pt(14)

    set_speaker_notes(s10, """
When moving from research prototypes to production enterprise deployment, governance is non-negotiable. FollowThrough ensures data integrity through three safety mechanisms:

First, SHA-256 payload hashing prevents redundant task creation when employees reply to or forward email threads. Second, our SQLite backend operates in append-only Write-Ahead Logging mode. Records are never overwritten or silently modified; every state mutation records who authorized it, what the previous state was, and the verbatim message that triggered it.

And third, privacy: our models run completely on-premise without external API egress, ensuring sensitive corporate conversations never leave organizational boundaries.
""")

    # =========================================================================
    # SLIDE 11: Academic Contributions
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "Summary of Academic & Technical Contributions")

    contribs = [
        ("1. Zero-Leakage Disjoint Graph Partitioning", "Pioneered connected-component clustering across 98k Enron emails, proving that naive random splitting introduces evaluation bias through forwarded reply chains."),
        ("2. Verifiable Evidence Grounding", "Engineered deterministic character offset extraction [c_start, c_end] that eliminates generative LLM hallucination and preserves complete evidentiary provenance."),
        ("3. Dual-Track Advisory Intelligence", "Empirically documented the failure modes of transformers under extreme class imbalance (0% recall) and demonstrated why neural models must advise rather than gate."),
        ("4. Human-Centered Operational Ledger", "Constructed a production-grade multi-turn task tracker linking dense bi-encoder semantic retrieval (all-MiniLM-L6-v2) with an immutable cryptographic audit ledger.")
    ]
    for idx, (ctitle, cdesc) in enumerate(contribs):
        cy = Inches(1.8) + idx * Inches(1.25)
        c = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), cy, Inches(11.7), Inches(1.1))
        c.fill.solid()
        c.fill.fore_color.rgb = COLOR_WHITE
        c.line.color.rgb = COLOR_CARD_BORDER
        c.line.width = Pt(1.5)

        ctf = c.text_frame
        ctf.word_wrap = True
        p = ctf.paragraphs[0]
        p.text = ctitle
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = COLOR_INDIGO

        p2 = ctf.add_paragraph()
        p2.text = cdesc
        p2.font.size = Pt(12)
        p2.font.color.rgb = COLOR_NAVY
        p2.space_before = Pt(3)

    set_speaker_notes(s11, """
To summarize our capstone contributions:

We developed a leakage-free graph partitioning methodology for conversational corpora. We designed a deterministic extraction framework anchored to exact character spans. We empirically demonstrated why neural classifiers suffer collapse under real-world class imbalance, establishing the necessity of an advisory architecture.

And finally, we engineered an end-to-end, audit-verified application that bridges conversational NLP with reliable enterprise task tracking.
""")

    # =========================================================================
    # SLIDE 12: Conclusion & Q&A
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)

    top_bar = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), Inches(1.5), Inches(0.08))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = COLOR_INDIGO
    top_bar.line.fill.background()

    tb = s12.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(1.8))
    tf = tb.text_frame
    p1 = tf.paragraphs[0]
    p1.text = "Thank You!"
    p1.font.name = FONT_FAMILY
    p1.font.size = Pt(44)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_NAVY

    p2 = tf.add_paragraph()
    p2.text = "FollowThrough: Verifiable Conversational NLP"
    p2.font.size = Pt(22)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_INDIGO
    p2.space_before = Pt(8)

    links_box = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.0), Inches(11.7), Inches(2.5))
    links_box.fill.solid()
    links_box.fill.fore_color.rgb = COLOR_WHITE
    links_box.line.color.rgb = COLOR_CARD_BORDER
    links_box.line.width = Pt(1.5)

    ltf = links_box.text_frame
    ltf.word_wrap = True
    p = ltf.paragraphs[0]
    p.text = "PROJECT ARTIFACTS & LINKS"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_INDIGO

    items = [
        ("GitHub Repository", "https://github.com/Rohita-G/FollowThrough"),
        ("Live Deployed Application", "https://followthrough-final-nlp-project.streamlit.app/"),
        ("IEEE Technical Report", "project_report.pdf (Standard 2-Column IEEE Format)"),
    ]
    for lbl, url in items:
        ip = ltf.add_paragraph()
        ip.text = f"•  {lbl}:  {url}"
        ip.font.size = Pt(14)
        ip.font.bold = True
        ip.font.color.rgb = COLOR_NAVY
        ip.space_before = Pt(8)

    set_speaker_notes(s12, """
Thank you to the faculty committee, our mentors, and fellow students. The complete source code, dataset processing scripts, evaluation suites, and IEEE technical report are available in our public GitHub repository.

I would be delighted to answer any questions or demonstrate edge cases in the live application.
""")

    # Save presentation
    prs.save(str(OUTPUT_PPTX))
    print(f"Presentation saved successfully to: {OUTPUT_PPTX}")


if __name__ == "__main__":
    create_deck()
