#!/usr/bin/env python3
"""Script to generate a strict, authentic IEEE Conference two-column PDF and LaTeX source."""

import subprocess
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
HTML_OUT = DOCS_DIR / "project_report.html"
PDF_OUT = DOCS_DIR / "project_report.pdf"
PDF_ALT = DOCS_DIR / "FollowThrough_Research_Paper.pdf"

IEEE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FollowThrough: Evidence-Backed Conversational Task Extraction and State Tracking in Asynchronous Workplace Communication</title>
<style>
  @page {
    size: letter;
    margin: 0.75in 0.68in 0.75in 0.68in;
    @bottom-center {
      content: counter(page);
      font-family: "Times New Roman", Times, serif;
      font-size: 9pt;
    }
  }

  *, *:before, *:after {
    box-sizing: border-box;
  }

  body {
    font-family: "Times New Roman", Times, "Nimbus Roman No9 L", serif;
    font-size: 10pt;
    line-height: 1.18;
    color: #000000;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
    text-align: justify;
    text-justify: inter-word;
  }

  .paper-container {
    column-count: 2;
    column-gap: 0.22in;
    width: 100%;
    overflow-wrap: anywhere;
  }

  .title-banner {
    column-span: all;
    text-align: center;
    margin-bottom: 12pt;
    padding-bottom: 4pt;
  }

  h1.paper-title {
    font-size: 24pt;
    font-weight: normal;
    line-height: 1.15;
    margin: 0 0 8pt 0;
    color: #000000;
  }

  .author-name {
    font-size: 11pt;
    font-weight: bold;
    margin: 0 0 2pt 0;
  }

  .author-affil {
    font-size: 10pt;
    font-style: italic;
    margin: 0 0 12pt 0;
  }

  .abstract-text {
    font-size: 9pt;
    font-weight: bold;
    line-height: 1.2;
    text-align: justify;
    margin-bottom: 5pt;
    text-indent: 0.14in;
  }

  .abstract-head {
    font-style: italic;
    font-weight: bold;
  }

  .index-terms {
    font-size: 9pt;
    font-weight: bold;
    line-height: 1.2;
    text-align: justify;
    margin-bottom: 12pt;
    text-indent: 0.14in;
  }

  .index-head {
    font-style: italic;
    font-weight: bold;
  }

  h2.ieee-sec {
    font-size: 10pt;
    font-weight: normal;
    font-variant: small-caps;
    text-align: center;
    margin-top: 11pt;
    margin-bottom: 3pt;
    page-break-after: avoid;
    letter-spacing: 0.05em;
  }

  h3.ieee-subsec {
    font-size: 10pt;
    font-weight: normal;
    font-style: italic;
    text-align: left;
    margin-top: 7pt;
    margin-bottom: 2pt;
    text-indent: 0.14in;
    page-break-after: avoid;
  }

  p {
    text-indent: 0.16in;
    margin: 0 0 2.5pt 0;
    overflow-wrap: anywhere;
  }

  p.no-indent {
    text-indent: 0;
  }

  .eq-container {
    width: 100%;
    margin: 5pt 0;
    display: table;
    page-break-inside: avoid;
  }

  .eq-math {
    display: table-cell;
    text-align: center;
    font-style: italic;
    font-size: 9.5pt;
  }

  .eq-num {
    display: table-cell;
    width: 25px;
    text-align: right;
    font-size: 9.5pt;
    font-style: normal;
    vertical-align: middle;
  }

  .table-wrapper {
    margin: 8pt 0 10pt 0;
    page-break-inside: avoid;
    width: 100%;
  }

  .table-caption {
    font-size: 8pt;
    text-align: center;
    font-variant: small-caps;
    margin-bottom: 3pt;
    line-height: 1.2;
  }

  table.ieee-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 7.6pt;
    line-height: 1.15;
    margin: 0 auto;
    border-top: 1.2pt solid #000000;
    border-bottom: 1.2pt solid #000000;
  }

  table.ieee-table th {
    font-weight: normal;
    font-variant: small-caps;
    padding: 2.2pt 2.5pt;
    border-bottom: 0.8pt solid #000000;
    text-align: center;
  }

  table.ieee-table td {
    padding: 1.8pt 2.5pt;
    border: none;
    text-align: center;
  }

  table.ieee-table td.left-align, table.ieee-table th.left-align {
    text-align: left;
  }

  table.ieee-table-compact {
    font-size: 7.0pt;
  }

  table.ieee-table-compact th {
    padding: 2pt 1.5pt;
  }

  table.ieee-table-compact td {
    padding: 1.8pt 1.5pt;
  }

  .alg-container {
    margin: 8pt 0 10pt 0;
    border-top: 1.2pt solid #000000;
    border-bottom: 1.2pt solid #000000;
    padding: 3pt 0 4pt 0;
    font-size: 7.8pt;
    line-height: 1.2;
    page-break-inside: avoid;
    width: 100%;
  }

  .alg-title {
    font-variant: small-caps;
    font-weight: normal;
    font-size: 8pt;
    border-bottom: 0.8pt solid #000000;
    padding-bottom: 2pt;
    margin-bottom: 3pt;
  }

  .alg-body {
    font-family: "Courier New", Courier, monospace;
    font-size: 7.4pt;
    white-space: pre-wrap;
    line-height: 1.22;
  }

  .figure-wrapper {
    margin: 8pt 0 10pt 0;
    text-align: center;
    page-break-inside: avoid;
    width: 100%;
  }

  .figure-caption {
    font-size: 8pt;
    text-align: left;
    margin-top: 4pt;
    line-height: 1.2;
  }

  .references-container {
    margin-top: 8pt;
    font-size: 8pt;
    line-height: 1.15;
  }

  .ref-item {
    text-indent: -0.2in;
    padding-left: 0.2in;
    margin-bottom: 3.5pt;
    text-align: justify;
  }

  code {
    font-family: "Courier New", Courier, monospace;
    font-size: 7.4pt;
    word-break: break-all;
    overflow-wrap: anywhere;
  }
</style>
</head>
<body>

<div class="paper-container">

  <div class="title-banner">
    <h1 class="paper-title">FollowThrough: Evidence-Backed Conversational Task Extraction and State Tracking in Asynchronous Workplace Communication</h1>
    <div class="author-name">Rohita Gangishetty</div>
    <div class="author-affil">Department of Computer Science, Natural Language Processing Capstone &bull; Fall Term 2026</div>
  </div>

  <p class="abstract-text">
    <span class="abstract-head">Abstract</span>&mdash;In modern enterprise environments, asynchronous communication channels such as email threads and collaborative chat streams function as the primary substrate for organizational coordination. Within these interactions, action items frequently slip past deadlines, remain unassigned, or are silently invalidated by subsequent correspondence. Existing natural language processing (NLP) approaches typically frame task management as either isolated sentence classification, standard entity extraction, or open-ended generative summarization via Large Language Models (LLMs), which are prone to hallucinating obligations or misrepresenting literal temporal deadlines. To address these challenges, this paper presents FollowThrough, an evidence-backed conversational task extraction and dynamic state-tracking system. FollowThrough integrates four complementary stages: multi-label speech act signal classification, conservative rule-based linguistic extraction with character-level evidence grounding, dense bi-encoder semantic follow-up linking, and a human-in-the-loop audit ledger backed by SQLite with write-ahead logging (WAL). The pipeline is empirically evaluated using a curated corpus of 100,000 emails from the Enron dataset (97,988 retained messages, 1,442,694 sentences, 29,556,736 tokens), segmented into thread-isolated train (78,385), validation (9,946), and test (9,657) sets via connected-component graph partitioning to eliminate data leakage. We systematically compare a balanced TF-IDF linear baseline against a fine-tuned DistilBERT transformer with a 128-token sliding-window inference architecture. On the held-out test split, TF-IDF achieves an exact match accuracy of 0.8801 and a macro-F1 of 0.6874 against programmatic weak labels, outperforming default fine-tuned DistilBERT (exact match 0.7145, macro-F1 0.3656) due to weak-supervision surface lexical alignment and severe commitment class imbalance (4.08% training prevalence). A combined intervention of class-weighted loss (20.0&times;) and batch oversampling recovers test commitment recall to 0.7879 at validation-tuned thresholds, exhibiting a 0.0955 precision trade-off (2,709 false positives). Crucially, FollowThrough operates as a non-gating advisory tool ensuring verifiable provenance and human-in-the-loop auditability for corporate communication.
  </p>

  <p class="index-terms">
    <span class="index-head">Index Terms</span>&mdash;Conversational task extraction, speech act classification, state tracking, weak supervision, audit ledger, human-in-the-loop NLP, Enron corpus.
  </p>

  <h2 class="ieee-sec">I. Introduction</h2>
  <p>
    Asynchronous, text-based communication represents the operational backbone of modern knowledge work. In enterprise organizations, platforms such as email clients (e.g., Microsoft Outlook, Gmail) and messaging ecosystems (e.g., Slack, Microsoft Teams) host thousands of daily micro-coordinations. Within these exchanges, collaborators routinely establish two fundamental speech acts that govern workplace momentum: <strong>requests</strong> (solicitations for action, information, or review directed toward others) and <strong>commitments</strong> (explicit undertakings, pledges, or promises made by the author).
  </p>
  <p>
    Despite their ubiquity, conversational obligations are inherently fluid and difficult to manage. Unlike structured project management systems (e.g., Jira, Asana) where tasks are defined as formal schema-constrained database entities, conversational tasks emerge spontaneously within unstructured natural language. Over the lifecycle of a typical email thread, initial commitments are frequently renegotiated: deadlines slip from "tomorrow morning" to "next Friday," tasks are delegated to third-party colleagues, project scopes expand or contract, and prior requests are superseded or rendered obsolete by subsequent messages. In corporate environments characterized by high communication volume and distributed teams, individuals experience cognitive overload. Consequently, actionable obligations become buried beneath conversational noise, leading to forgotten deliverables, missed project milestones, duplicate effort, and operational friction.
  </p>
  <p>
    The broader research area of natural language understanding (NLU) has long investigated computational techniques to extract structure from unstructured text. Historically, researchers have applied text classification, intent recognition, information extraction, and semantic parsing to automate task detection. In recent years, the rapid advancement of pretrained transformers and Large Language Models (LLMs) has prompted widespread interest in end-to-end task automation. However, applying NLU to operational task tracking exposes three primary technical challenges:
  </p>
  <p>
    <em>1) The Static Context Fallacy:</em> Traditional machine learning approaches treat action-item detection as an isolated, single-utterance classification or sequence-tagging task. By processing individual sentences in isolation, these models cannot maintain contextual continuity across multi-turn threads, remaining blind to subsequent conversational turns that revise, reassign, or cancel previously agreed-upon tasks.
  </p>
  <p>
    <em>2) Generative Hallucination and Unverifiable Outputs:</em> While generative LLMs can summarize conversations and identify action items, their unconstrained autoregressive nature frequently induces hallucinations. In enterprise settings, an LLM might infer an unstated deadline, fabricate an assignee based on associative bias, or hallucinate a task that was never agreed upon. Furthermore, generative outputs do not provide native character-level provenance linking the extracted entity back to the exact source span in the original communication.
  </p>
  <p>
    <em>3) The Premature Automation Dilemma:</em> Autonomous agents that programmatically write tasks directly into enterprise management software without human oversight introduce organizational risk. False positives pollute issue trackers with phantom tickets, while false negatives cause missed obligations. Without verifiable, evidence-backed human oversight, users quickly lose confidence in automated systems.
  </p>
  <h3 class="ieee-subsec">A. Research Objectives and Key Contributions</h3>
  <p>
    The primary objective of the <strong>FollowThrough</strong> project is to design, implement, and evaluate an evidence-backed conversational task extraction and state-tracking system that reconciles natural language processing with operational verifiability. Specifically, this research addresses four primary research questions:
  </p>
  <p>
    &bull; <em>RQ1:</em> How effectively can multi-label sequence classification models (both classical linear baselines and fine-tuned transformer architectures) detect request and commitment speech acts within large-scale, noisy workplace email corpora?<br>
    &bull; <em>RQ2:</em> Can conservative linguistic extraction combined with exact character-level evidence tracking mitigate generative hallucinations while capturing actionable task descriptions, explicit assignees, and literal deadlines?<br>
    &bull; <em>RQ3:</em> How well can dense bi-encoder semantic sentence representations (<code>all-MiniLM-L6-v2</code>) link subsequent conversational follow-up messages to open parent tasks across multi-turn threads?<br>
    &bull; <em>RQ4:</em> How can a human-in-the-loop user interface and an immutable, append-only persistence layer be structured to guarantee auditability and operational integrity?
  </p>
  <p>
    The principal contributions of this work are fourfold:
  </p>
  <p>
    1) <em>Large-Scale Data Pipeline and Disjoint Graph Partitioning:</em> We develop a reproducible data ingestion, cleaning, thread reconstruction, and weak-supervision pipeline operating over 100,000 raw emails from the Enron corpus. To prevent data leakage across email duplicates and conversational threads, we implement a disjoint-set (Union-Find) connected-component clustering algorithm that partitions the dataset into thread-isolated train, validation, and test splits (78,385, 9,946, and 9,657 messages respectively).
  </p>
  <p>
    2) <em>Dual-Model Comparative Benchmark and Sliding-Window Inference:</em> We implement, train, and empirically evaluate two distinct sequence classification models: a sublinear TF-IDF + Logistic Regression baseline and a fine-tuned <code>distilbert-base-uncased</code> neural classifier. Furthermore, we develop a 128-token overlapping sliding-window inference algorithm with a 64-token stride that enables transformer-based classification over arbitrarily long corporate emails while preserving exact token-to-character offset alignments.
  </p>
  <p>
    3) <em>Evidence-Backed Conservative Extraction and Temporal Grounding:</em> We construct a deterministic linguistic extraction engine that identifies candidate tasks, assignees, and deadlines without generative hallucination. Every extracted field is permanently bound to its exact character offsets (<code>[start_offset:end_offset]</code>) in the source text and coupled with transparent uncertainty flags. In addition, we implement a temporal grounding engine that resolves relative expressions against message header timestamps into standardized ISO-8601 calendar dates.
  </p>
  <p>
    4) <em>Interactive Audit Ledger and State Tracking:</em> We design an interactive application using Streamlit and SQLite (configured in Write-Ahead Logging mode). The system features an immutable, append-only <code>task_history</code> table that cryptographically deduplicates tasks via SHA-256 fingerprinting and records every state transition with full textual provenance and mandatory user authorization.
  </p>

  <h2 class="ieee-sec">II. Problem Statement</h2>
  <p>
    To formally define the operational challenge addressed by FollowThrough, consider an asynchronous conversation thread <em>T</em> = {<em>m</em><sub>1</sub>, <em>m</em><sub>2</sub>, ..., <em>m</em><sub><em>K</em></sub>} comprising an ordered sequence of <em>K</em> messages exchanged among a set of participants <em>P</em> = {<em>p</em><sub>1</sub>, <em>p</em><sub>2</sub>, ..., <em>p</em><sub><em>J</em></sub>}. Each message <em>m</em><sub><em>k</em></sub> consists of metadata (sender <em>s</em><sub><em>k</em></sub> &isin; <em>P</em>, recipients <em>R</em><sub><em>k</em></sub> &sube; <em>P</em>, timestamp &tau;<sub><em>k</em></sub>) and an unstructured body string <em>C</em><sub><em>k</em></sub>. Within any message <em>m</em><sub><em>k</em></sub>, an author may articulate one or more forward-looking task instances &tau; = (<em>a</em>, <em>o</em>, <em>d</em>, &sigma;), where <em>a</em> denotes the action description, <em>o</em> &isin; <em>P</em> &cup; {Unspecified} denotes the assigned owner, <em>d</em> denotes the temporal deadline or fulfillment target, and &sigma; &isin; {Pending, Completed, Cancelled} denotes the operational status.
  </p>
  <p>
    The core problem is to continuously detect, extract, and update the set of active tasks &Tau; across the evolution of thread <em>T</em>, while satisfying three fundamental constraints:
  </p>
  <p>
    <em>1) Linguistic Verifiability:</em> For every extracted task component (<em>a</em>, <em>o</em>, <em>d</em>), there must exist an exact character span [<em>c</em><sub>start</sub>, <em>c</em><sub>end</sub>] within message body <em>C</em><sub><em>k</em></sub> such that the extracted information is directly grounded in the text:
  </p>
  <div class="eq-container">
    <div class="eq-math">Substring(C<sub>k</sub>, c<sub>start</sub>, c<sub>end</sub>) &equiv; Evidence</div>
    <div class="eq-num">(1)</div>
  </div>
  <p class="no-indent">
    The extraction mechanism must not generate, interpolate, or extrapolate entities that lack explicit textual grounding.
  </p>
  <p>
    <em>2) Conversational Dynamic Alignment:</em> When a subsequent message <em>m</em><sub><em>k+j</em></sub> (<em>j</em> &ge; 1) arrives containing an update (such as "I've completed the slides" or "We need to push the review to Friday"), the system must correctly map <em>m</em><sub><em>k+j</em></sub> to its corresponding antecedent task &tau; &isin; &Tau; and propose the appropriate state transition &sigma; &rarr; &sigma;' or parameter revision <em>d</em> &rarr; <em>d'</em>, rather than creating a redundant, disconnected task.
  </p>
  <p>
    <em>3) Human Authorization and Auditability:</em> Because conversational language is inherently ambiguous, sarcastic, or tentative, autonomous systems cannot be permitted to make irrevocable modifications to organizational task ledgers. Every state change must pass through an interactive verification gate and be recorded in an immutable ledger &Lscr; that logs the operator, timestamp, previous state, new state, and supporting contextual evidence.
  </p>

  <h3 class="ieee-subsec">A. Limitations of Existing Solutions</h3>
  <p>
    Current industrial and academic approaches fail to satisfy these operational constraints:
  </p>
  <p>
    &bull; <em>Heuristic Regex / Rule-Based Enterprise Filters:</em> Early enterprise email systems deployed basic regular expressions matching keywords like "TODO:" or "Action Item:". While computationally negligible, these heuristics exhibit low recall: professional workplace communications rarely use explicit labels, relying instead on nuanced speech acts such as indirect requests ("Would you happen to have time to review the balance sheet?") and embedded commitments ("I'll make sure the servers are patched tonight").
  </p>
  <p>
    &bull; <em>Isolated Sequence Labeling (Token-Level NER):</em> Standard token-level sequence labelers (e.g., BiLSTM-CRF or BERT-NER) trained on datasets such as CoNLL-2003 or OntoNotes identify entities (Person, Date, Organization) but fail to capture the predicate-argument structure that defines an actionable task. Recognizing that "Friday" is a <code>DATE</code> and "Bob" is a <code>PERSON</code> does not determine whether Bob is requested to complete an action by Friday or whether Bob is merely reporting a past event.
  </p>
  <p>
    &bull; <em>Unconstrained Autoregressive LLMs:</em> Contemporary solutions frequently prompt generative models (e.g., GPT-4) to ingest an email thread and output JSON-formatted action items. While capable of synthesizing colloquial text, generative LLMs suffer from three critical weaknesses: (1) <em>Hallucination:</em> Models frequently infer deadlines that do not exist, fabricate assignees based on associative bias, or invent tasks based on pleasantries; (2) <em>Loss of Exact Span Provenance:</em> Because generation is autoregressive, LLMs rephrase actions into abstract summaries, severing the link to the original character offsets; and (3) <em>Latency and Privacy Constraints:</em> Routing proprietary enterprise communications containing sensitive commercial contracts or personal data to cloud-hosted LLM endpoints violates corporate data governance frameworks (GDPR, HIPAA, SOC-2).
  </p>

  <h2 class="ieee-sec">III. System Workflow & Architecture</h2>
  <h3 class="ieee-subsec">A. High-Level Modular Design</h3>
  <p>
    The architecture of FollowThrough is engineered around the principle of <strong>modular separation of concerns with human-in-the-loop verification</strong>. Rather than deploying a monolithic end-to-end model, the system divides the problem into three decoupled subsystems: the Data Ingestion and Lineage Pipeline, the Multi-Stage NLP Inference Engine, and the Persistent State Ledger.
  </p>

  <div class="figure-wrapper">
    <svg width="100%" height="345" viewBox="0 0 350 345" xmlns="http://www.w3.org/2000/svg" style="border: 0.8pt solid #000; background: #ffffff;">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#000000"/>
        </marker>
        <marker id="dotted-arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#444444"/>
        </marker>
      </defs>

      <!-- Input Message Block -->
      <rect x="40" y="8" width="270" height="25" rx="3" fill="#f4f4f4" stroke="#000" stroke-width="1.1"/>
      <text x="175" y="24" font-family="Times New Roman, serif" font-size="8.5" font-weight="bold" text-anchor="middle">Raw Workplace Message (Email / Thread Text)</text>

      <!-- Split Arrow from Input -->
      <line x1="175" y1="33" x2="175" y2="42" stroke="#000" stroke-width="1"/>
      <line x1="88" y1="42" x2="262" y2="42" stroke="#000" stroke-width="1"/>
      <line x1="88" y1="42" x2="88" y2="52" stroke="#000" stroke-width="1" marker-end="url(#arrow)"/>
      <line x1="262" y1="42" x2="262" y2="52" stroke="#000" stroke-width="1" marker-end="url(#arrow)"/>

      <!-- Stage 1 Left: Linguistic Extractor -->
      <rect x="10" y="52" width="156" height="66" rx="3" fill="#ffffff" stroke="#000" stroke-width="1.1"/>
      <text x="88" y="65" font-family="Times New Roman, serif" font-size="8.2" font-weight="bold" text-anchor="middle">Linguistic Task Extractor</text>
      <line x1="10" y1="70" x2="166" y2="70" stroke="#bbb" stroke-width="0.8"/>
      <text x="16" y="81" font-family="Times New Roman, serif" font-size="6.8">&bull; Deontic modals &amp; imperative syntax</text>
      <text x="16" y="92" font-family="Times New Roman, serif" font-size="6.8">&bull; Exact spans: [c<tspan font-size="5.5" baseline-shift="sub">start</tspan>, c<tspan font-size="5.5" baseline-shift="sub">end</tspan>] offsets</text>
      <text x="16" y="103" font-family="Times New Roman, serif" font-size="6.8">&bull; RFC-822 ISO temporal normalizer</text>
      <text x="16" y="113" font-family="Times New Roman, serif" font-size="6.8">&bull; Explicit uncertainty / owner flags</text>

      <!-- Stage 1 Right: Speech Act Classifiers -->
      <rect x="184" y="52" width="156" height="66" rx="3" fill="#ffffff" stroke="#000" stroke-width="1.1"/>
      <text x="262" y="65" font-family="Times New Roman, serif" font-size="8.2" font-weight="bold" text-anchor="middle">Speech Act Classifiers</text>
      <line x1="184" y1="70" x2="340" y2="70" stroke="#bbb" stroke-width="0.8"/>
      <text x="190" y="81" font-family="Times New Roman, serif" font-size="6.8">&bull; Sliding-Window DistilBERT (128 tok)</text>
      <text x="190" y="92" font-family="Times New Roman, serif" font-size="6.8">&bull; Balanced TF-IDF linear baseline</text>
      <text x="190" y="103" font-family="Times New Roman, serif" font-size="6.8">&bull; Multi-label request &amp; commitment</text>
      <text x="190" y="113" font-family="Times New Roman, serif" font-size="6.8" font-style="italic" fill="#444">&bull; Advisory signal (Non-Gating)</text>

      <!-- Arrow down from Extractor to Bi-Encoder -->
      <line x1="88" y1="118" x2="88" y2="136" stroke="#000" stroke-width="1" marker-end="url(#arrow)"/>
      <text x="91" y="129" font-family="Times New Roman, serif" font-size="6.2" font-style="italic">Candidate Tasks</text>

      <!-- Dotted Line down from Classifiers to Human Review Gate -->
      <line x1="262" y1="118" x2="262" y2="204" stroke="#444" stroke-width="1.1" stroke-dasharray="3,3" marker-end="url(#dotted-arrow)"/>
      <rect x="198" y="150" width="128" height="18" rx="2" fill="#f4f4f4" stroke="#777" stroke-width="0.7"/>
      <text x="262" y="162" font-family="Times New Roman, serif" font-size="6.5" font-style="italic" text-anchor="middle">Advisory Confidence (Non-Gating)</text>

      <!-- Stage 2: Bi-Encoder Semantic Linker -->
      <rect x="10" y="136" width="156" height="54" rx="3" fill="#ffffff" stroke="#000" stroke-width="1.1"/>
      <text x="88" y="149" font-family="Times New Roman, serif" font-size="8.2" font-weight="bold" text-anchor="middle">Bi-Encoder Semantic Linker</text>
      <line x1="10" y1="154" x2="166" y2="154" stroke="#bbb" stroke-width="0.8"/>
      <text x="16" y="165" font-family="Times New Roman, serif" font-size="6.8">&bull; all-MiniLM-L6-v2 (384-d vectors)</text>
      <text x="16" y="175" font-family="Times New Roman, serif" font-size="6.8">&bull; Cosine similarity vs. open tasks</text>
      <text x="16" y="185" font-family="Times New Roman, serif" font-size="6.8">&bull; Follow-up &amp; revision resolution</text>

      <!-- Arrow down from Bi-Encoder to Review Gate -->
      <line x1="88" y1="190" x2="88" y2="204" stroke="#000" stroke-width="1" marker-end="url(#arrow)"/>

      <!-- Stage 3: Mandatory Human Review Gate -->
      <rect x="10" y="206" width="330" height="58" rx="3" fill="#fafafa" stroke="#000" stroke-width="1.6"/>
      <rect x="228" y="210" width="106" height="12" rx="2" fill="#000"/>
      <text x="281" y="219" font-family="Times New Roman, serif" font-size="6.2" font-weight="bold" fill="#fff" text-anchor="middle">ZERO UNVERIFIED WRITES</text>
      <text x="18" y="222" font-family="Times New Roman, serif" font-size="8.5" font-weight="bold">Mandatory Human Review Gate (Interactive UI)</text>
      <line x1="10" y1="226" x2="340" y2="226" stroke="#000" stroke-width="0.8"/>
      <text x="18" y="238" font-family="Times New Roman, serif" font-size="7.0">&bull; Verbatim evidence inspection with highlighted character offsets &amp; model scores</text>
      <text x="18" y="249" font-family="Times New Roman, serif" font-size="7.0">&bull; Modify action description, assignees, or deadlines; dismiss false-positive candidates</text>
      <text x="18" y="259" font-family="Times New Roman, serif" font-size="7.0" font-weight="bold">&bull; Mandatory Authorization: [Confirm &amp; Commit Task]  |  [Dismiss / Reject]</text>

      <!-- Arrow down from Review Gate to Ledger -->
      <line x1="175" y1="264" x2="175" y2="280" stroke="#000" stroke-width="1.3" marker-end="url(#arrow)"/>
      <text x="175" y="275" font-family="Times New Roman, serif" font-size="6.6" font-weight="bold" text-anchor="middle" fill="#000">Explicit User Authorization</text>

      <!-- Stage 4: SQLite Ledger -->
      <rect x="25" y="282" width="300" height="52" rx="3" fill="#ffffff" stroke="#000" stroke-width="1.2"/>
      <text x="175" y="296" font-family="Times New Roman, serif" font-size="8.5" font-weight="bold" text-anchor="middle">Cryptographic State Store &amp; Ledger (SQLite WAL)</text>
      <line x1="25" y1="301" x2="325" y2="301" stroke="#bbb" stroke-width="0.8"/>
      <text x="32" y="313" font-family="Times New Roman, serif" font-size="7.0">&bull; tasks table: SHA-256 fingerprint deduplication &amp; active task state machine</text>
      <text x="32" y="325" font-family="Times New Roman, serif" font-size="7.0">&bull; task_history table: Append-only immutable audit trail of all lifecycle mutations</text>

      <!-- Feedback Loop Line from Ledger to Bi-Encoder -->
      <line x1="25" y1="308" x2="4" y2="308" stroke="#555" stroke-width="0.9"/>
      <line x1="4" y1="308" x2="4" y2="163" stroke="#555" stroke-width="0.9"/>
      <line x1="4" y1="163" x2="10" y2="163" stroke="#555" stroke-width="0.9" marker-end="url(#dotted-arrow)"/>
      <text x="7" y="238" font-family="Times New Roman, serif" font-size="6.2" font-style="italic" fill="#555" transform="rotate(-90 7 238)" text-anchor="middle">Active Tasks State</text>
    </svg>
    <div class="figure-caption">Fig. 1. End-to-end system architecture of FollowThrough. Unstructured messages are concurrently processed by deterministic linguistic extraction (with character-offset grounding and temporal normalization) and multi-label sequence classifiers (providing non-gating advisory confidence). The bi-encoder semantically links candidate follow-ups to active task states. Crucially, the human review gate enforces mandatory operator verification before state mutations are committed to the immutable append-only SQLite ledger.</div>
  </div>

  <h3 class="ieee-subsec">B. Component Workflow and End-to-End Execution</h3>
  <p>
    <em>1) Ingestion and Non-Gating Advisory Classification:</em> When an unstructured email or chat message enters the system via Tab 1 of the Streamlit interface, it is simultaneously routed to two distinct processing modules: the Sequence Classification Subsystem and the Linguistic Extraction Subsystem. Crucially, the classifiers do not serve as a hard filtering gate for the extractor. In traditional cascading pipelines, if a classifier predicts that a sentence is not an action item, downstream extraction is aborted. In FollowThrough, because weak-supervision labels carry inherent noise and human workplace commitments exhibit severe class imbalance, the sequence classifiers provide <em>advisory likelihood scores</em> (surfaced as visual confidence metrics in the UI). This ensures that subtle or rare commitments are never prematurely discarded by upstream classification errors. If an input message exceeds the 128-token context window of the fine-tuned DistilBERT model, the sliding-window engine partitions the text into overlapping sub-sequences of 126 content tokens plus <code>[CLS]</code> and <code>[SEP]</code> tokens, utilizing a 64-token stride. The token-level predictions are pooled using max-logit aggregation, and the exact character spans corresponding to each window are retained.
  </p>
  <p>
    <em>2) Deterministic Evidence Extraction and Temporal Grounding:</em> Concurrently, the input message is processed by the <code>TaskExtractor</code>. This module executes a multi-stage deterministic pipeline. First, it scans the message for reply header boundaries (e.g., <code>-----Original Message-----</code>, <code>From:</code>, <code>wrote:</code>, lines beginning with <code>&gt;</code>). Trailing quoted correspondence is segmented to ensure that action items are attributed strictly to the current author, while forwarded content is flagged to notify the user of potential attribution ambiguity. Next, the extractor evaluates sentences against conservative syntactic templates anchored by deontic and modal verb phrases (e.g., <em>please review</em>, <em>could you verify</em>, <em>I will deliver</em>, <em>we plan to</em>). Candidate actions are extracted as clean imperative phrases. The exact character coordinates [<em>c</em><sub>start</sub>, <em>c</em><sub>end</sub>] of the extracted sentence within the raw input string are recorded. If an explicit assignee cannot be resolved from vocative openers or pronoun structures, the field is assigned <code>"Unspecified"</code>, and an uncertainty flag <code>missing_explicit_owner</code> is attached. Furthermore, if a temporal expression (e.g., "by next Tuesday at 3pm") is identified, the <code>TemporalNormalizer</code> inspects the email's RFC 822 <code>Date</code> header, establishes an anchor datetime, and computes the exact resolved calendar date using modular weekday arithmetic, generating an ISO-8601 string alongside the literal surface text.
  </p>
  <p>
    <em>3) Interactive Human Review Gate:</em> The candidate tasks, accompanied by their advisory classification scores, normalized deadlines, and exact character-level evidence spans, are presented within the Streamlit UI. The operator reviews the proposed fields in an editable form. The user maintains full agency to modify the action description, assign a specific team member, adjust the target deadline, or reject false-positive candidates entirely. Only upon explicit human button confirmation ("Confirm & Save Task") does the system initiate a persistent write.
  </p>
  <p>
    <em>4) Cryptographic Storage and Audit Ledger:</em> The confirmed record is transmitted to the <code>TaskStore</code> persistence layer. The store calculates a cryptographic fingerprint <em>h</em> = SHA-256(<em>C</em><sub>source</sub> &parallel; <em>a</em>), where <em>C</em><sub>source</sub> is the source text and <em>a</em> is the action description. If an identical fingerprint already exists in the <code>tasks</code> table, the operation is safely rejected, preventing duplicate records.
  </p>
  <p>
    <em>5) Dense Semantic Follow-Up Linking and State Revision:</em> When a collaborator sends a subsequent message into Tab 3, the <code>FollowupLinker</code> queries the SQLite database for all currently open tasks. It uses <code>all-MiniLM-L6-v2</code> to compute dense 384-dimensional sentence embeddings for the follow-up text and all candidate task descriptions, calculating pairwise cosine similarities:
  </p>
  <div class="eq-container">
    <div class="eq-math">Sim(u, v) = (u &middot; v) / (||u||<sub>2</sub> ||v||<sub>2</sub>)</div>
    <div class="eq-num">(2)</div>
  </div>
  <p class="no-indent">
    Simultaneously, regular expression patterns scan the follow-up text for explicit state mutations: deadline revisions, owner reassignments, and completions or cancellations. When the user verifies the semantic link and accepts the proposed revision, the <code>TaskStore</code> updates the primary <code>tasks</code> record and appends a row to the <code>task_history</code> table logging the timestamp, update type, old value, new value, exact evidence, and full follow-up text.
  </p>

  <h2 class="ieee-sec">IV. Tools and Technologies</h2>
  <p>
    The FollowThrough system is implemented entirely in modern Python, leveraging established open-source libraries for natural language processing, deep learning, data manipulation, database management, and interactive web visualization.
  </p>

  <div class="table-wrapper">
    <div class="table-caption">TABLE I<br>TOOLS AND TECHNOLOGIES EMPLOYED IN FOLLOWTHROUGH</div>
    <table class="ieee-table">
      <thead>
        <tr>
          <th class="left-align">Component</th>
          <th>Specification</th>
          <th class="left-align">Role in System Architecture</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="left-align">Python Language</td>
          <td>Version 3.10+</td>
          <td class="left-align">Runtime environment; MIME parser & regex</td>
        </tr>
        <tr>
          <td class="left-align">PyTorch</td>
          <td>Version &ge; 2.0.0</td>
          <td class="left-align">Deep learning graph & MPS acceleration</td>
        </tr>
        <tr>
          <td class="left-align">Transformers</td>
          <td>DistilBERT</td>
          <td class="left-align">Sequence classification backbone</td>
        </tr>
        <tr>
          <td class="left-align">Sentence-Trans.</td>
          <td>all-MiniLM-L6-v2</td>
          <td class="left-align">384-d dense bi-encoder vector retrieval</td>
        </tr>
        <tr>
          <td class="left-align">Scikit-Learn</td>
          <td>Version &ge; 1.3.0</td>
          <td class="left-align">Sublinear TF-IDF & Logistic Regression</td>
        </tr>
        <tr>
          <td class="left-align">NumPy & Pandas</td>
          <td>Latest stable</td>
          <td class="left-align">Matrix operations & split manifest</td>
        </tr>
        <tr>
          <td class="left-align">SQLite3</td>
          <td>WAL Mode</td>
          <td class="left-align">Embedded ACID persistence store</td>
        </tr>
        <tr>
          <td class="left-align">Streamlit</td>
          <td>Version &ge; 1.25.0</td>
          <td class="left-align">3-tab reactive web interface</td>
        </tr>
        <tr>
          <td class="left-align">Hardware</td>
          <td>Apple M-Series</td>
          <td class="left-align">8-core CPU, MPS GPU, 16GB RAM</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 class="ieee-subsec">A. Hardware Infrastructure and Resource Profiling</h3>
  <p>
    The training, evaluation, and interactive hosting of FollowThrough were executed on dedicated local hardware: Apple Silicon M-series ARM64 architecture (8-core CPU: 4 performance cores, 4 efficiency cores), Apple Metal Performance Shaders (MPS) framework integrated with PyTorch, 16.0 GB Unified Memory Subsystem with 100 GB/s bandwidth, and 512 GB PCIe NVMe SSD.
  </p>
  <p>
    During model fine-tuning, the 66-million-parameter DistilBERT network occupied approximately 1.4 GB of unified GPU memory under a batch size of 2 and maximum sequence length of 128 tokens. Training completed across 5,000 gradient update steps in approximately 35 minutes using Apple MPS acceleration. Evaluation across the entire held-out test split of 9,657 messages completed in 57.57 seconds on MPS (averaging 5.96 ms per email), whereas the linear TF-IDF baseline executed inference over the identical 9,657 messages in only 0.02 seconds on CPU. The persistent model weights occupy 255.4 MB on disk for DistilBERT and 0.52 MB for the TF-IDF vectorizer and logistic regression coefficients.
  </p>

  <h2 class="ieee-sec">V. Approaches & Methodology</h2>
  <h3 class="ieee-subsec">A. Datasets and Corpus Lineage</h3>
  <p>
    The corpus is derived from the public Carnegie Mellon University Enron Email Dataset (May 7, 2015 archive release, <code>enron_mail_20150507.tar.gz</code>, 423 MB compressed, ~1.7 GB uncompressed maildir). The raw dataset comprises approximately 500,000 plain-text message files across 150 senior executive user mailboxes. Mailbox prioritization selects 100,000 files prioritizing core organizational participants (<code>kaminski-v</code>, <code>dasovich-j</code>, <code>farmer-d</code>, <code>shankman-j</code>) across standard communication folders (<code>inbox</code>, <code>sent</code>, <code>_sent_mail</code>, <code>notes_inbox</code>). After RFC 822 parsing, quote stripping, signature removal, and fragment exclusion (&lt;10 characters), <strong>97,988 high-quality records</strong> are retained, encompassing <strong>1,442,694 cleaned sentences</strong> and <strong>29,556,736 tokens</strong>.
  </p>

  <h3 class="ieee-subsec">B. Thread Reconstruction and Disjoint Graph Partitioning</h3>
  <p>
    To prevent data leakage caused by duplicate emails across mailboxes and contextual overlap within conversation threads, FollowThrough implements a <strong>Disjoint-Set (Union-Find) Graph Partitioning Algorithm</strong>:
  </p>
  <p>
    1) <em>Thread Reconstruction:</em> Subject lines are normalized by recursively stripping iterative prefixes (<code>Re:</code>, <code>Fwd:</code>). Messages sharing the normalized subject and overlapping participant sets are grouped into 42,317 distinct conversation threads.
  </p>
  <p>
    2) <em>Duplicate Fingerprinting:</em> Each message is assigned a duplicate key <em>K</em><sub>dup</sub> = SHA-256(<em>C</em><sub>body</sub>) &parallel; Sender &parallel; NormalizedSubject.
  </p>
  <p>
    3) <em>Connected Component Clustering:</em> Using a Disjoint-Set forest with union-by-rank and path compression, all messages belonging to the same conversation thread ID or sharing an identical duplicate key are clustered into an atomic entity.
  </p>
  <p>
    4) <em>Deterministic Split Hashing:</em> Each connected component cluster is assigned a deterministic pseudo-random hash value using a fixed seed (<code>seed=42</code>). Clusters are allocated to splits according to target proportions of approximately 80% Train (78,385 messages), 10% Validation (9,946 messages), and 10% Test (9,657 messages). This partitioning is designed to eliminate thread and duplicate leakage across splits.
  </p>

  <h3 class="ieee-subsec">C. Class Prevalence Across Splits</h3>
  <p>
    Crucially, commitment speech acts exhibit severe class imbalance across the partitioned corpus:
  </p>
  <p>
    &bull; <em>Training Split (N = 78,385):</em> 35,141 requests (44.83%) and 3,201 commitments (<strong>4.08% prevalence</strong>).<br>
    &bull; <em>Validation Split (N = 9,946):</em> 4,549 requests (45.74%) and 392 commitments (<strong>3.94% prevalence</strong>).<br>
    &bull; <em>Test Split (N = 9,657):</em> 4,270 requests (44.22%) and 363 commitments (<strong>3.76% prevalence</strong>).
  </p>
  <p>
    Distinguishing training prevalence (4.08%) from test prevalence (3.76%) is critical when assessing class weighting ratios and baseline expectations.
  </p>

  <h3 class="ieee-subsec">D. Programmatic Weak Supervision</h3>
  <p>
    We adopt a transparent weak-supervision methodology grounded in linguistic domain rules. We define two binary labeling functions for each message:
  </p>
  <p>
    &bull; <em>Request Speech Act (&lambda;<sub>req</sub>):</em> Fires if the message contains explicit interrogative or imperative deontic structures (<em>please send</em>, <em>could you provide</em>, <em>can you verify</em>, <em>let me know if</em>, <em>action required</em>).
  </p>
  <p>
    &bull; <em>Commitment Speech Act (&lambda;<sub>com</sub>):</em> Fires if the message contains first-person promissory or intentional commissive structures (<em>I will send</em>, <em>we will deliver</em>, <em>I'll follow up</em>, <em>I plan to</em>).
  </p>
  <p>
    &bull; <em>Negative Syntactic Guards:</em> Filter out past-tense reporting ("I sent it yesterday"), third-party descriptions, and hypothetical conditionals.
  </p>

  <h3 class="ieee-subsec">E. Sequence Classification Approaches</h3>
  <p>
    &bull; <em>Model A (TF-IDF + Logistic Regression):</em> Sublinear TF-IDF word and character n-gram extraction (10,000 features, n-gram range 1&ndash;2) paired with independent Logistic Regression classifiers with balanced class weighting. Decision thresholds tuned on the validation split: &theta;<sub>req</sub> = 0.45 and &theta;<sub>com</sub> = 0.65.
  </p>
  <p>
    &bull; <em>Model B (Fine-Tuned DistilBERT):</em> Pretrained <code>distilbert-base-uncased</code> configured with a two-unit classification head trained for 5,000 steps with binary cross-entropy loss (<code>BCEWithLogitsLoss</code>). Optimization: AdamW (lr = 2 &times; 10<sup>-5</sup>, weight decay 0.01). Default decision thresholds: &theta;<sub>req</sub> = 0.50 and &theta;<sub>com</sub> = 0.50; validation-tuned thresholds: &theta;<sub>req</sub> = 0.35 and &theta;<sub>com</sub> = 0.20.
  </p>

  <div class="alg-container">
    <div class="alg-title">Algorithm 1: Overlapping Sliding-Window Inference</div>
    <div class="alg-body">Input: Text C, Tokenizer Tok, Model M, L=128, S=64, Thresholds (θ_req, θ_com)
Output: Scores (P_req, P_com), Preds (Y_req, Y_com), Windows W
1:  tokens, offsets &larr; Tok.encode(C)
2:  N &larr; len(tokens), W &larr; [], start &larr; 0, max_content &larr; L - 2
3:  while start &lt; N do
4:      end &larr; min(start + max_content, N)
5:      ids &larr; [CLS] + tokens[start:end] + [SEP]
6:      c_start &larr; offsets[start].start, c_end &larr; offsets[end-1].end
7:      probs &larr; Sigmoid(M.forward(ids))
8:      W.append({c_start, c_end, probs})
9:      if end == N then break end if
10:     start &larr; start + S
11: end while
12: P_req &larr; max({w.req for w in W}), P_com &larr; max({w.com for w in W})
13: Y_req &larr; I(P_req &ge; θ_req), Y_com &larr; I(P_com &ge; θ_com)
14: return P_req, P_com, Y_req, Y_com, W</div>
  </div>

  <h3 class="ieee-subsec">F. Experimental Parameters and Hyperparameters</h3>
  <p>
    To guarantee scientific reproducibility, all algorithmic parameters and training hyperparameters across both classification pipelines and downstream modules are formally documented:
  </p>
  <p>
    &bull; <em>Transformer Optimization Parameters:</em> DistilBERT fine-tuning utilized the AdamW optimizer with initial learning rate &eta; = 2 &times; 10<sup>-5</sup>, linear learning rate decay, weight decay &lambda; = 0.01, batch size <em>B</em> = 2, and 5,000 gradient update steps under mixed precision on Apple MPS acceleration.
  </p>
  <p>
    &bull; <em>Sliding-Window Parameters:</em> The sliding-window inference architecture operated with maximum window length <em>L</em> = 128 tokens (126 content tokens plus <code>[CLS]</code> and <code>[SEP]</code>) and stride <em>S</em> = 64 tokens, enforcing a 50% window overlap across long corporate messages.
  </p>
  <p>
    &bull; <em>Loss Reweighting &amp; Sampling Parameters:</em> For imbalanced recovery experiments, positive class weights in <code>BCEWithLogitsLoss</code> were configured to pos_weight = 20.0 for commitments and 1.2 for requests, combined with batch oversampling (4 commitment examples per batch of 16).
  </p>
  <p>
    &bull; <em>Decision Threshold Parameters:</em> Threshold sweeps over validation precision-recall curves established optimal operating points: baseline default &theta; = 0.50, and validation-tuned thresholds &theta;<sub>req</sub> = 0.40 and &theta;<sub>com</sub> = 0.85 for the reweighted model.
  </p>
  <p>
    &bull; <em>Bi-Encoder Semantic Parameters:</em> Follow-up linking utilized <code>all-MiniLM-L6-v2</code> generating 384-dimensional dense vectors, paired with a cosine similarity candidate retrieval threshold &tau;<sub>sim</sub> = 0.40.
  </p>

  <h2 class="ieee-sec">VI. Evaluation and Empirical Results</h2>
  <p>
    We evaluate FollowThrough across three empirical dimensions: weak-label full-corpus sequence classification, computational efficiency and resource footprint, and threshold/loss reweighting experiments. All evaluations were conducted on the identical held-out test split of <strong>9,657 messages</strong> isolated via disjoint-set graph partitioning.
  </p>

  <div class="table-wrapper">
    <div class="table-caption">TABLE II<br>COMPARATIVE BENCHMARK ON HELD-OUT TEST SPLIT (N = 9,657)</div>
    <table class="ieee-table">
      <thead>
        <tr>
          <th class="left-align">Metric</th>
          <th>TF-IDF</th>
          <th>DistilBERT</th>
          <th>Diff (&Delta;)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="left-align">Exact Match Accuracy</td>
          <td><strong>0.8801</strong></td>
          <td>0.7145</td>
          <td>-0.1656</td>
        </tr>
        <tr>
          <td class="left-align">Hamming Loss</td>
          <td><strong>0.0630</strong></td>
          <td>0.1479</td>
          <td>+0.0849</td>
        </tr>
        <tr>
          <td class="left-align">Macro-Averaged F1</td>
          <td><strong>0.6874</strong></td>
          <td>0.3656</td>
          <td>-0.3218</td>
        </tr>
        <tr>
          <td class="left-align">Micro-Averaged F1</td>
          <td><strong>0.8708</strong></td>
          <td>0.7027</td>
          <td>-0.1681</td>
        </tr>
        <tr>
          <td class="left-align">Request Precision</td>
          <td><strong>0.8996</strong></td>
          <td>0.6801</td>
          <td>-0.2195</td>
        </tr>
        <tr>
          <td class="left-align">Request Recall</td>
          <td><strong>0.9169</strong></td>
          <td>0.7906</td>
          <td>-0.1263</td>
        </tr>
        <tr>
          <td class="left-align">Request F1-Score</td>
          <td><strong>0.9081</strong></td>
          <td>0.7312</td>
          <td>-0.1769</td>
        </tr>
        <tr>
          <td class="left-align">Commitment Precision</td>
          <td><strong>0.4286</strong></td>
          <td>0.0000</td>
          <td>-0.4286</td>
        </tr>
        <tr>
          <td class="left-align">Commitment Recall</td>
          <td><strong>0.5124</strong></td>
          <td>0.0000</td>
          <td>-0.5124</td>
        </tr>
        <tr>
          <td class="left-align">Commitment F1-Score</td>
          <td><strong>0.4668</strong></td>
          <td>0.0000</td>
          <td>-0.4668</td>
        </tr>
        <tr>
          <td class="left-align">Test Inference Time</td>
          <td><strong>0.02 s</strong></td>
          <td>57.57 s</td>
          <td>2,878&times;</td>
        </tr>
        <tr>
          <td class="left-align">Persistent Model Size</td>
          <td><strong>0.52 MB</strong></td>
          <td>255.4 MB</td>
          <td>491&times;</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="table-wrapper">
    <div class="table-caption">TABLE III<br>DETAILED CONFUSION MATRICES ON TEST SPLIT (N = 9,657)</div>
    <table class="ieee-table">
      <thead>
        <tr>
          <th class="left-align">Target</th>
          <th>Model</th>
          <th>TP</th>
          <th>FP</th>
          <th>FN</th>
          <th>TN</th>
          <th>Support</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="left-align">Request</td>
          <td>TF-IDF</td>
          <td>3,915</td>
          <td>437</td>
          <td>355</td>
          <td>4,950</td>
          <td>4,270</td>
        </tr>
        <tr>
          <td class="left-align">Request</td>
          <td>DistilBERT</td>
          <td>3,376</td>
          <td>1,588</td>
          <td>894</td>
          <td>3,799</td>
          <td>4,270</td>
        </tr>
        <tr>
          <td class="left-align">Commitment</td>
          <td>TF-IDF</td>
          <td>186</td>
          <td>248</td>
          <td>177</td>
          <td>9,046</td>
          <td>363</td>
        </tr>
        <tr>
          <td class="left-align">Commitment</td>
          <td>DistilBERT</td>
          <td>0</td>
          <td>11</td>
          <td>363</td>
          <td>9,283</td>
          <td>363</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 class="ieee-subsec">A. Scientific Interpretation of Findings</h3>
  <p>
    <em>1) Surface Lexical Alignment on Weak Labels:</em> The linear TF-IDF baseline significantly outperforms fine-tuned DistilBERT on programmatic weak-label agreement across every metric (Macro-F1 0.6874 vs. 0.3656). This occurs because the programmatic pseudo-labels were generated via deterministic lexical patterns and keyword regular expressions. Because TF-IDF constructs sparse n-gram features directly over exact word unigrams and bigrams, the linear model directly memorizes the exact lexical triggers defining the pseudo-labels. In contrast, DistilBERT maps tokens into dense continuous representations, which may lead it to diverge from the rigid, literal boundary rules of the programmatic labeling functions.
  </p>
  <p>
    <em>2) Severe Class Imbalance and Neural Collapse:</em> In the Enron training corpus, commitments constitute only 4.08% of training samples (3,201 positives out of 78,385 messages). In standard DistilBERT training with unweighted <code>BCEWithLogitsLoss</code> over 5,000 steps, gradients from the 95.92% negative instances dominated optimization, causing complete collapse on the minority commitment target (0 TP). In contrast, the TF-IDF baseline utilized analytical balanced class weighting, scaling updates inversely proportional to frequencies to maintain commitment recall at 0.5124.
  </p>

  <h3 class="ieee-subsec">B. Evaluation of Combined Reweighting & Threshold Tuning</h3>
  <p>
    To investigate minority-class collapse, we conducted experiments combining class-weighted loss and batch oversampling in <code>scripts/train_weighted_distilbert.py</code>, followed by threshold sweeping in <code>scripts/tune_thresholds_experiment.py</code>. Specifically, we configured a commitment positive weight of pos_weight = 20.0 (and 1.2 for requests) within <code>BCEWithLogitsLoss</code>, combined with batch oversampling (4 commitment examples per batch of 16).
  </p>

  <div class="table-wrapper">
    <div class="table-caption">TABLE IV<br>DISTILBERT REWEIGHTING & THRESHOLD TUNING EXPERIMENTS (N = 9,657)</div>
    <table class="ieee-table ieee-table-compact">
      <thead>
        <tr>
          <th class="left-align">Configuration</th>
          <th>Req F1</th>
          <th>Com R</th>
          <th>Com P</th>
          <th>Com F1</th>
          <th>Macro</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="left-align">1. Default (&theta;=0.50)</td>
          <td>0.7312</td>
          <td>0.0000</td>
          <td>0.0000</td>
          <td>0.0000</td>
          <td>0.3656</td>
        </tr>
        <tr>
          <td class="left-align">2. Thresh-Only (&theta;<sub>com</sub>=0.20)</td>
          <td>0.7312</td>
          <td>0.0000</td>
          <td>0.0000</td>
          <td>0.0000</td>
          <td>0.3656</td>
        </tr>
        <tr>
          <td class="left-align">3. Combined (&theta;=0.50)</td>
          <td>0.7374</td>
          <td><strong>1.0000</strong></td>
          <td>0.0495</td>
          <td>0.0943</td>
          <td>0.4158</td>
        </tr>
        <tr>
          <td class="left-align">4. Comb.+Tuned (0.40, 0.85)</td>
          <td><strong>0.7603</strong></td>
          <td><strong>0.7879</strong></td>
          <td><strong>0.0955</strong></td>
          <td><strong>0.1703</strong></td>
          <td><strong>0.4653</strong></td>
        </tr>
      </tbody>
    </table>
  </div>

  <h3 class="ieee-subsec">C. Analysis of Experimental Progression</h3>
  <p>
    <em>1) Why Threshold Tuning Alone Failed (Run 2):</em> On the unweighted model, all predicted commitment probabilities on the test set were near zero. Sweeping the threshold on validation PR curves down to &theta;<sub>com</sub> = 0.20 yielded only 11 positive predictions on test, all 11 of which were false positives (0 TP, 11 FP, 363 FN). Consequently, precision, recall, and F1 remained at 0.0000. Threshold adjustment alone without loss or sampling intervention was fundamentally unable to overcome minority-class suppression.
  </p>
  <p>
    <em>2) Combined Intervention Recovery and Trade-Offs (Runs 3 and 4):</em> Applying class weighting (20.0&times;) and batch oversampling jointly forced the network to predict positive commitments, identifying all 363 test commitments at &theta;=0.50 (100% recall, 6,975 FP). When thresholds were subsequently tuned on validation PR curves (&theta;<sub>req</sub>=0.40, &theta;<sub>com</sub>=0.85), Run 4 achieved a commitment recall of <strong>78.79%</strong> (286 TP out of 363), raising Request F1 to 0.7603 and Macro-F1 to 0.4653.
  </p>
  <p>
    <em>3) Realistic Assessment of Precision:</em> Crucially, Run 4 produced 2,709 false positives on the test set against programmatic weak labels, yielding a commitment precision of <strong>9.55%</strong>. In practice, this corresponds to approximately one correct prediction for every ten commitment alerts. While this demonstrates that class reweighting can successfully recover recall on an imbalanced corpus, the severe recall&ndash;precision trade-off confirms that the classifier is not standalone production-grade. TF-IDF remains the superior and more balanced model on the weak-label benchmark.
  </p>

  <h2 class="ieee-sec">VII. Discussion</h2>
  <h3 class="ieee-subsec">A. Key Strengths and Systemic Principles</h3>
  <p>
    &bull; <em>Non-Gating Advisory Paradigm:</em> Decoupling classification from extraction prevents downstream error propagation. Because classification scores are treated as advisory indicators rather than hard filters, low classifier precision does not invalidate downstream candidate extraction.
  </p>
  <p>
    &bull; <em>Linguistic Extraction with Provenance:</em> Deterministic linguistic extraction anchored to deontic modals preserves exact character offsets ([<em>c</em><sub>start</sub>, <em>c</em><sub>end</sub>]), providing verifiable provenance and mitigating generative hallucination risks.
  </p>
  <p>
    &bull; <em>Cryptographic Auditability:</em> Content hashing prevents duplicate tasks, and the append-only <code>task_history</code> table maintains full operational accountability.
  </p>

  <h3 class="ieee-subsec">B. Limitations and Future Work</h3>
  <p>
    &bull; <em>Weak-Label Benchmark Limitations:</em> The evaluation relies on rule-derived pseudo-labels that inherently contain linguistic noise; human double-blind ground-truth annotation is required for unbiased real-world assessment.
  </p>
  <p>
    &bull; <em>Colloquial Commitments:</em> Conservative patterns miss indirect or colloquial undertakings ("Consider it done").
  </p>
  <p>
    &bull; <em>Temporal Pragmatics:</em> Relative dates ("next Friday") require contextual disambiguation against sender timezones.
  </p>
  <p>
    &bull; <em>Future Directions:</em> Transitioning to token-level sequence labeling (BIO NER tagging) and modeling conversational threads via Graph Neural Networks (GNNs).
  </p>

  <div class="table-wrapper">
    <div class="table-caption">TABLE V<br>OPERATIONAL RISKS AND MITIGATION STRATEGIES</div>
    <table class="ieee-table">
      <thead>
        <tr>
          <th class="left-align">Risk</th>
          <th class="left-align">System Impact</th>
          <th class="left-align">Mitigation Strategy</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="left-align">Premature Writes</td>
          <td class="left-align">Missed deliverables / false tickets</td>
          <td class="left-align">Mandatory Human Gate; zero unverified commits.</td>
        </tr>
        <tr>
          <td class="left-align">Privacy Egress</td>
          <td class="left-align">Corporate NDA / GDPR violations</td>
          <td class="left-align">Local embedded execution; zero 3P API calls.</td>
        </tr>
        <tr>
          <td class="left-align">Database Lock</td>
          <td class="left-align">Streamlit concurrency failure</td>
          <td class="left-align">SQLite Write-Ahead Logging (WAL) enabled.</td>
        </tr>
        <tr>
          <td class="left-align">Boundary Loss</td>
          <td class="left-align">Lost tasks in long messages</td>
          <td class="left-align">Overlapping 128-tok windows with 64-tok stride.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h2 class="ieee-sec">VIII. Conclusion</h2>
  <p>
    FollowThrough demonstrates that practical conversational NLP systems must extend beyond isolated sentence classification. By integrating sequence classification, conservative linguistic extraction with exact evidence offsets, bi-encoder semantic linking, and persistent state ledgering within an interactive human-in-the-loop application, FollowThrough provides a bounded, verifiable solution to operational commitment tracking in email and workplace messaging.
  </p>

  <h2 class="ieee-sec">Reproducibility & Artifact Lineage</h2>
  <p class="no-indent">
    All code, models, and evaluation artifacts are organized within the repository structure:
  </p>
  <p class="no-indent" style="line-height: 1.25;">
    &bull; <em>Training Scripts:</em> <code>scripts/<wbr>train_full_<wbr>distilbert.py</code> (Run 1) and <code>scripts/<wbr>train_weighted_<wbr>distilbert.py</code> (Runs 3&ndash;4).<br>
    &bull; <em>Threshold Tuning:</em> <code>scripts/<wbr>tune_thresholds_<wbr>experiment.py</code>.<br>
    &bull; <em>Results Artifacts:</em> <code>evaluation/<wbr>results/<wbr>weighted_experiment_<wbr>results.json</code> and <code>distilbert_<wbr>val_test_probs.npz</code>.<br>
    &bull; <em>Model Checkpoints:</em> Saved under <code>models/<wbr>distilbert_fine_tuned/</code> and <code>models/<wbr>distilbert_weighted/</code>.
  </p>

  <h2 class="ieee-sec">References</h2>
  <div class="references-container">
    <div class="ref-item">[1] B. Klimt and Y. Yang, "The Enron Corpus: A New Dataset for Email Classification Research," in <em>Proc. 15th ECML</em>, Pisa, Italy, 2004, pp. 217&ndash;226.</div>
    <div class="ref-item">[2] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," <em>arXiv:1910.01108</em>, 2019.</div>
    <div class="ref-item">[3] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in <em>Proc. EMNLP</em>, Hong Kong, 2019, pp. 3982&ndash;3992.</div>
    <div class="ref-item">[4] A. Ratner, C. M. De Sa, S. Wu, D. Selsam, and C. R&eacute;, "Snorkel: Rapid Training Data Creation with Weak Supervision," <em>Proc. VLDB</em>, vol. 11, no. 3, pp. 269&ndash;282, 2017.</div>
    <div class="ref-item">[5] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in <em>Proc. NAACL-HLT</em>, Minneapolis, MN, 2019, pp. 4171&ndash;4186.</div>
    <div class="ref-item">[6] J. R. Searle, <em>Speech Acts: An Essay in the Philosophy of Language</em>. Cambridge, UK: Cambridge Univ. Press, 1969.</div>
    <div class="ref-item">[7] J. L. Austin, <em>How to Do Things with Words</em>. Oxford, UK: Oxford Univ. Press, 1962.</div>
    <div class="ref-item">[8] V. R. Carvalho and W. W. Cohen, "On the collective classification of email speech acts," in <em>Proc. 28th ACM SIGIR</em>, Salvador, Brazil, 2005, pp. 345&ndash;352.</div>
    <div class="ref-item">[9] P. S. Turney, "Thumbs up or thumbs down? Semantic orientation applied to unsupervised classification of reviews," in <em>Proc. 40th ACL</em>, Philadelphia, PA, 2002, pp. 417&ndash;424.</div>
    <div class="ref-item">[10] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," <em>J. Mach. Learn. Res.</em>, vol. 12, pp. 2825&ndash;2830, 2011.</div>
    <div class="ref-item">[11] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in <em>NeurIPS 32</em>, Vancouver, Canada, 2019, pp. 8024&ndash;8035.</div>
    <div class="ref-item">[12] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in <em>ICLR</em>, New Orleans, LA, 2019.</div>
    <div class="ref-item">[13] T. Mikolov, K. Chen, G. S. Corrado, and J. Dean, "Efficient Estimation of Word Representations in Vector Space," in <em>Workshop at ICLR</em>, Scottsdale, AZ, 2013.</div>
    <div class="ref-item">[14] D. Corney, D. R. Stent, and H. Di Fabbrizio, "What are you going to do? Detecting task-oriented emails," in <em>Proc. 37th ACM SIGIR</em>, Gold Coast, Australia, 2014, pp. 1007&ndash;1010.</div>
    <div class="ref-item">[15] <em>RFC 822: Standard for the Format of ARPA Internet Text Messages</em>, Internet Engineering Task Force (IETF), Aug. 1982.</div>
  </div>

</div>

</body>
</html>
"""

def generate():
    print(f"Writing corrected IEEE HTML template to {HTML_OUT}...")
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(IEEE_HTML)

    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    user_data_dir = "/tmp/chrome_ieee_pdf_userdata"
    
    cmd = [
        chrome_path,
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        f"--user-data-dir={user_data_dir}",
        f"--print-to-pdf={PDF_OUT}",
        "--no-pdf-header-footer",
        str(HTML_OUT)
    ]
    
    print(f"Compiling Chrome Headless IEEE PDF generation to {PDF_OUT}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    if PDF_OUT.exists():
        size_kb = PDF_OUT.stat().st_size / 1024
        print(f"SUCCESS: IEEE PDF generated at {PDF_OUT} ({size_kb:.1f} KB)")
        shutil.copyfile(PDF_OUT, PDF_ALT)
        print(f"SUCCESS: Copied to {PDF_ALT}")
    else:
        print(f"ERROR: PDF generation failed. Stderr:\n{res.stderr}")

if __name__ == "__main__":
    generate()
