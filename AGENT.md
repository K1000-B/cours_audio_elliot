# AGENTS.md — Whisper Transcriber

## 1. Purpose of the Project

This project is a **local transcription and structuring pipeline** designed to convert recorded lectures (Webex `.m3u8`) into **high-quality structured course material**, ready to be transformed into LaTeX.

The goal is not just transcription, but:
- extracting usable academic content
- structuring it for engineering-level study
- enabling downstream automation (LaTeX, PDF, revision sheets)

The system is designed for:
- reproducibility
- local execution (no API dependency)
- integration into technical workflows

---

## 2. What Has Been Implemented

### 2.1 Full Pipeline

The pipeline is fully operational and automated:

```

URL (.m3u8)
↓
ffmpeg → audio extraction (wav, 16kHz, mono)
↓
whisper.cpp → transcription
↓
post-processing → structured prompt

```

---

### 2.2 GUI Client

A Python Tkinter interface has been implemented with:

- URL input
- project name input
- language selection (input/output)
- execution button
- status feedback

The GUI is intentionally simple and functional.

---

### 2.3 Audio Processing

- ffmpeg is used to:
  - stream Webex recordings
  - extract audio only
  - normalize format:
    - mono
    - 16 kHz
    - PCM (s16le)

This format is required for optimal Whisper performance.

---

### 2.4 Transcription Engine

- `whisper.cpp` is used (C++ implementation)
- running locally with Metal acceleration (Mac M-series)
- model: `medium` or `small`
    > change 
    
    ```MODEL_PATH = os.path.expanduser("~/dev/ai/whisper-cpp/models/ggml-small.bin")```

    > by 

    ```MODEL_PATH = os.path.expanduser("~/dev/ai/whisper-cpp/models/ggml-medium.bin")```

Options supported:
- forced language (`-l`)
- translation to English (`--translate`)

---

### 2.5 Output Structure

Each run creates a dedicated folder:

```

project_name/
├── audio.wav
├── raw.txt
└── formatted.txt

````

- `raw.txt` → direct transcription
- `formatted.txt` → prompt-ready for LLM

---

### 2.6 LLM Preparation Layer

The system generates a structured prompt that allows an AI to:

- rewrite the content
- structure it into sections
- extract definitions, equations, key ideas
- produce LaTeX-ready material

This layer is critical: it bridges raw ASR → usable course.

---

## 3. Design Choices

### 3.1 Local-first

- no external API
- full control over data
- deterministic pipeline

---

### 3.2 Separation of Concerns

| Component | Responsibility |
|----------|--------------|
| ffmpeg | signal processing |
| whisper.cpp | transcription |
| Python | orchestration |
| LLM | structuring |

---

### 3.3 Minimal UI

The GUI is not a product:
- it is a thin execution layer
- logic must remain outside UI

---

## 4. What MUST NOT Be Modified

This section is critical.

Any AI modifying this project must strictly respect the following constraints.

---

### 4.1 Core Pipeline Integrity

DO NOT:
- change the order of the pipeline
- merge steps (e.g., transcription + formatting)
- remove intermediate files (`audio.wav`, `raw.txt`)

These are required for:
- debugging
- reproducibility
- modularity

---

### 4.2 Audio Format

DO NOT modify:

```bash
-ac 1 -ar 16000 -c:a pcm_s16le
````

Reason:

* Whisper is optimized for this format
* changing it degrades performance

---

### 4.3 whisper.cpp Usage

DO NOT:

* replace with Python Whisper
* introduce external APIs
* change model path logic

This project is intentionally:

* local
* lightweight
* hardware-accelerated (Metal)

---

### 4.4 File System Structure

DO NOT:

* change folder hierarchy
* store outputs elsewhere
* remove deterministic naming

The structure:

```
project/
├── audio.wav
├── raw.txt
└── formatted.txt
```

is a **contract**.

---

### 4.5 Separation UI / Logic

DO NOT:

* embed core logic inside UI callbacks beyond orchestration
* tightly couple UI and pipeline

The pipeline must remain callable independently.

---

### 4.6 LLM Prompt Philosophy

DO NOT:

* simplify the prompt excessively
* remove constraints like:

  * structure (sections)
  * equations
  * academic tone

The goal is **engineering-grade output**, not summarization.

---

### 4.7 Language Handling

DO NOT:

* assume Whisper supports all translations
* remove LLM fallback for non-English output

Correct logic:

* Whisper handles transcription
* LLM handles semantic transformation

---

## 5. What Can Be Modified

Allowed improvements:

### UI

* better design (CustomTkinter, etc.)
* progress bar
* logs

### Pipeline robustness

* retry logic
* chunking long audio
* threading

### Output

* `.tex` generation
* PDF compilation

### Performance

* batching
* parallelization

---

## 6. Non-Goals

This project is NOT:

* a SaaS product
* a general transcription tool
* a real-time system

It is:

* a **personal engineering tool**
* focused on **course generation**

---

## 7. Summary

This project implements a **deterministic, modular, local pipeline**:

* Input: recorded lecture
* Output: structured, LLM-ready content

Key principles:

* do not break the pipeline
* do not remove intermediate steps
* do not externalize processing
* preserve engineering-level rigor

---


