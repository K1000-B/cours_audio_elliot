import os
import subprocess
import threading
import queue
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from pathlib import Path
from urllib.parse import urlparse

# ===== CONFIG =====
BASE_DIR = Path(__file__).parent.resolve()

WHISPER_PATH = os.path.expanduser("~/dev/ai/whisper-cpp/build/bin/whisper-cli")
WHISPER_MODELS_DIR = os.path.expanduser("~/dev/ai/whisper-cpp/models")
CPU_COUNT = max(1, os.cpu_count() or 1)
CHUNK_DURATION_SECONDS = 3 * 60
MAX_PARALLEL_WORKERS = max(1, min(8, CPU_COUNT))
DEFAULT_PARALLEL_WORKERS = max(1, min(4, MAX_PARALLEL_WORKERS))

LANGUAGES = ["auto", "fr", "en", "it", "es", "de"]
WHISPER_MODELS = ["base", "small", "medium", "large-v3"]
EXECUTION_MODES = ["Série", "Parallèle"]
SOURCE_MODES = ["URL Webex", "Fichier local"]
LANGUAGE_CHOICES = [
    ("Auto-detect", "auto"),
    ("Français", "fr"),
    ("English", "en"),
    ("Italiano", "it"),
    ("Español", "es"),
    ("Deutsch", "de"),
]

UI_COLORS = {
    "background": "#0D174E",
    "surface": "#16235F",
    "panel": "#1D2E72",
    "primary_button": "#9A560F",
    "primary_button_active": "#8A4C0E",
    "text": "#FFFFFF",
    "text_secondary": "#C9D4EE",
    "success": "#7DD3A7",
    "error": "#FFB4B4",
    "field": "#0F1A55",
    "field_border": "#314693",
}

# ===== CORE =====

def run_cmd(cmd, log=None):
    if log:
        log(f"$ {cmd}")

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,
    )

    assert process.stdout is not None
    buffer = ""
    while True:
        chunk = process.stdout.read(1)
        if chunk == "" and process.poll() is not None:
            break
        if not chunk:
            continue
        if chunk in ("\n", "\r"):
            if buffer and log:
                log(buffer)
            buffer = ""
            continue
        buffer += chunk

    if buffer and log:
        log(buffer)

    process.stdout.close()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed:\n{cmd}")

def is_probable_url(value):
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def extract_audio_to_wav(source, output_audio, log=None):
    source_path = Path(source).expanduser()
    if not is_probable_url(source) and not source_path.is_file():
        raise FileNotFoundError(f"Fichier source introuvable: {source_path}")

    cmd = f'ffmpeg -y -i "{source}" -vn -map 0:a:0 -c:a pcm_s16le "{output_audio}"'
    run_cmd(cmd, log=log)


def prepare_intermediate_dir(dir_path, log=None):
    dir_path = Path(dir_path)
    if dir_path.exists():
        shutil.rmtree(dir_path)
        if log:
            log(f"Réinitialisation du dossier intermédiaire: {dir_path}")
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def split_audio_into_chunks(audio_path, chunks_dir, log=None, chunk_duration=CHUNK_DURATION_SECONDS):
    audio_path = Path(audio_path)
    chunks_dir = prepare_intermediate_dir(chunks_dir, log=log)
    chunk_pattern = chunks_dir / "chunk_%03d.wav"

    if log:
        log(
            f"Découpage audio en segments de {chunk_duration // 60} minutes max vers {chunks_dir}"
        )

    cmd = (
        f'ffmpeg -y -i "{audio_path}" -f segment -segment_time {chunk_duration} '
        f'-reset_timestamps 1 -map 0:a:0 -c:a copy "{chunk_pattern}"'
    )
    run_cmd(cmd, log=log)

    chunks = sorted(chunks_dir.glob("chunk_*.wav"))
    if not chunks:
        raise FileNotFoundError("Aucun segment audio généré après découpage.")

    if log:
        log(f"{len(chunks)} segment(s) audio généré(s).")

    return chunks


def resolve_whisper_model_path(model_name):
    return os.path.join(WHISPER_MODELS_DIR, f"ggml-{model_name}.bin")


def transcribe(audio_path, output_txt, lang_in, lang_out, model_name, whisper_threads, log=None):
    audio_path = Path(audio_path)
    output_txt = Path(output_txt)
    model_path = resolve_whisper_model_path(model_name)

    lang_flag = "" if lang_in == "auto" else f"-l {lang_in}"

    translate_flag = ""
    if lang_out == "en" and lang_in != "en":
        translate_flag = "--translate"

    # whisper-cli output naming may vary by version:
    # - audio.txt
    # - audio.wav.txt
    # - audio.text
    # - audio.wav.text
    candidates = [
        audio_path.with_suffix(".txt"),
        Path(str(audio_path) + ".txt"),
        audio_path.with_suffix(".text"),
        Path(str(audio_path) + ".text"),
    ]

    for candidate in candidates:
        if candidate.exists():
            candidate.unlink()

    cmd_parts = [
        f'"{WHISPER_PATH}"',
        f'-m "{model_path}"',
        f'-f "{audio_path}"',
        lang_flag,
        translate_flag,
        f"-t {whisper_threads}",
        "-otxt",
    ]
    run_cmd(" ".join(part for part in cmd_parts if part), log=log)

    generated_txt = next((p for p in candidates if p.exists()), None)
    if generated_txt is None:
        expected = "\n".join(f"- {p}" for p in candidates)
        raise FileNotFoundError(
            f"Whisper output file not found. Checked:\n{expected}"
        )

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    if output_txt.exists():
        output_txt.unlink()
    generated_txt.rename(output_txt)


def concat_transcriptions(parts, output_txt, log=None):
    # The caller provides parts in chunk index order; keep that exact order.
    ordered_parts = [Path(part) for part in parts]
    merged_parts = []

    for part in ordered_parts:
        with open(part, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            merged_parts.append(content)
        elif log:
            log(f"Segment vide ignoré dans la concaténation: {part.name}")

    raw_content = "\n\n".join(merged_parts)
    if raw_content:
        raw_content += "\n"

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(raw_content)

    if log:
        log(f"Concaténation terminée vers {output_txt} ({len(ordered_parts)} partie(s)).")


def format_for_llm(input_txt, output_txt, lang_out):
    with open(input_txt, "r", encoding="utf-8") as f:
        text = f.read()

    formatted = f"""
# TRANSCRIPTION BRUTE

{text}

# INSTRUCTIONS POUR IA

Rewrite this into a structured LaTeX course.

Constraints:
- Output language: {lang_out}
- Academic style
- Extract definitions, equations, key ideas
- Structure with \\section, \\subsection
"""

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(formatted)

# ===== UI =====

class App:
    def __init__(self, root):
        self.root = root
        self.log_queue = queue.Queue()
        root.title("Whisper Transcriber")
        root.geometry("1280x780")
        root.minsize(1040, 700)
        root.configure(bg=UI_COLORS["background"])

        self._build_styles()

        workspace = tk.Frame(root, bg=UI_COLORS["background"])
        workspace.pack(fill="both", expand=True, padx=22, pady=22)
        workspace.grid_columnconfigure(0, weight=0, minsize=440)
        workspace.grid_columnconfigure(1, weight=1)
        workspace.grid_rowconfigure(0, weight=1)

        self.controls_panel = tk.Frame(
            workspace,
            bg=UI_COLORS["surface"],
            highlightthickness=1,
            highlightbackground=UI_COLORS["field_border"],
        )
        self.controls_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        self.controls_panel.grid_columnconfigure(0, weight=1)
        self.controls_panel.grid_rowconfigure(0, weight=1)

        self.controls_canvas = tk.Canvas(
            self.controls_panel,
            bg=UI_COLORS["surface"],
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self.controls_scrollbar = ttk.Scrollbar(
            self.controls_panel,
            orient="vertical",
            command=self.controls_canvas.yview,
        )
        self.controls_canvas.configure(yscrollcommand=self.controls_scrollbar.set)
        self.controls_canvas.grid(row=0, column=0, sticky="nsew")
        self.controls_scrollbar.grid(row=0, column=1, sticky="ns")

        self.controls_content = tk.Frame(self.controls_canvas, bg=UI_COLORS["surface"])
        self.controls_canvas_window = self.controls_canvas.create_window(
            (0, 0),
            window=self.controls_content,
            anchor="nw",
        )
        self.controls_content.bind("<Configure>", self._update_controls_scrollregion)
        self.controls_canvas.bind("<Configure>", self._resize_controls_content)
        self.controls_panel.bind("<Enter>", self._bind_controls_mousewheel)
        self.controls_panel.bind("<Leave>", self._unbind_controls_mousewheel)

        header = tk.Frame(self.controls_content, bg=UI_COLORS["surface"])
        header.pack(fill="x", padx=22, pady=(22, 16))

        tk.Label(
            header,
            text="Whisper Transcriber",
            fg=UI_COLORS["text"],
            bg=UI_COLORS["surface"],
            font=("Helvetica", 24, "bold"),
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Extraction Webex, transcription whisper.cpp et formatage prêt pour LLM dans une interface claire et compacte.",
            fg=UI_COLORS["text_secondary"],
            bg=UI_COLORS["surface"],
            font=("Helvetica", 11),
            justify="left",
            wraplength=360,
        ).pack(anchor="w", pady=(6, 0))

        self.lang_input = tk.StringVar(value="Auto-detect")
        self.lang_output = tk.StringVar(value="Français")
        self.whisper_model = tk.StringVar(value="base")
        self.execution_mode = tk.StringVar(value=EXECUTION_MODES[0])
        self.max_workers = tk.IntVar(value=DEFAULT_PARALLEL_WORKERS)
        self.source_mode = tk.StringVar(value=SOURCE_MODES[0])
        self.local_file_path = tk.StringVar(value="")

        form = tk.Frame(self.controls_content, bg=UI_COLORS["surface"])
        form.pack(fill="x", padx=22, pady=(0, 16))
        form.grid_columnconfigure(0, weight=1)
        form.grid_columnconfigure(1, weight=1)

        source_card = self._card_frame(form)
        source_card.grid(row=0, column=0, columnspan=2, sticky="ew")
        source_body = tk.Frame(source_card, bg=UI_COLORS["surface"])
        source_body.grid(row=0, column=0, sticky="nsew", padx=20, pady=18)
        source_card.grid_columnconfigure(0, weight=1)

        tk.Label(
            source_body,
            text="Source",
            fg=UI_COLORS["text"],
            bg=UI_COLORS["surface"],
            font=("Helvetica", 13, "bold"),
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            source_body,
            text="Type de source",
            fg=UI_COLORS["text_secondary"],
            bg=UI_COLORS["surface"],
            font=("Helvetica", 10, "bold"),
        ).grid(row=1, column=0, sticky="w", pady=(14, 6))

        source_mode_shell = self._field_shell(source_body)
        source_mode_shell.grid(row=2, column=0, sticky="ew")
        self.source_mode_selector = ttk.Combobox(
            source_mode_shell,
            textvariable=self.source_mode,
            values=SOURCE_MODES,
            state="readonly",
            style="App.TCombobox",
        )
        self.source_mode_selector.pack(fill="x", padx=1, pady=1)
        self.source_mode_selector.current(0)
        self.source_mode.trace_add("write", self._update_source_inputs_state)

        tk.Label(
            source_body,
            text="URL Webex (.m3u8) ou URL média",
            fg=UI_COLORS["text_secondary"],
            bg=UI_COLORS["surface"],
            font=("Helvetica", 10, "bold"),
        ).grid(row=3, column=0, sticky="w", pady=(14, 6))

        url_shell = self._field_shell(source_body)
        url_shell.grid(row=4, column=0, sticky="ew")
        self.url_entry = ttk.Entry(url_shell, style="App.TEntry")
        self.url_entry.pack(fill="x", padx=1, pady=1)

        tk.Label(
            source_body,
            text="Fichier audio local (mp3, wav, m4a, flac, ogg, aac, ...)",
            fg=UI_COLORS["text_secondary"],
            bg=UI_COLORS["surface"],
            font=("Helvetica", 10, "bold"),
        ).grid(row=5, column=0, sticky="w", pady=(14, 6))

        local_shell = self._field_shell(source_body)
        local_shell.grid(row=6, column=0, sticky="ew")
        local_shell.grid_columnconfigure(0, weight=1)
        self.local_file_entry = ttk.Entry(
            local_shell,
            style="App.TEntry",
            textvariable=self.local_file_path,
        )
        self.local_file_entry.grid(row=0, column=0, sticky="ew", padx=(1, 6), pady=1)
        self.local_file_button = tk.Button(
            local_shell,
            text="Parcourir...",
            bg=UI_COLORS["field_border"],
            fg=UI_COLORS["text"],
            activebackground=UI_COLORS["panel"],
            activeforeground=UI_COLORS["text"],
            relief="flat",
            padx=10,
            pady=7,
            command=self.select_local_file,
            cursor="hand2",
        )
        self.local_file_button.grid(row=0, column=1, sticky="e", padx=(0, 1), pady=1)

        tk.Label(
            source_body,
            text="Nom du dossier projet",
            fg=UI_COLORS["text_secondary"],
            bg=UI_COLORS["surface"],
            font=("Helvetica", 10, "bold"),
        ).grid(row=7, column=0, sticky="w", pady=(14, 6))

        name_shell = self._field_shell(source_body)
        name_shell.grid(row=8, column=0, sticky="ew")
        self.name_entry = ttk.Entry(name_shell, style="App.TEntry")
        self.name_entry.pack(fill="x", padx=1, pady=1)
        self._update_source_inputs_state()

        transcription_card = self._card_frame(form)
        transcription_card.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        transcription_body = tk.Frame(transcription_card, bg=UI_COLORS["surface"])
        transcription_body.grid(row=0, column=0, sticky="nsew", padx=20, pady=18)
        transcription_card.grid_columnconfigure(0, weight=1)
        transcription_body.grid_columnconfigure(0, weight=1)
        transcription_body.grid_columnconfigure(1, weight=1)

        tk.Label(
            transcription_body,
            text="Paramètres de transcription",
            fg=UI_COLORS["text"],
            bg=UI_COLORS["surface"],
            font=("Helvetica", 13, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        self._language_selector(transcription_body, row=1, col=0, title="Langue audio", variable=self.lang_input, padx=(0, 10))
        self._language_selector(transcription_body, row=1, col=1, title="Langue sortie", variable=self.lang_output, padx=(10, 0))

        tk.Label(
            transcription_body,
            text="Modèle Whisper",
            fg=UI_COLORS["text_secondary"],
            bg=UI_COLORS["surface"],
            font=("Helvetica", 10, "bold"),
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(14, 6))

        model_shell = self._field_shell(transcription_body)
        model_shell.grid(row=4, column=0, columnspan=2, sticky="ew")
        self.model_selector = ttk.Combobox(
            model_shell,
            textvariable=self.whisper_model,
            values=WHISPER_MODELS,
            state="readonly",
            style="App.TCombobox",
        )
        self.model_selector.pack(fill="x", padx=1, pady=1)
        self.model_selector.current(0)

        tk.Label(
            transcription_body,
            text="Mode d'exécution",
            fg=UI_COLORS["text_secondary"],
            bg=UI_COLORS["surface"],
            font=("Helvetica", 10, "bold"),
        ).grid(row=5, column=0, sticky="w", pady=(14, 6))

        mode_shell = self._field_shell(transcription_body)
        mode_shell.grid(row=6, column=0, sticky="ew", padx=(0, 10))
        self.mode_selector = ttk.Combobox(
            mode_shell,
            textvariable=self.execution_mode,
            values=EXECUTION_MODES,
            state="readonly",
            style="App.TCombobox",
        )
        self.mode_selector.pack(fill="x", padx=1, pady=1)
        self.mode_selector.current(0)

        tk.Label(
            transcription_body,
            text=f"Workers max (1-{MAX_PARALLEL_WORKERS})",
            fg=UI_COLORS["text_secondary"],
            bg=UI_COLORS["surface"],
            font=("Helvetica", 10, "bold"),
        ).grid(row=5, column=1, sticky="w", padx=(10, 0), pady=(14, 6))

        worker_shell = self._field_shell(transcription_body)
        worker_shell.grid(row=6, column=1, sticky="ew", padx=(10, 0))
        self.worker_selector = ttk.Combobox(
            worker_shell,
            textvariable=self.max_workers,
            values=list(range(1, MAX_PARALLEL_WORKERS + 1)),
            state="readonly",
            style="App.TCombobox",
        )
        self.worker_selector.pack(fill="x", padx=1, pady=1)
        self.worker_selector.set(str(DEFAULT_PARALLEL_WORKERS))
        self.execution_mode.trace_add("write", self._update_worker_selector_state)
        self._update_worker_selector_state()

        footer = tk.Frame(self.controls_content, bg=UI_COLORS["surface"])
        footer.pack(fill="x", padx=22, pady=(0, 22))

        self.run_button = tk.Button(
            footer,
            text="Lancer le pipeline",
            bg=UI_COLORS["primary_button"],
            fg=UI_COLORS["text"],
            activebackground=UI_COLORS["primary_button_active"],
            activeforeground=UI_COLORS["text"],
            font=("Helvetica", 12, "bold"),
            relief="flat",
            padx=18,
            pady=12,
            command=self.run_pipeline,
            cursor="hand2",
        )
        self.run_button.pack(fill="x")

        self.status = tk.Label(
            footer,
            text="Prêt",
            fg=UI_COLORS["success"],
            bg=UI_COLORS["surface"],
            font=("Helvetica", 10, "bold"),
            anchor="w",
        )
        self.status.pack(fill="x", pady=(10, 0))

        self.logs_panel = tk.Frame(
            workspace,
            bg=UI_COLORS["surface"],
            highlightthickness=1,
            highlightbackground=UI_COLORS["field_border"],
        )
        self.logs_panel.grid(row=0, column=1, sticky="nsew")
        self.logs_panel.grid_columnconfigure(0, weight=1)
        self.logs_panel.grid_rowconfigure(1, weight=1)

        tk.Label(
            self.logs_panel,
            text="Logs runtime",
            fg=UI_COLORS["text"],
            bg=UI_COLORS["surface"],
            font=("Helvetica", 13, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=22, pady=(22, 10))

        self.log_text = scrolledtext.ScrolledText(
            self.logs_panel,
            wrap="word",
            bg=UI_COLORS["panel"],
            fg=UI_COLORS["text"],
            insertbackground=UI_COLORS["text"],
            relief="flat",
            padx=14,
            pady=10,
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=22, pady=(0, 22))
        self.log_text.configure(state="disabled")

        self.root.after(100, self.process_log_queue)

    def _build_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "App.TEntry",
            padding=(10, 9),
            fieldbackground=UI_COLORS["field"],
            background=UI_COLORS["field"],
            foreground=UI_COLORS["text"],
            insertcolor=UI_COLORS["text"],
            relief="flat",
            borderwidth=0,
            font=("Helvetica", 11),
        )
        style.map(
            "App.TEntry",
            fieldbackground=[("focus", UI_COLORS["field"]), ("active", UI_COLORS["field"])],
            foreground=[("disabled", UI_COLORS["text_secondary"])],
        )
        style.configure(
            "App.TCombobox",
            padding=(10, 9),
            fieldbackground=UI_COLORS["field"],
            background=UI_COLORS["field"],
            foreground=UI_COLORS["text"],
            arrowcolor=UI_COLORS["text_secondary"],
            relief="flat",
            borderwidth=0,
            font=("Helvetica", 11),
        )
        style.map(
            "App.TCombobox",
            fieldbackground=[
                ("readonly", UI_COLORS["field"]),
                ("focus", UI_COLORS["field"]),
                ("active", UI_COLORS["field"]),
            ],
            background=[("readonly", UI_COLORS["field"]), ("focus", UI_COLORS["field"])],
            foreground=[("readonly", UI_COLORS["text"])],
        )

    def _card_frame(self, parent):
        card = tk.Frame(
            parent,
            bg=UI_COLORS["surface"],
            highlightthickness=1,
            highlightbackground=UI_COLORS["field_border"],
        )
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(0, weight=0)
        card.grid_rowconfigure(1, weight=0)
        return card

    def _field_shell(self, parent):
        shell = tk.Frame(
            parent,
            bg=UI_COLORS["field"],
            highlightthickness=1,
            highlightbackground=UI_COLORS["field_border"],
        )
        shell.grid_columnconfigure(0, weight=1)
        return shell

    def _language_selector(self, parent, row, col, title, variable, padx):
        tk.Label(
            parent,
            text=title,
            fg=UI_COLORS["text_secondary"],
            bg=UI_COLORS["surface"],
            font=("Helvetica", 10, "bold"),
        ).grid(row=row, column=col, sticky="w", padx=padx)

        shell = self._field_shell(parent)
        shell.grid(row=row + 1, column=col, sticky="ew", padx=padx, pady=(6, 0))

        selector = ttk.Combobox(
            shell,
            textvariable=variable,
            values=[label for label, _code in LANGUAGE_CHOICES],
            state="readonly",
            style="App.TCombobox",
        )
        selector.pack(fill="x", padx=1, pady=1)

    def _language_code(self, value, fallback):
        for label, code in LANGUAGE_CHOICES:
            if value == label or value == code:
                return code
        return fallback

    def _update_controls_scrollregion(self, _event):
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _resize_controls_content(self, event):
        self.controls_canvas.itemconfigure(self.controls_canvas_window, width=event.width)

    def _bind_controls_mousewheel(self, _event):
        self.root.bind_all("<MouseWheel>", self._on_controls_mousewheel)
        self.root.bind_all("<Button-4>", self._on_controls_mousewheel)
        self.root.bind_all("<Button-5>", self._on_controls_mousewheel)

    def _unbind_controls_mousewheel(self, _event):
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _on_controls_mousewheel(self, event):
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        elif event.delta:
            delta = -1 if event.delta > 0 else 1
        else:
            delta = 0

        if delta != 0:
            self.controls_canvas.yview_scroll(delta, "units")

    def _update_worker_selector_state(self, *_args):
        state = "readonly" if self.execution_mode.get() == "Parallèle" else "disabled"
        self.worker_selector.configure(state=state)

    def _update_source_inputs_state(self, *_args):
        use_url = self.source_mode.get() == "URL Webex"
        self.url_entry.configure(state="normal" if use_url else "disabled")
        self.local_file_entry.configure(state="disabled" if use_url else "normal")
        self.local_file_button.configure(state="disabled" if use_url else "normal")

    def select_local_file(self):
        selected = filedialog.askopenfilename(
            title="Sélectionner un fichier audio",
            filetypes=[
                ("Audio", "*.mp3 *.wav *.m4a *.flac *.ogg *.oga *.opus *.aac *.wma *.aiff *.aif *.alac *.amr *.mp4 *.mkv *.webm *.mov"),
                ("Tous les fichiers", "*.*"),
            ],
        )
        if selected:
            self.local_file_path.set(selected)

    def set_running(self, running):
        state = "disabled" if running else "normal"
        self.root.after(0, lambda: self.run_button.config(state=state))

    def update_status(self, text, color=None):
        def _apply():
            self.status.config(text=text)
            if color:
                self.status.config(fg=color)
        self.root.after(0, _apply)

    def append_log(self, text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {text}")

    def process_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.log_text.configure(state="normal")
                self.log_text.insert("end", line + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass

        self.root.after(100, self.process_log_queue)

    def _transcribe_chunk(
        self,
        chunk_index,
        total_chunks,
        chunk_path,
        raw_parts_dir,
        lang_in,
        lang_out,
        whisper_model,
        whisper_threads,
    ):
        part_path = raw_parts_dir / f"{chunk_path.stem}.txt"
        chunk_tag = f"[Segment {chunk_index + 1}/{total_chunks}]"

        def chunk_log(message):
            self.append_log(f"{chunk_tag} {message}")

        chunk_log(f"Début transcription: {chunk_path.name} -> {part_path.name}")
        transcribe(
            chunk_path,
            part_path,
            lang_in,
            lang_out,
            whisper_model,
            whisper_threads=whisper_threads,
            log=chunk_log,
        )
        chunk_log("Transcription terminée")
        return chunk_index, part_path

    def run_pipeline_worker(
        self,
        source_mode,
        source_input,
        name,
        lang_in,
        lang_out,
        whisper_model,
        execution_mode,
        max_workers,
    ):
        try:
            project_dir = BASE_DIR / name
            os.makedirs(project_dir, exist_ok=True)

            audio_path = project_dir / "audio.wav"
            raw_txt = project_dir / "raw.txt"
            final_txt = project_dir / "formatted.txt"
            chunks_dir = project_dir / "chunks"
            raw_parts_dir = project_dir / "raw_parts"
            model_path = resolve_whisper_model_path(whisper_model)

            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Modèle introuvable: {model_path}")

            self.append_log(f"Project: {project_dir}")
            self.append_log(f"Whisper model: {model_path}")
            self.append_log(f"CPU détectés: {CPU_COUNT}")
            self.append_log(
                f"Mode d'exécution: {execution_mode} | workers max demandés: {max_workers}"
            )

            self.update_status("Ingestion audio...", UI_COLORS["text_secondary"])
            if source_mode == "URL Webex":
                self.append_log("Téléchargement/ingestion depuis URL...")
            else:
                self.append_log("Chargement d'un fichier local...")
            extract_audio_to_wav(source_input, audio_path, log=self.append_log)
            self.append_log(f"Audio source converti en WAV: {audio_path}")

            self.update_status("Découpage audio...", UI_COLORS["text_secondary"])
            chunks = split_audio_into_chunks(audio_path, chunks_dir, log=self.append_log)
            prepare_intermediate_dir(raw_parts_dir, log=self.append_log)

            self.update_status("Transcription...", UI_COLORS["text_secondary"])
            requested_workers = max(1, min(int(max_workers), MAX_PARALLEL_WORKERS))
            actual_workers = 1 if execution_mode == "Série" else min(requested_workers, len(chunks))
            whisper_threads = max(1, CPU_COUNT // actual_workers)
            total_chunks = len(chunks)
            self.append_log(
                f"Transcription chunkée: {total_chunks} segment(s), {actual_workers} worker(s), {whisper_threads} thread(s) whisper/processus."
            )

            raw_parts = [None] * total_chunks
            completed_chunks = 0
            progress_lock = threading.Lock()

            def mark_completed(index, part_path):
                nonlocal completed_chunks
                raw_parts[index] = part_path
                with progress_lock:
                    completed_chunks += 1
                    self.update_status(
                        f"Transcription {completed_chunks}/{total_chunks}...",
                        UI_COLORS["text_secondary"],
                    )
                self.append_log(f"Segment {index + 1}/{total_chunks} prêt: {part_path}")

            if actual_workers == 1:
                self.append_log("Transcription en série des segments.")
                for index, chunk_path in enumerate(chunks):
                    result_index, part_path = self._transcribe_chunk(
                        index,
                        total_chunks,
                        chunk_path,
                        raw_parts_dir,
                        lang_in,
                        lang_out,
                        whisper_model,
                        whisper_threads,
                    )
                    mark_completed(result_index, part_path)
            else:
                self.append_log("Transcription en parallèle des segments.")
                with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                    future_map = {
                        executor.submit(
                            self._transcribe_chunk,
                            index,
                            total_chunks,
                            chunk_path,
                            raw_parts_dir,
                            lang_in,
                            lang_out,
                            whisper_model,
                            whisper_threads,
                        ): index
                        for index, chunk_path in enumerate(chunks)
                    }

                    for future in as_completed(future_map):
                        result_index, part_path = future.result()
                        mark_completed(result_index, part_path)

            if any(part is None for part in raw_parts):
                raise RuntimeError("Concaténation impossible: au moins une transcription de segment est manquante.")

            self.update_status("Concaténation brute...", UI_COLORS["text_secondary"])
            self.append_log("Concaténation des segments transcrits vers raw.txt...")
            concat_transcriptions(raw_parts, raw_txt, log=self.append_log)

            self.update_status("Formatage...", UI_COLORS["text_secondary"])
            self.append_log("Formatage...")
            format_for_llm(raw_txt, final_txt, lang_out)
            self.append_log(f"Fichier généré: {final_txt}")

            self.update_status("Terminé", UI_COLORS["success"])
            self.append_log("Terminé")

            self.root.after(
                0,
                lambda: messagebox.showinfo("Succès", f"Projet créé :\n{project_dir}")
            )

        except Exception as e:
            self.append_log(f"ERREUR: {e}")
            self.root.after(0, lambda: messagebox.showerror("Erreur", str(e)))
            self.update_status("Erreur", UI_COLORS["error"])
        finally:
            self.set_running(False)

    def run_pipeline(self):
        try:
            source_mode = self.source_mode.get()
            url = self.url_entry.get().strip()
            local_file = self.local_file_path.get().strip()
            name = self.name_entry.get().strip()
            lang_in = self._language_code(self.lang_input.get(), "auto")
            lang_out = self._language_code(self.lang_output.get(), "fr")
            whisper_model = self.whisper_model.get()
            execution_mode = self.execution_mode.get()
            max_workers = int(self.max_workers.get())

            if not name:
                raise ValueError("Nom du dossier projet manquant")

            if source_mode == "URL Webex":
                if not url:
                    raise ValueError("URL manquante")
                source_input = url
            else:
                if not local_file:
                    raise ValueError("Fichier local manquant")
                local_path = Path(local_file).expanduser()
                if not local_path.is_file():
                    raise FileNotFoundError(f"Fichier introuvable: {local_path}")
                source_input = str(local_path)

            self.set_running(True)
            self.append_log("Lancement du pipeline")
            threading.Thread(
                target=self.run_pipeline_worker,
                args=(
                    source_mode,
                    source_input,
                    name,
                    lang_in,
                    lang_out,
                    whisper_model,
                    execution_mode,
                    max_workers,
                ),
                daemon=True,
            ).start()

        except Exception as e:
            self.set_running(False)
            self.append_log(f"ERREUR: {e}")
            messagebox.showerror("Erreur", str(e))
            self.update_status("Erreur", UI_COLORS["error"])

# ===== RUN =====

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
