import os
import subprocess
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# ===== CONFIG =====
BASE_DIR = Path(__file__).parent.resolve()

WHISPER_PATH = os.path.expanduser("~/dev/ai/whisper-cpp/build/bin/whisper-cli")
MODEL_PATH = os.path.expanduser("~/dev/ai/whisper-cpp/models/ggml-medium.bin")

LANGUAGES = ["auto", "fr", "en", "it", "es", "de"]

# ===== CORE =====

def run_cmd(cmd):
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed:\n{cmd}")

def download_audio(url, output_wav):
    cmd = f'ffmpeg -y -i "{url}" -vn -ac 1 -ar 16000 -c:a pcm_s16le "{output_wav}"'
    run_cmd(cmd)

def transcribe(audio_path, output_txt, lang_in, lang_out):

    lang_flag = "" if lang_in == "auto" else f"-l {lang_in}"

    translate_flag = ""
    if lang_out == "en" and lang_in != "en":
        translate_flag = "--translate"

    cmd = f'"{WHISPER_PATH}" -m "{MODEL_PATH}" -f "{audio_path}" {lang_flag} {translate_flag} -otxt'
    run_cmd(cmd)

    base = os.path.splitext(audio_path)[0]
    generated_txt = base + ".txt"
    os.rename(generated_txt, output_txt)

def format_for_llm(input_txt, output_txt, lang_out):
    with open(input_txt, "r") as f:
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

    with open(output_txt, "w") as f:
        f.write(formatted)

# ===== UI =====

class App:
    def __init__(self, root):
        self.root = root
        root.title("Whisper Transcriber")
        root.geometry("720x450")
        root.configure(bg="#0D174E")

        container = tk.Frame(root, bg="#0D174E")
        container.pack(expand=True)

        # ===== LOGO =====
        try:
            self.logo = tk.PhotoImage(file="logo.png")
            tk.Label(container, image=self.logo, bg="#0D174E").pack(pady=10)
        except:
            tk.Label(container,
                     text="WHISPER TRANSCODER",
                     fg="white",
                     bg="#0D174E",
                     font=("Helvetica", 18, "bold")).pack(pady=10)

        # ===== URL =====
        tk.Label(container, text="URL Webex (.m3u8)",
                 fg="white", bg="#0D174E").pack()

        self.url_entry = tk.Entry(container, width=85)
        self.url_entry.pack(pady=5)

        # ===== PROJECT NAME =====
        tk.Label(container, text="Nom du dossier projet",
                 fg="white", bg="#0D174E").pack()

        self.name_entry = tk.Entry(container, width=40)
        self.name_entry.pack(pady=5)

        # ===== LANGUAGE SELECTORS =====
        lang_frame = tk.Frame(container, bg="#0D174E")
        lang_frame.pack(pady=10)

        tk.Label(lang_frame, text="Langue audio",
                 fg="white", bg="#0D174E").grid(row=0, column=0, padx=10)

        self.lang_input = tk.StringVar(value="auto")
        tk.OptionMenu(lang_frame, self.lang_input, *LANGUAGES).grid(row=0, column=1)

        tk.Label(lang_frame, text="Langue sortie",
                 fg="white", bg="#0D174E").grid(row=0, column=2, padx=10)

        self.lang_output = tk.StringVar(value="fr")
        tk.OptionMenu(lang_frame, self.lang_output, *LANGUAGES).grid(row=0, column=3)

        # ===== BUTTON =====
        tk.Button(container,
                  text="Lancer pipeline",
                  bg="#C87920",
                  fg="white",
                  font=("Helvetica", 12, "bold"),
                  command=self.run_pipeline).pack(pady=20)

        # ===== STATUS =====
        self.status = tk.Label(container,
                               text="Prêt",
                               fg="#3C73A1",
                               bg="#0D174E")
        self.status.pack()

    def update_status(self, text):
        self.status.config(text=text)
        self.root.update()

    def run_pipeline(self):
        try:
            url = self.url_entry.get().strip()
            name = self.name_entry.get().strip()
            lang_in = self.lang_input.get()
            lang_out = self.lang_output.get()

            if not url or not name:
                raise ValueError("Champs manquants")

            project_dir = BASE_DIR / name
            os.makedirs(project_dir, exist_ok=True)

            audio_path = project_dir / "audio.wav"
            raw_txt = project_dir / "raw.txt"
            final_txt = project_dir / "formatted.txt"

            self.update_status("Téléchargement audio...")
            download_audio(url, audio_path)

            self.update_status("Transcription...")
            transcribe(audio_path, raw_txt, lang_in, lang_out)

            self.update_status("Formatage...")
            format_for_llm(raw_txt, final_txt, lang_out)

            self.update_status("Terminé")
            messagebox.showinfo("Succès", f"Projet créé :\n{project_dir}")

        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            self.update_status("Erreur")

# ===== RUN =====

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()