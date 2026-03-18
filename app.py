import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

# ===== CONFIG =====
WHISPER_PATH = os.path.expanduser("~/dev/ai/whisper-cpp/build/bin/whisper-cli")
MODEL_PATH = os.path.expanduser("~/dev/ai/whisper-cpp/models/ggml-medium.bin")

# ===== CORE FUNCTIONS =====

def run_cmd(cmd):
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def download_audio(url, output_wav):
    cmd = f'ffmpeg -y -i "{url}" -vn -ac 1 -ar 16000 -c:a pcm_s16le "{output_wav}"'
    run_cmd(cmd)

def transcribe(audio_path, output_txt):
    cmd = f'"{WHISPER_PATH}" -m "{MODEL_PATH}" -f "{audio_path}" -otxt'
    run_cmd(cmd)

    # whisper génère un .txt avec le même nom
    base = os.path.splitext(audio_path)[0]
    generated_txt = base + ".txt"

    os.rename(generated_txt, output_txt)

def format_for_llm(input_txt, output_txt):
    with open(input_txt, "r") as f:
        text = f.read()

    formatted = f"""
# TRANSCRIPTION BRUTE

{text}

# INSTRUCTIONS POUR IA

Convert this transcription into a structured LaTeX course:

Requirements:
- Organize into sections, subsections
- Extract key concepts, definitions, formulas
- Remove filler words and noise
- Rewrite in formal academic style
- Use LaTeX environments: \\section, \\subsection, \\begin{{equation}}, etc.
- Add examples when relevant
- Preserve technical rigor
"""

    with open(output_txt, "w") as f:
        f.write(formatted)

# ===== GUI =====

class App:
    def __init__(self, root):
        self.root = root
        root.title("Whisper Transcriber")

        tk.Label(root, text="URL Webex (.m3u8)").pack()
        self.url_entry = tk.Entry(root, width=100)
        self.url_entry.pack()

        tk.Label(root, text="Nom du dossier projet").pack()
        self.name_entry = tk.Entry(root)
        self.name_entry.pack()

        tk.Button(root, text="Choisir dossier parent", command=self.select_folder).pack()
        self.folder_label = tk.Label(root, text="Aucun dossier sélectionné")
        self.folder_label.pack()

        tk.Button(root, text="Lancer", command=self.run_pipeline).pack()

        self.parent_dir = None

    def select_folder(self):
        self.parent_dir = filedialog.askdirectory()
        self.folder_label.config(text=self.parent_dir)

    def run_pipeline(self):
        try:
            url = self.url_entry.get()
            name = self.name_entry.get()

            if not url or not name or not self.parent_dir:
                raise ValueError("Champs manquants")

            project_dir = os.path.join(self.parent_dir, name)
            os.makedirs(project_dir, exist_ok=True)

            audio_path = os.path.join(project_dir, "audio.wav")
            raw_txt = os.path.join(project_dir, "raw.txt")
            final_txt = os.path.join(project_dir, "formatted.txt")

            # STEP 1: Download audio
            download_audio(url, audio_path)

            # STEP 2: Transcription
            transcribe(audio_path, raw_txt)

            # STEP 3: Format
            format_for_llm(raw_txt, final_txt)

            messagebox.showinfo("Succès", f"Terminé : {project_dir}")

        except Exception as e:
            messagebox.showerror("Erreur", str(e))


# ===== RUN =====

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()