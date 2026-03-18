# Whisper Transcriber — Webex → Audio → Transcription → LaTeX-ready
Pipeline automatisé pour :
1. Télécharger un stream Webex (.m3u8)
2. Extraire uniquement l’audio
3. Transcrire avec whisper.cpp (modèle medium)
4. Générer un texte structuré prêt pour un agent IA → cours LaTeX

---

## Architecture

```

GUI (Python Tkinter)
↓
ffmpeg (download + extraction audio)
↓
whisper.cpp (transcription C++ optimisée Metal)
↓
post-processing (format LLM → LaTeX)

````

---

## Fonctionnalités

- Interface simple (GUI)
- Support des liens Webex `.m3u8`
- Extraction audio optimisée (mono, 16 kHz)
- Transcription locale (aucune API externe)
- Output structuré pour génération automatique de cours
- Organisation automatique des fichiers

---

## Installation

### 1. Dépendances système

```bash
brew install ffmpeg cmake git
````

---

### 2. Installer whisper.cpp

```bash
cd ~/dev/ai
git clone https://github.com/ggerganov/whisper.cpp.git whisper-cpp
cd whisper-cpp

cmake -B build -DGGML_METAL=ON
cmake --build build -j
```

---

### 3. Télécharger modèle medium

```bash
./models/download-ggml-model.sh medium
```

---

### 4. Python

```bash
pip install tk
```

---

## Configuration

Dans `app.py`, vérifier :

```python
WHISPER_PATH = "~/dev/ai/whisper-cpp/build/bin/whisper-cli"
MODEL_PATH = "~/dev/ai/whisper-cpp/models/ggml-medium.bin"
```

---

## Utilisation

```bash
python app.py
```

### Workflow utilisateur

1. Coller URL Webex `.m3u8`
2. Choisir dossier parent
3. Donner un nom de projet
4. Cliquer sur **Lancer**

---

## Structure générée

```
parent/
└── project_name/
    ├── audio.wav          # audio extrait (16 kHz mono)
    ├── raw.txt            # transcription brute
    └── formatted.txt      # prêt pour LLM → LaTeX
```

---

## Pipeline technique

### 1. Download + extraction audio

```bash
ffmpeg -i <URL> -vn -ac 1 -ar 16000 -c:a pcm_s16le audio.wav
```

---

### 2. Transcription

```bash
whisper-cli -m ggml-medium.bin -f audio.wav -otxt
```

---

### 3. Post-processing

Le fichier final contient :

* transcription brute
* instructions structurées pour un agent IA

---

## Format de sortie (LLM-ready)

```text
# TRANSCRIPTION BRUTE

...

# INSTRUCTIONS POUR IA

Convert this transcription into a structured LaTeX course:
- sections / subsections
- équations
- style académique
- suppression bruit oral
```

---

## Limitations

* dépend de la validité du stream Webex
* pas de gestion avancée des erreurs réseau
* pas de segmentation audio (long fichiers → lent)
* UI minimale

---

## Extensions possibles

* génération automatique `.tex`
* export `.pdf`
* batch processing (dossier complet)
* découpage audio (chunking)
* diarisation (speaker separation)
* UI React/Electron

---

## Performance

| Modèle | Vitesse     | Qualité    |
| ------ | ----------- | ---------- |
| tiny   | très rapide | faible     |
| base   | rapide      | correcte   |
| medium | équilibré   | élevé      |
| large  | lent        | très élevé |

---

## Bonnes pratiques

* ne pas versionner :

  * `models/`
  * `build/`
  * fichiers audio
* utiliser `.gitignore`
* stocker les outputs localement

---

## Dépendances clés

* ffmpeg
* whisper.cpp (C++ / Metal)
* Python (GUI uniquement)

---

## Licence

MIT 

---

## Auteur

Projet orienté usage ingénierie :

* prise de notes automatisée
* génération de supports de cours
* pipeline reproductible local



---

## Option (fortement recommandé)

Ajoute un `.gitignore` cohérent avec ce README :

```bash
models/
build/
*.wav
*.mp3
*.txt
__pycache__/
.DS_Store
````

