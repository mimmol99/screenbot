# 📸 Screenchat - Chat With Your Desktop

Screenchat reads text from your open windows (using OCR) and lets you ask an AI about it via a simple chat interface. Runs on Linux.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python Badge">
  <img src="https://img.shields.io/badge/OS-Linux-orange.svg" alt="OS Badge">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License Badge">
</p>

## ✨ Features

* Reads text from any open window (via OCR).
* Simple Gradio chat interface.
* Selectable Model: Choose your preferred OpenAI model from a dropdown.
* Smart context (AI picks relevant windows) or force reading all windows (via always-visible checkbox).
* See your messages instantly while AI responds.
* Remembers conversation history for follow-ups.
* Private: No screenshots saved to disk; API key stored locally in `.env`.

## 🚀 Setup

**1. Requirements:**
* **System:** Linux, Python 3.9+, `git`.
* **Tools:** `wmctrl`, `imagemagick`, `tesseract-ocr` (plus language packs like `-eng`).
    * *(Debian/Ubuntu): `sudo apt update && sudo apt install wmctrl imagemagick tesseract-ocr tesseract-ocr-eng`*
    * *(Fedora): `sudo dnf check-update && sudo dnf install wmctrl ImageMagick tesseract tesseract-langpack-eng`*

**2. Installation Steps:**
```bash
# 1. Clone your repository
git clone [https://github.com/mimmol99/screenchat.git](https://github.com/mimmol99/screenchat.git)
cd screenchat

# 2. Install system tools (if not already done - see Requirements above)

# 3. Setup Python environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # Use .venv\Scripts\activate on Windows/Git Bash

# 4. Install Python packages from requirements file
pip install -r requirements.txt

# 5. Configure API Key
cp .env.example .env
# --> Now edit .env and add your OPENAI_API_KEY

# 6. Run!
python gui.py
