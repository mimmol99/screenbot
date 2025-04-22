# 📸 Screenbot

Screenbot is a lightweight desktop assistant that **reads the text on your open application windows in real time (via OCR)** and lets you query that information through an OpenAI‑powered chat interface built with [Gradio](https://gradio.app/).

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="python badge">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="license badge">
</p>

## ✨ Features
- **One‑click chat UI** – talk to your desktop with Gradio’s ChatInterface  
- **Context‑aware answers** – only the windows relevant to your prompt are OCR’d (selection done by the LLM)  
- **Stateless OCR pipeline** – uses `wmctrl` + ImageMagick’s `import` + Tesseract; no screenshots written to disk  
- **Session memory** – past turns are kept in `ChatSession`, so follow‑ups work naturally  
- **Environment‑only keys** – reads `OPENAI_API_KEY` from a `.env` file (never hard‑code secrets)

## 🖥  Quick start

```bash
git clone https://github.com/your‑username/screenbot.git
cd screenbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # then paste your OPENAI_API_KEY
python screenbot/gui.py
