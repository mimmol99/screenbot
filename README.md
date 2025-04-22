Screenbot Chat Assistant

Screenbot is a lightweight desktop chatbot that combines screen capture, OCR, and a large language model to answer user queries based on the current contents of your open windows.

Features

Context‑aware: Filters and OCRs only the windows relevant to your question.

History memory: Maintains conversation context across turns.

Cross‑platform: Uses wmctrl and ImageMagick on Linux; easily extensible to other environments.

Easy interface: Simple chat UI built with Gradio.

Getting Started

Clone the repository:

git clone https://github.com/yourusername/screenbot.git
cd screenbot

Create and activate a virtual environment (recommended):

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Create a .env file in the project root with your OpenAI API key:

OPENAI_API_KEY=your_api_key_here

Run the chat interface:

python gui.py

Usage

Type a question into the chat input.

The bot will automatically capture and OCR relevant windows, then respond based on that content and past dialogue.

Use standard copy/paste to export responses.

