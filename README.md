Screenbot is a lightweight desktop chatbot that combines screen capture, OCR, and a large language model to answer your questions based on the content of your open windows.

Features

Context-aware: Automatically filters and OCRs only the windows relevant to your query.

Conversation memory: Maintains dialogue history for more coherent multi-turn interactions.

Cross-platform: Uses wmctrl and ImageMagick on Linux; easily extensible to other environments.

Simple UI: Chat interface built with Gradio for fast setup and use.

🛠️ Installation

Clone the repo

git clone https://github.com/yourusername/screenbot.git
cd screenbot

Create a virtual environment (recommended)

python3 -m venv .venv
source .venv/bin/activate

Install dependencies

pip install -r requirements.txt

Configure your API key
Create a file named .env in the project root:

OPENAI_API_KEY=your_api_key_here

Run the chat interface

python gui.py

💬 Usage

Open your preferred windows and applications.

Type a question or command into the chat box.

Screenbot will capture and OCR relevant windows, consult the model, and reply.

Copy/paste responses as needed.

📦 Files

gui.py: Launches the Gradio chat interface.

windows_to_text.py: Manages window filtering, OCR, and LLM interaction with built-in history.

.env: Stores your API key (not committed).

requirements.txt: Lists project dependencies.

