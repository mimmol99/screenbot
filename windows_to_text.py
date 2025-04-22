import io
import subprocess
import pytesseract
from PIL import Image
import os
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
dotenv_path = load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL = "gpt-4.1-mini"


class Window(BaseModel):
    window_id: str
    window_name: str


class WindowList(BaseModel):
    windows: list[Window]


def filter_windows(prompt, windows):
    """
    Ask the model to pick relevant windows for the given prompt.
    """
    messages = [
        {"role": "system", "content": "Filter the given windows to those useful for answering the prompt."},
        {"role": "user", "content": f"Prompt: {prompt}\nWindows: {windows}"}
    ]
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        temperature=0.2,
        messages=messages,
        response_format=WindowList,
    )
    selected = completion.choices[0].message.parsed.windows
    return [w.window_id for w in selected], [w.window_name for w in selected]


def extract_windows_text(prompt):
    """
    OCR screenshots of windows filtered by the prompt.
    """
    result = subprocess.run(["wmctrl", "-l"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("Failed to list windows")

    windows = result.stdout.splitlines()
    window_ids, window_names = filter_windows(prompt, windows)

    texts = []
    for wid, name in zip(window_ids, window_names):
        cap = subprocess.run(["import", "-window", wid, "png:-"], capture_output=True)
        if cap.returncode == 0:
            img = Image.open(io.BytesIO(cap.stdout))
            texts.append(f"{name}:\n{pytesseract.image_to_string(img)}")

    return "\n\n".join(texts)


class ChatSession:
    """
    Maintains conversation history and augments each query with screen OCR.
    """
    def __init__(self):
        self.history: list[tuple[str, str]] = []

    def ask(self, prompt: str) -> str:
        # OCR current screen windows
        screen_text = extract_windows_text(prompt)

        # Build messages: system + past turns + new user turn with screen context
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that uses provided screen content "
                    "and full conversation context to answer the user."
                )
            }
        ]

        for user_msg, assistant_msg in self.history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Screen content:\n{screen_text}\n\nUser prompt: {prompt}"
                )
            }
        )

        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=messages
        )
        answer = completion.choices[0].message.content

        # Save turn and return reply
        self.history.append((prompt, answer))
        return answer


# Single session instance for the app
session = ChatSession()


def answer_prompt_using_screen(prompt: str) -> str:
    """
    Public function: returns the session-managed, context-aware answer.
    """
    return session.ask(prompt)
