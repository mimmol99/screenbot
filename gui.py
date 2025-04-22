import gradio as gr
from windows_to_text import answer_prompt_using_screen


def send_message(prompt, history):
    """
    Gradio callback: accepts prompt and history but uses only the prompt.
    Returns the assistant response string.
    """
    if not prompt:
        return "Please enter a message."

    try:
        return answer_prompt_using_screen(prompt)
    except Exception as e:
        return f"Error: {e}"


iface = gr.ChatInterface(
    fn=send_message,
    type="messages",
    autofocus=True
)
iface.launch()
