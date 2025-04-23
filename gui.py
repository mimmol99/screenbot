# [Folder] screenchat
#   [File] gui.py
import gradio as gr
from windows_to_text import answer_prompt_using_screen
import traceback # Import traceback for detailed error logging


def send_message(prompt: str, history: list, use_all_windows: bool) -> str:
    """
    Gradio callback: accepts prompt, history, and checkbox state.
    Returns the assistant response string.
    """
    if not prompt:
        return "Please enter a message."

    try:
        # Pass the state of the checkbox to the backend function
        return answer_prompt_using_screen(prompt, use_all_windows)
    except Exception as e:
        # Log the full traceback for debugging
        print(f"Error occurred: {e}")
        traceback.print_exc()
        return f"An error occurred: {e}. Check console for details."


# Create the checkbox component
force_all_windows_checkbox = gr.Checkbox(
    label="Force use of all windows",
    info="If checked, text from all open windows will be used. If unchecked, an AI model will try to select relevant windows based on your prompt.",
    value=False # Default to unchecked (model selects windows)
)

# Explicitly create Chatbot with type='messages' to address the warning
chatbot_component = gr.Chatbot(
    type="messages",
    height=500  # You can adjust the height here if needed
)

# Create the ChatInterface, passing the pre-configured chatbot
iface = gr.ChatInterface(
    fn=send_message,
    chatbot=chatbot_component, # Pass the explicit chatbot component
    textbox=gr.Textbox(placeholder="Ask a question based on your screen...", container=False, scale=7),
    title="ScreenChat Assistant",
    description="Ask questions about your open windows. Optionally force using all windows.",
    theme="soft",
    examples=[["Summarize the content of the editor window."], ["What is the main topic of the browser window?"], ["List the files shown in the file manager."]],
    additional_inputs=[force_all_windows_checkbox],
)

if __name__ == "__main__":
    iface.launch()
    # To create a public link (if needed and safe):
    # iface.launch(share=True)
