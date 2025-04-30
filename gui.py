# [Folder] screenchat
#  [File] gui.py
# --- Start of Updated gui.py ---
import gradio as gr
from windows_to_text import answer_prompt_using_screen
import traceback # Keep for detailed error logging

# Function to handle chat interaction within gr.Blocks (using yield and placeholder)
def handle_chat(user_message: str, history: list, use_all_windows: bool):
    """
    Handles a single chat turn for gr.Blocks, yielding updates with a thinking placeholder.
    Args:
        user_message: The message input by the user.
        history: The current chat history (list of lists).
        use_all_windows: The state of the checkbox.
    Yields:
        Updates for the chatbot history and the message textbox.
    """
    if not user_message:
        history.append((None, "Please enter a message."))
        yield history, ""
        return

    # --- Use a placeholder message instead of None ---
    thinking_placeholder = "..." # You can customize this, e.g., "Thinking..."
    history.append((user_message, thinking_placeholder))
    # Yield the updated history (shows user message and placeholder) and clear the textbox
    yield history, ""
    # --- End of placeholder modification ---

    try:
        # Call the backend function (this might take time)
        assistant_response = answer_prompt_using_screen(user_message, use_all_windows)
        # Update the history by replacing the placeholder
        history[-1] = (user_message, assistant_response)

    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
        error_message = f"An error occurred: {e}. Check console for details."
        # Update history with error message, replacing the placeholder
        history[-1] = (user_message, error_message)

    # Yield the final history (shows assistant response or error)
    yield history, ""


# --- Build UI with gr.Blocks ---
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# ScreenChat Assistant")
    gr.Markdown("Ask questions about your open windows.")

    # Chatbot display
    chatbot = gr.Chatbot(
        label="Chat",
        height=500,
    )

    # Checkbox - placed visibly
    with gr.Row():
         force_all_windows_checkbox = gr.Checkbox(
            label="Force use of all windows",
            info="If checked, text from all open windows will be used. If unchecked, an AI model will try to select relevant windows based on your prompt.",
            value=False, # Default to unchecked
            scale=1 # Adjust scale as needed
        )

    # Text input area and submit button
    with gr.Row():
        msg_textbox = gr.Textbox(
            label="Your message:",
            placeholder="Ask a question based on your screen...",
            scale=7, # Make textbox larger
            container=False,
        )
        submit_btn = gr.Button("Send", scale=1, variant="primary")

    # Examples (using gr.Examples component)
    gr.Examples(
        examples=[
            ["Summarize the content of the editor window."],
            ["What is the main topic of the browser window?"],
            ["List the files shown in the file manager."]
        ],
        inputs=[msg_textbox], # Link examples to the message textbox
        label="Example Prompts"
    )

    # Define interactions
    submit_btn.click(
        fn=handle_chat,
        inputs=[msg_textbox, chatbot, force_all_windows_checkbox],
        outputs=[chatbot, msg_textbox]
    )
    msg_textbox.submit(
        fn=handle_chat,
        inputs=[msg_textbox, chatbot, force_all_windows_checkbox],
        outputs=[chatbot, msg_textbox]
    )

if __name__ == "__main__":
    # Consider adding queue for better handling if tasks are very long or multiple users
    # demo.queue()
    print("Launching Gradio interface... Your browser should open automatically.")
    demo.launch(inbrowser=True)
# --- End of Updated gui.py ---
