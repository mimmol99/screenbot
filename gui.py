# [Folder] screenchat
#  [File] gui.py
# --- Start of Updated gui.py ---
import gradio as gr
# Import the function to get models and the default model name
from windows_to_text import answer_prompt_using_screen, get_available_models, openai_client, DEFAULT_MODEL
import traceback

# --- Fetch available models ONCE at startup ---
# Pass the client instance initialized in windows_to_text
AVAILABLE_MODELS = get_available_models(openai_client)
# Ensure default is selected if list fetching failed or default isn't present
INITIAL_MODEL = DEFAULT_MODEL if DEFAULT_MODEL in AVAILABLE_MODELS else (AVAILABLE_MODELS[0] if AVAILABLE_MODELS else DEFAULT_MODEL)

# Modified to accept selected_model
def handle_chat(user_message: str, history: list, use_all_windows: bool, selected_model: str):
    """
    Handles a single chat turn for gr.Blocks, yielding updates with a thinking placeholder.
    Args:
        user_message: The message input by the user.
        history: The current chat history (list of lists).
        use_all_windows: The state of the checkbox.
        selected_model: The model chosen from the dropdown.
    Yields:
        Updates for the chatbot history and the message textbox.
    """
    if not user_message:
        history.append((None, "Please enter a message."))
        yield history, ""
        return

    thinking_placeholder = "..."
    history.append((user_message, thinking_placeholder))
    yield history, ""

    try:
        # Pass selected_model to the backend function
        assistant_response = answer_prompt_using_screen(user_message, use_all_windows, selected_model)
        history[-1] = (user_message, assistant_response)

    except Exception as e:
        print(f"Error occurred in Gradio handler: {e}")
        traceback.print_exc()
        error_message = f"An error occurred: {e}. Check console for details."
        history[-1] = (user_message, error_message)

    yield history, ""


# --- Build UI with gr.Blocks ---
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# ScreenChat Assistant")
    gr.Markdown("Ask questions about your open windows.")

    chatbot = gr.Chatbot(
        label="Chat",
        height=450, # Adjusted height slightly
    )

    # --- Options Row ---
    with gr.Row():
        model_dropdown = gr.Dropdown(
            label="Select OpenAI Model",
            choices=AVAILABLE_MODELS,
            value=INITIAL_MODEL, # Use fetched initial model
            scale=2 # Give dropdown more space than checkbox
        )
        force_all_windows_checkbox = gr.Checkbox(
            label="Force use of all windows",
            info="Use all windows instead of AI selection.",
            value=False,
            scale=1
        )

    # --- Input Row ---
    with gr.Row():
        msg_textbox = gr.Textbox(
            label="Your message:",
            placeholder="Ask a question based on your screen...",
            scale=7,
            container=False,
        )
        submit_btn = gr.Button("Send", scale=1, variant="primary")

    gr.Examples(
        examples=[
            ["Summarize the content of the editor window."],
            ["What is the main topic of the browser window?"],
            ["List the files shown in the file manager."]
        ],
        inputs=[msg_textbox],
        label="Example Prompts"
    )

    # --- Define interactions ---
    # Gather all inputs for the handler function
    inputs_list = [msg_textbox, chatbot, force_all_windows_checkbox, model_dropdown]

    submit_btn.click(
        fn=handle_chat,
        inputs=inputs_list,
        outputs=[chatbot, msg_textbox]
    )
    msg_textbox.submit(
        fn=handle_chat,
        inputs=inputs_list,
        outputs=[chatbot, msg_textbox]
    )

if __name__ == "__main__":
    print(f"Default model set to: {INITIAL_MODEL}")
    print(f"Available models for dropdown: {AVAILABLE_MODELS}")
    print("Launching Gradio interface... Your browser should open automatically.")
    demo.launch(inbrowser=True)
