# [Folder] screenchat
#  [File] windows_to_text.py
# --- Start of Updated windows_to_text.py ---
import io
import subprocess
import pytesseract
from PIL import Image
import os
from pydantic import BaseModel, ValidationError, Field
from openai import OpenAI
# Importing RateLimitError for specific error handling might be useful
from openai import RateLimitError, APIError, APITimeoutError
from dotenv import load_dotenv
import traceback

# Load environment variables
dotenv_path = load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables or .env file.")
client = OpenAI(api_key=api_key)

# Define outside function or globally if preferred
DEFAULT_MODEL = "gpt-4.1-mini" # Example default
DEFAULT_FALLBACK_MODELS = [DEFAULT_MODEL, "gpt-4.1-nano", "gpt-4.1"]

# More targeted exclusion list
EXCLUDED_MODEL_SUBSTRINGS = {
    "instruct", "vision", "embed", "audio", "tts", "image",
    "moderation", "computer", "dall", "whisper", "transcribe",
    "davinci", "curie", "babbage", "ada"
}

def get_available_models(client_instance: OpenAI) -> list[str]:
    """
    Fetches available models from OpenAI API and filters for likely chat/reasoning models.
    Returns a default list if the API call fails or filtering yields no results.
    """
    print("Fetching available OpenAI models...")
    try:
        models_response = client_instance.models.list()
        available_models = models_response.data # Access the list of model objects

        filtered_models = set() # Use set for efficient addition

        for model in available_models:
            model_id = model.id
            model_id_lower = model_id.lower()

            # --- Refined Filtering ---
            # 1. Positive check: Must be a GPT or O-series model (common prefixes)
            is_potential_chat = model_id.startswith(('gpt-', 'o'))

            # 2. Negative check: Exclude based on common non-chat substrings
            is_excluded_type = any(sub in model_id_lower for sub in EXCLUDED_MODEL_SUBSTRINGS)

            # 3. Negative check: Exclude fine-tuned (often contain ':') and specific versions ('@')
            is_finetuned_or_versioned = ':' in model_id or '@' in model_id

            if is_potential_chat and not is_excluded_type and not is_finetuned_or_versioned:
                filtered_models.add(model_id)
            # --- End Refined Filtering ---

        # Add default model to ensure it's available if accessible
        filtered_models.add(DEFAULT_MODEL)

        if not filtered_models:
            print("Warning: No chat models found after filtering. Returning default fallback list.")
            return DEFAULT_FALLBACK_MODELS

        # Convert set to sorted list
        chat_models_list = sorted(list(filtered_models))
        print(f"Filtered models available for selection: {chat_models_list}")
        return chat_models_list

    # Handle specific API errors
    except (APIError, RateLimitError, APITimeoutError) as api_err:
        print(f"Error fetching models from OpenAI API: {api_err}. Returning default fallback list.")
        return DEFAULT_FALLBACK_MODELS
    # Handle other potential exceptions
    except Exception as e:
        print(f"An unexpected error occurred fetching models: {e}. Returning default fallback list.")
        traceback.print_exc()
        return DEFAULT_FALLBACK_MODELS


class Window(BaseModel):
    window_id: str = Field(..., description="The unique identifier of the window (e.g., '0x0...').")
    window_name: str = Field(..., description="The title/name of the window.")


class WindowList(BaseModel):
    windows: list[Window] = Field(..., description="A list of windows relevant to the user's prompt.")


def parse_wmctrl_output(output: str) -> list[tuple[str, str]]:
    """Parses the output of `wmctrl -l` into a list of (id, name) tuples."""
    windows = []
    lines = output.strip().splitlines()
    for line in lines:
        parts = line.split(maxsplit=3)
        if len(parts) >= 4:
            window_id = parts[0]
            window_name = parts[3]
            windows.append((window_id, window_name))
    return windows

# Modified to accept selected_model
def filter_windows(prompt: str, all_windows: list[tuple[str, str]], selected_model: str) -> tuple[list[str], list[str]]:
    """
    Ask the model to pick relevant windows for the given prompt using client.responses.parse.
    Args:
        prompt: The user's query.
        all_windows: A list of tuples (window_id, window_name) for all windows.
        selected_model: The OpenAI model ID to use.
    Returns:
        A tuple containing (list of selected window IDs, list of selected window names).
    """
    window_list_str = "\n".join([f"ID: {wid}, Name: {name}" for wid, name in all_windows])
    if not window_list_str:
        print("No windows found to filter.")
        return [], []

    messages = [
        {"role": "system", "content": f"You are a window filtering assistant. Based on the user's prompt and the list of available windows, identify the windows relevant for answering the prompt. Respond using the structure defined by the '{WindowList.__name__}' tool/format."},
        {"role": "user", "content": f"User Prompt: {prompt}\n\nAvailable Windows:\n{window_list_str}\n\nSelect the relevant windows."}
    ]

    try:
        print(f"Requesting structured window filtering from model ({selected_model}) using client.responses.parse...") # Use selected_model

        response = client.responses.parse(
            model=selected_model, # Use selected_model
            input=messages,
            text_format=WindowList
        )

        selected_windows = []
        if hasattr(response, 'output_parsed') and response.output_parsed:
             parsed_data: WindowList = response.output_parsed
             if isinstance(parsed_data, WindowList):
                selected_windows = parsed_data.windows
             else:
                 print(f"Warning: Parsed data is not of type WindowList. Type received: {type(parsed_data)}")
                 return [], []
        else:
            print("Warning: Model response did not contain parsed output or parsing failed.")
            return [], []

        all_window_map = {wid: name for wid, name in all_windows}
        valid_ids = set(all_window_map.keys())
        selected_ids = []
        selected_names = []
        for w in selected_windows:
            if w.window_id in valid_ids:
                selected_ids.append(w.window_id)
                selected_names.append(all_window_map[w.window_id])
            else:
                print(f"Warning: Model selected a window ID not in the original list: {w.window_id}, Name: {w.window_name}. Skipping.")

        print(f"Model selected windows (validated): {selected_names}")
        return selected_ids, selected_names

    except ValidationError as pyd_err:
        print(f"Error validating model's output against Pydantic model: {pyd_err}")
        return [], []
    except AttributeError as attr_err:
        print(f"AttributeError: Could not call 'client.responses.parse' or access 'output_parsed'. Check SDK version, method name, and model compatibility. Error: {attr_err}")
        traceback.print_exc()
        return [], []
    except Exception as e:
        # Improved error message detail
        error_info = str(e).lower()
        print(f"An unexpected error occurred during structured window filtering with model '{selected_model}': {e}")
        if "model is not supported" in error_info or "invalid model" in error_info or "does not exist" in error_info or "responses.parse" in error_info:
            print(f"Error might be due to model '{selected_model}' not supporting this feature, the feature not being available, or the model not existing/accessible.")
        traceback.print_exc()
        return [], []

# Modified to accept selected_model
def extract_windows_text(prompt: str, use_all_windows: bool, selected_model: str) -> tuple[str, list[str]]:
    """
    Lists windows, optionally filters them, captures screenshots, and performs OCR.
    Args:
        prompt: The user's query.
        use_all_windows: If True, use all windows; otherwise, filter them.
        selected_model: The OpenAI model ID to use for filtering (if applicable).
    Returns:
        A tuple containing:
        - The combined OCR text from the selected windows.
        - A list of the names of the windows that were successfully processed.
    """
    try:
        result = subprocess.run(["wmctrl", "-l"], capture_output=True, text=True, check=True)
    except FileNotFoundError:
         raise RuntimeError("Failed to run 'wmctrl'. Is it installed and in PATH?")
    except subprocess.CalledProcessError as e:
         raise RuntimeError(f"Failed to list windows using 'wmctrl': {e}")

    all_windows: list[tuple[str, str]] = parse_wmctrl_output(result.stdout)
    if not all_windows:
        print("No windows found by wmctrl.")
        return "", []

    target_window_ids: list[str] = []
    target_window_names: list[str] = []

    if use_all_windows:
        print("Forcing use of all windows.")
        target_window_ids = [wid for wid, name in all_windows]
        target_window_names = [name for wid, name in all_windows]
    else:
        print("Filtering windows based on prompt...")
        # Pass selected_model to filter_windows
        target_window_ids, target_window_names = filter_windows(prompt, all_windows, selected_model)
        if not target_window_ids:
            print("Model did not select any relevant windows or filtering failed.")

    texts = []
    processed_window_names = []
    for wid, name in zip(target_window_ids, target_window_names):
        try:
            cap_proc = subprocess.run(["import", "-window", wid, "png:-"], capture_output=True, check=True, timeout=10)
            img = Image.open(io.BytesIO(cap_proc.stdout))
            try:
                 text = pytesseract.image_to_string(img)
                 texts.append(f"Window Name: {name}\n--- Start Content ---\n{text}\n--- End Content ---")
                 processed_window_names.append(name)
                 print(f"Successfully OCR'd window: {name}")
            except pytesseract.TesseractNotFoundError:
                 raise RuntimeError("Tesseract is not installed or not in PATH.")
            except Exception as ocr_err:
                 print(f"Error during OCR for window '{name}' ({wid}): {ocr_err}")
        except FileNotFoundError:
             raise RuntimeError("Failed to run 'import'. Is ImageMagick installed and in PATH?")
        except subprocess.CalledProcessError as e:
            print(f"Failed to capture screenshot for window '{name}' ({wid}): {e}")
        except subprocess.TimeoutExpired:
            print(f"Timeout capturing screenshot for window '{name}' ({wid})")
        except Exception as img_err:
            print(f"Error processing image for window '{name}' ({wid}): {img_err}")

    combined_text = "\n\n".join(texts)
    return combined_text, processed_window_names


class ChatSession:
    """Maintains conversation history and processes queries."""
    def __init__(self):
        self.history: list[dict[str, str]] = []

    # Modified to accept selected_model
    def ask(self, prompt: str, use_all_windows: bool, selected_model: str) -> str:
        """
        Processes prompt, gets screen context, calls LLM, updates history.
        Args:
            prompt: The user's input query.
            use_all_windows: Flag to control window filtering.
            selected_model: The OpenAI model ID to use.
        Returns:
            The assistant's response string.
        """
        print(f"\n--- New Turn ---")
        print(f"User Prompt: {prompt}")
        print(f"Use All Windows: {use_all_windows}")
        print(f"Using Model: {selected_model}") # Log selected model

        try:
            # Pass selected_model to extract_windows_text -> filter_windows
            screen_text, used_window_names = extract_windows_text(prompt, use_all_windows, selected_model)
            if not screen_text:
                 print("No screen text extracted or no relevant windows found/processed.")
        except Exception as e:
            print(f"Error extracting window text: {e}")
            traceback.print_exc()
            return f"Error during screen text extraction: {e}"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. Use the provided Screen Content section, "
                    "which contains text extracted from relevant application windows, "
                    "along with the conversation history to answer the User Prompt concisely and accurately. "
                    "If the screen content is empty or irrelevant, answer based on the prompt and history alone."
                )
            }
        ]
        messages.extend(self.history)

        user_content = f"Screen Content:\n"
        if screen_text:
            user_content += f"{screen_text}\n\n"
        else:
            user_content += "[No relevant screen content was extracted or provided]\n\n"
        user_content += f"User Prompt: {prompt}"
        messages.append({"role": "user", "content": user_content})

        try:
            # Use selected_model in the API call
            completion = client.chat.completions.create(
                model=selected_model, # Use the selected model
                temperature=0.3,
                messages=messages
            )
            answer = completion.choices[0].message.content

        except Exception as e:
             print(f"Error calling OpenAI API with model '{selected_model}': {e}")
             traceback.print_exc()
             return f"Error communicating with AI model '{selected_model}': {e}"

        self.history.append({"role": "user", "content": user_content}) # Store full context
        self.history.append({"role": "assistant", "content": answer})

        final_response = answer
        # (Existing logic to prepend window info - unchanged)
        if not use_all_windows and used_window_names:
            window_list_str = ", ".join(used_window_names)
            final_response = f"(Info: Used content from windows: {window_list_str})\n\n{answer}"
        elif use_all_windows and used_window_names:
             final_response = f"(Info: Used content from all detected windows)\n\n{answer}"
        # ... (rest of the info messages)

        print(f"Assistant Response:\n{final_response}")
        return final_response

# Single session instance
session = ChatSession()

# Modified to accept selected_model
def answer_prompt_using_screen(prompt: str, use_all_windows: bool, selected_model: str) -> str:
    """
    Public function: returns the session-managed, context-aware answer.
    Args:
        prompt: The user's input query.
        use_all_windows: Flag to control window filtering.
        selected_model: The OpenAI model ID to use.
    Returns:
        The assistant's response string.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return "Error: Prompt cannot be empty."
    if not isinstance(use_all_windows, bool):
         print("Warning: Invalid type for use_all_windows, defaulting to False.")
         use_all_windows = False
    if not isinstance(selected_model, str) or not selected_model.strip():
         print(f"Warning: Invalid or empty model selected, defaulting to {DEFAULT_MODEL}.")
         selected_model = DEFAULT_MODEL

    # Pass selected_model to session.ask
    return session.ask(prompt.strip(), use_all_windows, selected_model)

# Export the client instance if needed by gui.py for fetching models
openai_client = client
