# [Folder] screenchat
#  [File] windows_to_text.py
# --- Start of Optimized windows_to_text.py ---
import io
import subprocess
import pytesseract
from PIL import Image
import os
from pydantic import BaseModel, ValidationError, Field
from openai import OpenAI
from dotenv import load_dotenv
import traceback

# Load environment variables
dotenv_path = load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables or .env file.")
client = OpenAI(api_key=api_key)
# --- Using the user-specified model ---
MODEL = "gpt-4.1-mini"
# Warning: 'gpt-4.1-mini' might not be a standard OpenAI model identifier.
# Consider using 'gpt-4o-mini' or another verified model if errors occur.
# Smaller models may struggle with structured output tasks.


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


# --- Updated filter_windows function using client.responses.parse ---
def filter_windows(prompt: str, all_windows: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """
    Ask the model to pick relevant windows for the given prompt using client.responses.parse.
    Args:
        prompt: The user's query.
        all_windows: A list of tuples (window_id, window_name) for all windows.
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
        print(f"Requesting structured window filtering from model ({MODEL}) using client.responses.parse...")

        # --- Using client.responses.parse as requested ---
        response = client.responses.parse(
            model=MODEL,
            input=messages,
            text_format=WindowList # Pass the Pydantic class as text_format
            # Note: Additional parameters like temperature might not be directly supported here.
        )

        # Access the parsed Pydantic object using 'output_parsed'
        selected_windows = [] # Initialize
        if hasattr(response, 'output_parsed') and response.output_parsed:
             parsed_data: WindowList = response.output_parsed
             # Check if the parsed data is actually of the expected type
             if isinstance(parsed_data, WindowList):
                selected_windows = parsed_data.windows
             else:
                 print(f"Warning: Parsed data is not of type WindowList. Type received: {type(parsed_data)}")
                 return [], []
        else:
            print("Warning: Model response did not contain parsed output or parsing failed.")
            # print(f"Raw response object (if available): {response}") # For debugging if needed
            return [], []


        # Validate selected windows against the original list
        all_window_map = {wid: name for wid, name in all_windows}
        valid_ids = set(all_window_map.keys())

        selected_ids = []
        selected_names = []
        for w in selected_windows:
            if w.window_id in valid_ids:
                selected_ids.append(w.window_id)
                selected_names.append(all_window_map[w.window_id]) # Use original name
            else:
                print(f"Warning: Model selected a window ID not in the original list: {w.window_id}, Name: {w.window_name}. Skipping.")

        print(f"Model selected windows (validated): {selected_names}")
        return selected_ids, selected_names

    # Catch Pydantic validation errors
    except ValidationError as pyd_err:
        print(f"Error validating model's output against Pydantic model: {pyd_err}")
        # The raw response might be harder to get from client.responses.parse on error
        return [], []
    # Catch potential API errors, attribute errors if method/model is invalid, etc.
    except AttributeError as attr_err:
        print(f"AttributeError: Could not call 'client.responses.parse' or access 'output_parsed'. Check SDK version, method name, and model compatibility. Error: {attr_err}")
        traceback.print_exc()
        return [], []
    except Exception as e:
        print(f"An unexpected error occurred during structured window filtering with client.responses.parse: {e}")
        # Check if the error message suggests model incompatibility
        if "model is not supported" in str(e).lower() or "invalid model" in str(e).lower() or "responses.parse" in str(e).lower():
            print(f"Error might be due to model '{MODEL}' not supporting this feature, the feature not being available, or the model not existing.")
        traceback.print_exc()
        return [], []


def extract_windows_text(prompt: str, use_all_windows: bool) -> tuple[str, list[str]]:
    """
    Lists windows, optionally filters them based on the prompt using an AI model,
    and performs OCR on the screenshots of the selected windows.

    Args:
        prompt: The user's query.
        use_all_windows: If True, use all windows; otherwise, filter them.

    Returns:
        A tuple containing:
        - The combined OCR text from the selected windows.
        - A list of the names of the windows that were used.
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
        target_window_ids, target_window_names = filter_windows(prompt, all_windows)
        if not target_window_ids:
            print("Model did not select any relevant windows.")
            # Optionally, we could fall back to using all windows here,
            # but current logic will proceed with no window context.

    texts = []
    processed_window_names = [] # Keep track of windows successfully OCR'd
    for wid, name in zip(target_window_ids, target_window_names):
        try:
            # Use 'import' command (from ImageMagick) to capture window screenshot
            cap_proc = subprocess.run(["import", "-window", wid, "png:-"], capture_output=True, check=True, timeout=10) # Added timeout
            img = Image.open(io.BytesIO(cap_proc.stdout))
            try:
                 # Perform OCR using pytesseract
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
    # Optimization: Removed commented-out print statement below
    # # print(f"Combined text extracted:\n{combined_text[:500]}...")
    return combined_text, processed_window_names # Return text and names of successfully processed windows


class ChatSession:
    """
    Maintains conversation history and augments each query with screen OCR.
    """
    def __init__(self):
        self.history: list[dict[str, str]] = [] # Store as list of message dicts

    def ask(self, prompt: str, use_all_windows: bool) -> str:
        """
        Processes the user prompt, gets screen context, calls the LLM, and updates history.
        """
        print(f"\n--- New Turn ---")
        print(f"User Prompt: {prompt}")
        print(f"Use All Windows: {use_all_windows}")

        # OCR current screen windows (conditionally filtered)
        try:
            screen_text, used_window_names = extract_windows_text(prompt, use_all_windows)
            if not screen_text:
                 print("No screen text extracted.")
            # else:
                 # Optionally print extracted text for debugging, can be very verbose
                 # print(f"Extracted Screen Text (first 500 chars):\n{screen_text[:500]}...")
        except Exception as e:
            print(f"Error extracting window text: {e}")
            traceback.print_exc()
            # Return error message instead of crashing
            return f"Error during screen text extraction: {e}"

        # Build messages: system + past turns + new user turn with screen context
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

        # Add conversation history
        messages.extend(self.history)

        # Add current user prompt with screen context
        user_content = f"Screen Content:\n"
        if screen_text:
            user_content += f"{screen_text}\n\n"
        else:
            user_content += "[No relevant screen content was extracted]\n\n"

        user_content += f"User Prompt: {prompt}"

        messages.append({"role": "user", "content": user_content})

        try:
            completion = client.chat.completions.create(
                model=MODEL,
                temperature=0.3, # Slightly increased temperature for more natural responses
                messages=messages
            )
            answer = completion.choices[0].message.content

        except Exception as e:
             print(f"Error calling OpenAI API: {e}")
             traceback.print_exc()
             return f"Error communicating with AI model: {e}"

        # Add interaction to history
        # Store the full user message including context that was sent to the model
        self.history.append({"role": "user", "content": user_content})
        self.history.append({"role": "assistant", "content": answer})

        # Construct final response string
        final_response = answer
        if not use_all_windows and used_window_names:
            # Prepend the list of used windows if filtering was active and successful
            window_list_str = ", ".join(used_window_names)
            final_response = f"(Info: Used content from windows: {window_list_str})\n\n{answer}"
        elif use_all_windows and used_window_names:
             final_response = f"(Info: Used content from all detected windows)\n\n{answer}"
        elif not use_all_windows and not used_window_names and screen_text: # Model filtered, OCR worked, but model selected none
             final_response = f"(Info: Model did not select specific windows based on the prompt, but screen context was searched.)\n\n{answer}"
        elif not use_all_windows and not used_window_names and not screen_text: # Model filtered, but no windows found/OCR'd
             final_response = f"(Info: No specific windows selected and no screen text was available.)\n\n{answer}"
        # Add other cases? E.g. use_all_windows=True but no windows found/OCR'd
        elif use_all_windows and not used_window_names:
             final_response = f"(Info: Tried using all windows, but none could be processed.)\n\n{answer}"


        print(f"Assistant Response:\n{final_response}")
        return final_response


# Single session instance for the app
session = ChatSession()


def answer_prompt_using_screen(prompt: str, use_all_windows: bool) -> str:
    """
    Public function: returns the session-managed, context-aware answer.
    Accepts the flag to control window usage.
    """
    # Basic input validation
    if not isinstance(prompt, str) or not prompt.strip():
        return "Error: Prompt cannot be empty."
    if not isinstance(use_all_windows, bool):
         # Defaulting if type is wrong, though Gradio should handle this
         print("Warning: Invalid type for use_all_windows, defaulting to False.")
         use_all_windows = False

    return session.ask(prompt.strip(), use_all_windows)
# --- End of Optimized windows_to_text.py ---
