import os
import gradio as gr
import requests
import inspect
import pandas as pd
from huggingface_hub import login
from soupsieve import match

from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool, HfApiModel, VisitWebpageTool

from smolagents import tool

@tool
def safe_web_search(query: str) -> str:
    """Searches the web safely with fallback queries if DuckDuckGo fails.
    
    Args:
        query: The search keywords (keep concise, 1-3 words).
    """
    search = DuckDuckGoSearchTool()
    
    # Clean query to basic keywords
    keywords = " ".join(query.split()[:3])
    
    try:
        return search(keywords)
    except Exception:
        # Fallback to even simpler query
        try:
            simple_query = query.split()[0]
            return search(simple_query)
        except Exception as e:
            return f"Search failed: {e}. Please use alternative search terms or visit direct URLs."

import pandas as pd
import whisper

@tool
def transcribe_audio(file_path: str) -> str:
    """Transcribes local audio files (.mp3, .wav) to text using Whisper.
    
    Args:
        file_path: The path or name of the audio file.
    """
    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        return f"Audio transcription error: {e}"

@tool
def read_excel_file(file_path: str) -> str:
    """Reads an Excel spreadsheet file into a text/pandas representation.
    
    Args:
        file_path: Path to the .xlsx or .xls file.
    """
    try:
        df = pd.read_excel(file_path)
        return df.to_string()
    except Exception as e:
        return f"Excel reading error: {e}"

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        fixed_answer = "This is a default answer."
        print(f"Agent returning fixed answer: {fixed_answer}")
        return fixed_answer

def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # Initializing with native tools + web search
    search_tool = DuckDuckGoSearchTool()
    visit_tool = VisitWebpageTool()


    # Upgrading to a powerful model like Gemini 2.5 Flash, Claude 3.5 Sonnet, or GPT-4o
    model = LiteLLMModel(
        model_id="gemini/gemini-3.6-flash", # Highly popular for this benchmark due to cost & performance
        api_key=os.getenv("GEMINI_API_KEY"),
        num_retries=5,       # Automatically retries on 503/429 errors
        retry_min_wait=2,
        retry_max_wait=10
    )

    agent = CodeAgent(
        tools=[search_tool, visit_tool, safe_web_search, transcribe_audio, read_excel_file],
        model=model,
        max_steps=1, # Gives the agent plenty of reasoning/retry cycles
        additional_authorized_imports=["math", "datetime", "re", "json", "collections", 
        "pandas", "numpy", "bs4", "requests", "PIL", "pdfplumber", "whisper"] # Authorize code libraries for calculations
    )

    # Uses Hugging Face's internal serverless architecture
    model_hf = HfApiModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=os.getenv("HF_TOKEN")  # Ensure you have set this in your environment variables
    )

    # Custom system prompt instructions to FORCE Qwen to output raw answers without JSON/dicts
    custom_system_prompt = """
    You are an expert AI agent that solves tasks using Python code and search tools.
    
    CRITICAL ANSWER FORMATTING RULES:
    1. When providing the final answer, call final_answer(answer=...) with ONLY the exact, raw answer string.
    2. DO NOT format the final answer as a dictionary, JSON object, markdown document, or multi-line key-value structure.
    3. If the answer is a word, number, or short text string, pass ONLY that raw string directly to final_answer().
    """

    agent_hf = CodeAgent(tools=[search_tool], model=model_hf, #web_visit_tool
        max_steps=2,
        additional_authorized_imports=["math",
            "datetime",
            "math",
            "re",
            "json",
            "collections",
            "pandas",
            "numpy",
            "bs4",
            "requests",
            "pdfplumber",
            "openpyxl",
            "whisper",
            "PIL",
            "bs4"
            ]
    )

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = agent # Use the Hugging Face model agent for this example
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    import re

    def clean_gaia_answer(answer: str) -> str:
        """Strips conversational wrappers and formats exact match answers."""
        if not isinstance(answer, str):
            answer = str(answer)
        
        answer = str(answer).strip()

        # Robust regex to extract inner content regardless of whitespace or quotes
        match = re.search(r"final_answer\s*\(\s*(?:answer\s*=\s*)?(['\"]?)(.*?)\1\s*\)\s*$", answer, re.DOTALL)
        if match:
            answer = match.group(2).strip()
        else:
            # Fallback: simple string splitting if regex fails
            if "final_answer(" in answer:
                answer = answer.split("final_answer(", 1)[1].rstrip(")").strip()
                if answer.startswith("answer="):
                    answer = answer[7:].strip()
        
        # Strip final_answer wrapper if left as raw code text
        match = re.search(r"final_answer\((?:answer=)?['\"]?(.*?)['\"]?\)$", answer, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            
        # Strip markdown code fencing
        answer = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", answer).strip()

            # 3. Clean trailing code junk like `")`, `")`, `"]`, etc.
        answer = re.sub(r'[\"\']+\)\`*$', '', answer).strip()
        answer = re.sub(r'^\`*[\"\']+', '', answer).strip()

        # 4. Strip single/double outer quotes
        if (answer.startswith('"') and answer.endswith('"')) or (answer.startswith("'") and answer.endswith("'")):
            answer = answer[1:-1].strip()
        
        # Remove outer quotes
        if (answer.startswith('"') and answer.endswith('"')) or (answer.startswith("'") and answer.endswith("'")):
            answer = answer[1:-1].strip()
            
        return answer

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        # Append prompt instructions directly to the task query
        # final_query = (
        #     f"{question_text}\n\n"
        #     "CRITICAL FORMATTING INSTRUCTIONS:\n"
        #     "- Call final_answer(answer=...) with ONLY the exact, raw answer string.\n"
        #     "- DO NOT format the answer as a dictionary, JSON, markdown section, or key-value pair."
        # )
        
        prompt = (
            f"{question_text}\n\n"
            "STRICT INSTRUCTIONS:\n"
            "1. Use `visit_webpage(url)` for URLs, `web_search(keywords)` ONLY for keyword queries.\n"
            "2. For Excel/CSV/Audio, process them via Python code blocks (`pandas`, `whisper`).\n"
            "3. Output ONLY the raw final value (no explanations, no punctuation) inside `final_answer(answer=...)`.\n"
            "4. Execute Python code to retrieve or compute data. Do NOT return Python code snippets as your final answer.\n"
            "5. For comma-separated lists, do NOT put spaces after commas (e.g., return `b,e`, NOT `b, e`).\n"
            "6. Call `final_answer(answer=...)` with ONLY the exact string/numeric value as soon as derived."
        )
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent.run(prompt)
            submitted_answer = clean_gaia_answer(submitted_answer)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)