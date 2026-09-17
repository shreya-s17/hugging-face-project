import os
import time
import requests
import pandas as pd
import gradio as gr
from huggingface_hub import login

from smolagents import CodeAgent, HfApiModel, tool
from duckduckgo_search import DDGS

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


# --- Free Search Tool Wrapper (No API Key Required) ---
import time
from smolagents import tool
from duckduckgo_search import DDGS

import time
from duckduckgo_search import DDGS
from smolagents import tool


import time
from smolagents import tool
from duckduckgo_search import DDGS

@tool
def safe_web_search(query: str) -> str:
    """
    Searches the web safely for factual information.

    Args:
        query: Search keywords or query string.
    """
    time.sleep(2.0)  # Necessary 2-second sleep to prevent IP soft-blocks
    
    clean_query = " ".join(query.replace('"', ' ').replace("'", ' ').split()[:6])

    try:
        results = []
        # Explicitly pass backend='lite' to bypass DDG scraping blocks
        with DDGS() as ddgs:
            search_results = list(ddgs.text(clean_query, max_results=5, backend='lite'))
            if search_results:
                for r in search_results:
                    title = r.get('title', '')
                    snippet = r.get('body', '')
                    results.append(f"Title: {title}\nSnippet: {snippet}\n")
                return "\n".join(results)
            else:
                return f"No results for '{clean_query}'."
    except Exception as e:
        return f"Search error: {e}. Try a simpler search query."

def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Initialize Model
    model_hf = HfApiModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=os.getenv("HF_TOKEN")
    )

    # Initialize CodeAgent with safe_web_search
    agent = CodeAgent(
        tools=[safe_web_search],
        model=model_hf,
        max_steps=12,
        additional_authorized_imports=["math", "datetime", "re", "json", "collections"]
    )

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except Exception as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None

    # 3. Run Agent Loop
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")

    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")

        if not task_id or question_text is None:
            continue

        final_query = (
            f"{question_text}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Call final_answer(answer=...) with ONLY the exact, raw answer string.\n"
            "2. DO NOT format the answer as a dictionary, JSON, markdown section, or key-value pair.\n"
            "3. Keep web search queries short (1-3 words max)."
        )

        try:
            raw_answer = agent.run(final_query)
            submitted_answer = str(raw_answer).strip()

            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare and Submit Answers
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload
    }

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
        return final_status, pd.DataFrame(results_log)
    except Exception as e:
        status_message = f"Submission Failed: {e}"
        return status_message, pd.DataFrame(results_log)


# --- Build Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=False)