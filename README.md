# AI Agent Evaluation Runner

A Gradio application for running an AI agent against the Hugging Face Agents Course Unit 4 evaluation benchmark. The app fetches evaluation questions, uses a tool-enabled `smolagents` agent to generate answers, and submits the results to the scoring service.

## Features

- Hugging Face OAuth login for associating submissions with your account
- Tool-enabled agent with web search, webpage visiting, audio transcription, and Excel file reading
- Gemini support through LiteLLM using a `GEMINI_API_KEY`
- Per-question results table and submission score reporting
- Deployment-ready Hugging Face Space configuration

## Project structure

```text
.
├── Final_Assignment_Template/
│   ├── app.py             # Gradio app and evaluation workflow
│   ├── app_testing.py     # Alternative local testing implementation
│   ├── requirements.txt   # Python dependencies
│   └── README.md          # Hugging Face Space configuration
└── README.md
```

## Prerequisites

- Python 3.10 or later
- A Hugging Face account
- A Gemini API key for the default agent configuration

## Run locally

1. Clone the repository and enter the application directory:

   ```bash
   git clone https://github.com/shreya-s17/hugging-face-project.git
   cd hugging-face-project/Final_Assignment_Template
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows, use `.venv\Scripts\activate`.

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure the required environment variable:

   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   ```

5. Start the application:

   ```bash
   python app.py
   ```

Open the local URL shown by Gradio, sign in with Hugging Face, and select **Run Evaluation & Submit All Answers**.

## Configuration

| Variable | Required | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | Yes for the default agent | Authenticates requests made through LiteLLM to Gemini. |
| `HF_TOKEN` | Required when using the Hugging Face API model | Authenticates `HfApiModel` requests. |
| `SPACE_ID` | Set automatically on Hugging Face Spaces | Builds the public source URL included with a submission. |

Keep API keys and access tokens out of source control. For a Hugging Face Space, add them through **Settings → Variables and secrets** rather than committing them to the repository.

## Deploy to Hugging Face Spaces

The application directory includes Space metadata in `Final_Assignment_Template/README.md`. To deploy:

1. Create a new Gradio Space on Hugging Face.
2. Upload the contents of `Final_Assignment_Template/` to the Space repository root.
3. Add `GEMINI_API_KEY` as a Space secret.
4. Enable Hugging Face OAuth in the Space settings if it is not enabled automatically.

Once the Space is running, log in and run the evaluation from its interface.

## Notes

- Evaluations can take several minutes because each question is processed independently.
- The scoring endpoint is `https://agents-course-unit4-scoring.hf.space`.
- The agent configuration is intentionally easy to customize in `app.py`; adjust tools, model settings, and step limits to experiment with different approaches.
