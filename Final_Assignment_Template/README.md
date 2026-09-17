---
title: Unit 4 GAIA Agent
emoji: "🧭"
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# Unit 4 GAIA Agent

A public Gradio Space for the Hugging Face Agents Course Unit 4 final assignment. It
retrieves the official 20 GAIA level-1 tasks, gives a tool-using agent safe access to
task attachments and public web research, normalizes the requested exact-match answer,
and posts the result to the course scorer.

## Deploy

1. Create a **public Gradio Space** and upload this directory's contents.
2. In the Space **Settings → Variables and secrets**, add `HF_TOKEN` as a **secret**.
   It must be a Hugging Face token allowed to use the selected inference model.
3. Optional variables: `HF_MODEL_ID` (defaults to
   `Qwen/Qwen2.5-Coder-32B-Instruct`) and `AGENT_MAX_STEPS` (defaults to `12`).
4. Open the Space, sign in with the Hugging Face login button, and select **Solve and
   submit all questions**.

`SPACE_ID` is supplied by Hugging Face in a deployed Space and is used to submit the
public `.../tree/main` code URL.

## Run locally

Local runs do not require Gradio OAuth. Create `Final_Assignment_Template/.env`
(already ignored by Git) and add a newly generated token:

```bash
HF_TOKEN=hf_your_new_token
```

The application loads that file on startup, so no editor terminal setting is required.
Then install dependencies and start the application:

```bash
python3 -m pip install --user -r requirements.txt
python3 app.py
```

The local interface asks for a username instead of displaying the Hugging Face login
button. To submit from this checkout, add `AGENT_CODE_URL` with the public Hugging Face
Space `.../tree/main` URL to the same `.env` file; otherwise, deploy the Space and
submit there.

## Safety and behavior

- `HF_TOKEN` is read only from a Space secret or Git-ignored local `.env` file and is
  never rendered, logged, or included in a tool result.
- The file tool can access only the attachment for the task being solved, enforces a
  25 MB limit, and saves non-text files in a private temporary directory.
- The web-page tool accepts only public HTTP(S) URLs, blocks loopback hosts, and limits
  downloads. The agent is instructed to treat task, attachment, and web content as
  untrusted data rather than executable instructions.
- The scorer uses exact matching. The agent receives explicit formatting instructions
  and the app removes only an accidental leading `Answer:` / `Final answer:` label.

The scorer endpoints are `GET /questions`, `GET /random-question`, `GET
/files/{task_id}`, and `POST /submit` at
`https://agents-course-unit4-scoring.hf.space`.
