
# NewsURL: News Research Tool

NewsURL is a Streamlit app that lets you paste **news article URLs**, build a **FAISS vector index**, and then ask questions with **retrieved evidence + (optional) LLM answers**.

The app supports:
- **Gemini** (embeddings + answers) via `GOOGLE_API_KEY` / `GEMINI_API_KEY` (using the official `google-genai` client)
- **DeepSeek** (answers) via `DEEPSEEK_API_KEY` (OpenAI-compatible)
- **Local embeddings** via `sentence-transformers`

You must provide **at least one** API key (Gemini or DeepSeek) to use the app.

## Features

- **URL ingestion**: paste one URL per line
- **Indexing**: chunking + embeddings + FAISS
- **Retrieval UI**: shows “Retrieved evidence” with source links
- **Graceful fallback**: if Gemini is rate-limited / quota-exhausted, the app falls back to an evidence-only answer

## Project structure

- `main.py`: Streamlit app
- `requirements.txt`: Python dependencies (for local + Streamlit Community Cloud)
- `.env` (optional, local): store API keys

## Requirements

- Python 3.10+ (Streamlit Community Cloud uses a Linux container)
- Provide **one** of the following:
  - `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) for Gemini
  - `DEEPSEEK_API_KEY` for DeepSeek

## Local setup

Clone and install:

```bash
git clone <your-github-repo-url>.git
cd 2_news_research_tool_project
pip install -r requirements.txt
```

Create a `.env` file in the project root (optional but recommended):

```bash
GOOGLE_API_KEY=your_key_here
```

Or use DeepSeek:

```bash
DEEPSEEK_API_KEY=sk-your_key_here
```

Run:

```bash
streamlit run main.py
```

Open the **Local URL** Streamlit prints (usually `http://localhost:8501`).

## Usage

1. Paste URLs (one per line) in the sidebar.
2. Choose embeddings:
   - **gemini**: requires a valid key
   - **local**: uses `sentence-transformers` (works without a key)
3. Click **Build / Refresh Index**.
4. Go to **Ask** and enter a question.

## Deploy to GitHub + Streamlit Community Cloud

### Push to GitHub

Make sure these files exist in the repo root:
- `main.py`
- `requirements.txt`
- `README.md`

Then push:

```bash
git add .
git commit -m "Update NewsURL app + docs"
git push
```

### Streamlit Community Cloud

1. Go to Streamlit Community Cloud and create a new app.
2. Select your GitHub repo and set:
   - **Main file path**: `main.py`
3. Add **Secrets** (recommended) under App settings:

```toml
GOOGLE_API_KEY="your_key_here"
```

Or DeepSeek:

```toml
DEEPSEEK_API_KEY="sk-your_key_here"
```

4. Deploy.

### Notes for Community Cloud

- If you choose **local embeddings**, the build may take longer because `sentence-transformers` pulls in large ML dependencies.
- If Gemini returns `429 RESOURCE_EXHAUSTED`, the app will show retrieved evidence and provide an evidence-only fallback answer.