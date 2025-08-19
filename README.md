# Streamlit RAG Client

A minimal, good-looking chat UI for your RAG FastAPI backend.

## Setup

```bash
cd streamlit_client
pip install -r requirements.txt
```

If your backend runs on a non-default URL, set `RAG_BACKEND_URL` in a `.env` file or via environment variable.

## Run

```bash
streamlit run app.py
```

In the sidebar you can change the backend URL and the Top-K setting.
