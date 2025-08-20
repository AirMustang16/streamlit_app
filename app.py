import os
import time
import html
from typing import List, Dict, Any

import requests
import streamlit as st
from dotenv import load_dotenv


# --- Config ---
load_dotenv()
DEFAULT_BACKEND_URL = os.getenv("RAG_BACKEND_URL", "https://rag-pipeline-1gmu.onrender.com").rstrip("/")


# --- Helpers ---
def _get_backend_url() -> str:
    url = st.session_state.get("backend_url") or DEFAULT_BACKEND_URL
    return url.rstrip("/")


def _healthcheck(url: str) -> Dict[str, Any]:
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        if resp.ok:
            return resp.json()
        return {"status": "error", "detail": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def _post_query(url: str, query_text: str, top_k: int, history: List[Dict[str, str]]):
    payload = {
        "query": query_text,
        "top_k": top_k,
        "history": history[-6:] if history else None,
    }
    resp = requests.post(f"{url}/query", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _render_citations(citations: List[Dict[str, Any]]):
    if not citations:
        return
    with st.expander("Citations"):
        tooltip = (
            "Practical notes:\n"
            "- The number of citations equals retrieved matches (up to top_k).\n"
            "- rank reflects Pinecone similarity order.\n"
            "- snippet is the chunk text; title, source_url, etc., come from metadata."
        )
        encoded = html.escape(tooltip).replace("\n", "&#10;")
        st.markdown(
            f'<span title="{encoded}">ℹ️</span> '
            f'<span style="color:#6b7280">Hover for citation notes</span>',
            unsafe_allow_html=True,
        )
        for c in citations:
            rank = c.get("rank")
            title = c.get("title") or "Untitled"
            src = c.get("source_url")
            domain = c.get("source_domain")
            date = c.get("date")
            snippet = c.get("snippet")
            header = f"[{rank}] {title}"
            st.markdown(f"**{header}**")
            meta_bits = [b for b in [domain, date] if b]
            if meta_bits:
                st.caption(" • ".join(meta_bits))
            if snippet:
                st.write(snippet)
            if src:
                st.markdown(f"[Open source]({src})")
            st.divider()


def _render_followups(follow_ups: List[str]):
    if not follow_ups:
        return
    st.caption("Try one of these:")
    cols = st.columns(min(3, len(follow_ups)))
    for i, fu in enumerate(follow_ups):
        with cols[i % len(cols)]:
            if st.button(fu, use_container_width=True, key=f"fu_{i}"):
                st.session_state["pending_input"] = fu


# --- Page ---
st.set_page_config(
    page_title="Software Finder Demo",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Minimal, modern styling
st.markdown(
    """
    <style>
    .main {padding-top: 1rem;}
    .stChatFloatingInputContainer {border-top: 1px solid #ececec;}
    .stChatMessage {border-radius: 12px;}
    .stChatMessage[data-testid="stChatMessageUser"] {background-color: #f5f7fb;}
    .stChatMessage[data-testid="stChatMessageAssistant"] {background-color: #ffffff;}
    .block-container {max-width: 880px;}
    .stButton>button {border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Sidebar ---
st.sidebar.subheader("Backend")
if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL

backend_url = st.sidebar.text_input("Backend URL", st.session_state.backend_url, help="FastAPI base URL")
st.session_state.backend_url = backend_url

health = _healthcheck(_get_backend_url())
status = health.get("status", "unknown")
detail = health.get("detail")

if status == "ok":
    st.sidebar.success("API healthy")
elif status == "missing_keys":
    st.sidebar.warning("API reachable — missing keys")
elif status == "error":
    msg = f"API error: {detail}" if detail else "API error"
    st.sidebar.error(msg)
else:
    st.sidebar.warning("API status unknown")
with st.sidebar.expander("Options", expanded=True):
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    st.session_state.top_k = st.slider(
        "Top K",
        1,
        20,
        st.session_state.top_k,
        help=(
            "Default/range: top_k defaults to config.TOP_K (5) and is limited to 1–20.\n\n"
            "Increase top_k:\n"
            "- Pros: Higher recall; more diverse evidence; better for broad/ambiguous queries.\n"
            "- Cons: Slower; higher token costs; more noise can dilute the answer or citations.\n\n"
            "Decrease top_k:\n"
            "- Pros: Faster; cheaper; higher precision; clearer grounding.\n"
            "- Cons: Risk of missing key context; more ‘insufficient context’ cases.\n\n"
            "Guidance:\n"
            "- With precise metadata filters (e.g., author), use 3–5.\n"
            "- For open-ended queries, 5–8 is a good balance; 8–12 only if needed.\n"
            "- Avoid the max unless chunks are very short; more snippets → larger prompt to the LLM."
        ),
    )


# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content}]


st.title("Software Finder Demo")
st.caption("Ask questions about your ingested reviews. Clean, minimal, and fast.")


# --- Chat history ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# --- Input ---
prompt_prefill = st.session_state.pop("pending_input", "") if "pending_input" in st.session_state else ""
user_input = st.chat_input("Ask a question", key="chat_input", max_chars=2000)
if prompt_prefill and not user_input:
    # populate prefill if follow-up was clicked
    st.session_state.chat_input = prompt_prefill
    user_input = prompt_prefill


def _send_and_render(query_text: str):
    st.session_state.messages.append({"role": "user", "content": query_text})
    with st.chat_message("user"):
        st.markdown(query_text)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                start = time.time()
                resp = _post_query(_get_backend_url(), query_text, st.session_state.top_k, st.session_state.messages)
                elapsed = time.time() - start
                answer = resp.get("answer") or "No answer returned."
                citations = resp.get("citations") or []
                follow_ups = resp.get("follow_ups") or []

                placeholder.markdown(answer)
                st.caption(f"Answered in {elapsed:.1f}s")
                _render_citations(citations)
                _render_followups(follow_ups)

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                placeholder.error(f"Request failed: {e}")


if user_input:
    _send_and_render(user_input)


