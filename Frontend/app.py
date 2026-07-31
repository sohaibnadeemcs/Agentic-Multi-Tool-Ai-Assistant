"""
Streamlit frontend for the Agentic Multi-Tool AI Assistant.

Features:
  - Chat interface backed by the FastAPI /chat streaming endpoint
  - PDF upload widget (feeds the rag_qa agent)
  - Inline image rendering for image_gen results
  - Sidebar showing which agent handled each turn
"""
import json
import os
import re
import uuid

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Agentic Multi-Tool AI Assistant", page_icon="🧠", layout="wide")

# --- Session state setup -----------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, intent}
if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = []

IMAGE_MARKDOWN_RE = re.compile(r"!\[[^\]]*\]\((?P<url>[^)]+)\)")

INTENT_LABELS = {
    "research": "🔎 Research",
    "code_fix": "🐛 Code Fix",
    "summarize": "📝 Summarize",
    "image_gen": "🎨 Image Generation",
    "rag_qa": "📄 Document Q&A",
    "general_chat": "💬 General Chat",
}


def render_message_content(content: str):
    """Render text, pulling out any markdown image syntax into st.image calls."""
    last_end = 0
    for match in IMAGE_MARKDOWN_RE.finditer(content):
        pre_text = content[last_end:match.start()].strip()
        if pre_text:
            st.markdown(pre_text)
        st.image(match.group("url"))
        last_end = match.end()
    remaining = content[last_end:].strip()
    if remaining:
        st.markdown(remaining)


# --- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Multi-Tool Assistant")
    st.caption(f"Session: `{st.session_state.session_id[:8]}`")

    st.markdown("---")
    st.subheader("📄 Upload a PDF (for rag_qa)")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file is not None and uploaded_file.name not in st.session_state.uploaded_pdfs:
        with st.spinner(f"Indexing {uploaded_file.name}..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"session_id": st.session_state.session_id}
                resp = requests.post(f"{BACKEND_URL}/upload_pdf", files=files, data=data, timeout=60)
                resp.raise_for_status()
                result = resp.json()
                st.session_state.uploaded_pdfs.append(uploaded_file.name)
                st.success(result["message"])
            except Exception as e:
                st.error(f"Failed to index PDF: {e}")

    if st.session_state.uploaded_pdfs:
        st.markdown("**Indexed documents:**")
        for name in st.session_state.uploaded_pdfs:
            st.markdown(f"- {name}")

    st.markdown("---")
    if st.button("🗑️ New session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.uploaded_pdfs = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "Agents: research · code_fix · summarize · image_gen · rag_qa · general_chat\n\n"
        "The supervisor auto-classifies your message. Mention 'the PDF' / "
        "'the document' to route to rag_qa after uploading a file."
    )


# --- Main chat area --------------------------------------------------
st.title("Agentic Multi-Tool AI Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("intent"):
            st.caption(INTENT_LABELS.get(msg["intent"], msg["intent"]))
        render_message_content(msg["content"])

user_input = st.chat_input("Ask a question, paste code to fix, request an image, or query your PDF...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        intent_placeholder = st.empty()
        content_placeholder = st.empty()
        full_text = ""
        current_intent = None

        try:
            with requests.post(
                f"{BACKEND_URL}/chat",
                json={"session_id": st.session_state.session_id, "message": user_input},
                stream=True,
                timeout=180,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    payload = json.loads(line[len("data: "):])

                    if payload["type"] == "intent":
                        current_intent = payload["intent"]
                        intent_placeholder.caption(
                            INTENT_LABELS.get(current_intent, current_intent)
                        )
                    elif payload["type"] == "token":
                        full_text += payload["content"]
                        with content_placeholder.container():
                            render_message_content(full_text)
                    elif payload["type"] == "error":
                        full_text += f"\n\n⚠️ Error: {payload['detail']}"
                        with content_placeholder.container():
                            render_message_content(full_text)
                    elif payload["type"] == "done":
                        break
        except Exception as e:
            full_text = f"⚠️ Failed to reach backend: {e}"
            with content_placeholder.container():
                st.error(full_text)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_text, "intent": current_intent}
        )
