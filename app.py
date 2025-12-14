import streamlit as st
import os
import tempfile
import json
import requests
from typing import List
from openai import OpenAI

from rag_pipeline.vector_store import FaissVectorStore
from rag_pipeline.ingest import ingest_paths

# =====================================================
# Helper
# =====================================================
def log(msg):
    st.info(msg)
    print(msg)

# =====================================================
# Persistent Files
# =====================================================
HISTORY_FILE = "chat_history.json"
INGESTED_FILES_DB = "ingested_files.json"

# ---------------------
# Chat history
# ---------------------
def load_history() -> List[dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return []
    return []

def save_history(history: List[dict]):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

# ---------------------
# Ingested files registry
# ---------------------
def load_ingested_files() -> set:
    if os.path.exists(INGESTED_FILES_DB):
        try:
            with open(INGESTED_FILES_DB, "r") as f:
                data = json.load(f)
                return set(data)
        except (json.JSONDecodeError, ValueError):
            # Reset empty or invalid JSON
            with open(INGESTED_FILES_DB, "w") as f:
                json.dump([], f)
            return set()
    return set()

def save_ingested_files(files: set):
    with open(INGESTED_FILES_DB, "w") as f:
        json.dump(list(files), f, indent=2)

# =====================================================
# Gemini Client
# =====================================================
def init_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY environment variable not set!")
        st.stop()
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

# =====================================================
# Web Search (best effort)
# =====================================================
def web_search(query: str) -> str:
    url = "https://duckduckgo.com/html/"
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"q": query}
    try:
        r = requests.post(url, data=params, headers=headers, timeout=10)
        text = r.text
    except Exception:
        return ""
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 60]
    return "\n".join(lines[:30])

def is_web_data_useless(text: str) -> bool:
    if not text:
        return True
    bad_signals = ["<html", "<form", "DuckDuckGo", "search input", "homepage"]
    return any(bad.lower() in text.lower() for bad in bad_signals)

def answer_from_web(llm_client, question: str, web_text: str) -> str:
    prompt = f"""
Answer the question using the web data below.
Be concise and factual.
Do NOT mention HTML or search engines.

Web data:
{web_text}

Question:
{question}
"""
    response = llm_client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# =====================================================
# Memory Extraction
# =====================================================
def extract_memory(llm_client, question: str, answer: str) -> str | None:
    prompt = f"""
Extract reusable factual knowledge.

Conversation:
User: {question}
Assistant: {answer}

Rules:
- One fact per line
- No opinions
- If nothing useful return NONE
"""
    response = llm_client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    memory = response.choices[0].message.content.strip()
    return None if memory == "NONE" else memory

def save_memory_to_store(memory_text: str, store: FaissVectorStore):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(memory_text.encode("utf-8"))
        path = f.name
    ingest_paths([path])
    store.load()

# =====================================================
# AGENT LOOP - FIXED
# =====================================================
def chat_with_brain(question, llm_client, store, history, top_k=3):
    # 1️⃣ Vector search
    results = store.query(question, top_k=top_k)
    context_chunks = []
    for r in results:
        text = r.get("page_content") or r.get("metadata", {}).get("text", "")
        if text and len(text.strip()) > 30:
            context_chunks.append(text)
    context = "\n\n".join(context_chunks)
    is_doc_question = len(context_chunks) > 0

    # 2️⃣ Answer using documents if available
    if is_doc_question:
        prompt = f"""
Answer ONLY using the uploaded documents.

Rules:
- Do NOT use web knowledge
- Do NOT switch to other entities
- If info is missing, say "Not available in uploaded documents"

Context:
{context}

Question:
{question}
"""
        response = llm_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()
    else:
        answer = "NOT_FOUND"

    # 3️⃣ Web fallback only if not document-based
    if answer == "NOT_FOUND" and not is_doc_question:
        log("🌐 Memory insufficient → searching web...")
        web_text = web_search(question)
        if is_web_data_useless(web_text):
            response = llm_client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": question}]
            )
            answer = response.choices[0].message.content.strip()
        else:
            answer = answer_from_web(llm_client, question, web_text)

    # 4️⃣ Save UI history
    history.append({"user": question, "bot": answer})
    save_history(history)

    # 5️⃣ Learn only for non-document questions
    if not is_doc_question:
        memory = extract_memory(llm_client, question, answer)
        if memory:
            save_memory_to_store(memory, store)

    return answer

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="Second Brain", layout="centered")
st.title("🧠 Your Second Brain")

# =====================================================
# Init Brain
# =====================================================
if "brain" not in st.session_state:
    llm_client = init_gemini_client()
    store = FaissVectorStore(persist_dir="faiss_store")
    if os.path.exists("faiss_store/faiss.index"):
        store.load()
    st.session_state.brain = {"llm_client": llm_client, "store": store}

llm_client = st.session_state.brain["llm_client"]
store = st.session_state.brain["store"]
history = load_history()

# =====================================================
# File Ingestion (ONE-TIME)
# =====================================================
st.subheader("📥 Upload documents (one-time only)")
uploaded_files = st.file_uploader(
    "Already ingested files will be skipped",
    type=["txt", "pdf", "docx", "md"],
    accept_multiple_files=True
)

if st.button("Ingest files"):
    ingested = load_ingested_files()
    new_paths = []
    new_files = []
    for file in uploaded_files or []:
        if file.name in ingested:
            log(f"⏭️ Skipped: {file.name}")
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
            tmp.write(file.read())
            new_paths.append(tmp.name)
            new_files.append(file.name)
    if new_paths:
        ingest_paths(new_paths)
        store.load()
        ingested.update(new_files)
        save_ingested_files(ingested)
        st.success(f"Ingested {len(new_files)} new file(s)")
    else:
        st.info("No new files to ingest")

# =====================================================
# Chat UI
# =====================================================
st.subheader("❓ Ask your brain")
question = st.text_input("Type your question")

col1, col2 = st.columns(2)
with col1:
    if st.button("Ask") and question.strip():
        answer = chat_with_brain(question, llm_client, store, history)
        st.markdown("### 📌 Answer")
        st.write(answer)
with col2:
    if st.button("🧹 Clear Chat"):
        clear_history()
        st.success("Chat cleared. Refreshing...")
        st.rerun()

# =====================================================
# History
# =====================================================
if history:
    st.subheader("💬 Recent Chats")
    for chat in history[-10:]:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown("---")
