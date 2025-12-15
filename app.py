import streamlit as st
import os
import tempfile
import json
import requests
from typing import List
from openai import OpenAI
import re
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
    safe_history = []
    for h in history:
        safe_history.append({
            "user": redact_pii(h["user"]),
            "bot": redact_pii(h["bot"])
        })

    with open(HISTORY_FILE, "w") as f:
        json.dump(safe_history, f, indent=2)


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
    question = redact_pii(question)
    answer = redact_pii(answer)
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
    memory_text = redact_pii(memory_text)

    # 🚫 Block storing if still contains redacted markers
    if "[REDACTED_" in memory_text:
        log("🚫 Memory blocked due to PII")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(memory_text.encode("utf-8"))
        path = f.name

    ingest_paths([path])
    store.load()


# =====================================================
# AGENT LOOP - FIXED
# =====================================================
def chat_with_brain(question, llm_client, store, history, top_k=3):

    # 🚫 0️⃣ HARD BLOCK PII INTENT (FAIL FAST)
    if PII_INTENT_PATTERN.search(question):
        log("🚫 PII request blocked before LLM call")
        return "I can’t help with personal contact or identity details."

    # 1️⃣ Vector search
    question = redact_pii(question)
    results = store.query(question, top_k=top_k)

    context_chunks = []
    for r in results:
        text = r.get("page_content") or r.get("metadata", {}).get("text", "")
        if text and len(text.strip()) > 30:
            context_chunks.append(text)

    context = "\n\n".join(context_chunks)
    is_doc_question = len(context_chunks) > 0

    # 2️⃣ Answer using documents
    if is_doc_question:
        prompt = f"""
Answer ONLY using the uploaded documents.

Rules:
- NEVER provide personal contact details
- NEVER guess phone numbers or emails
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

    # 3️⃣ Web fallback
    if answer == "NOT_FOUND" or "Not available in uploaded documents" in answer:
        log("🌐 Document insufficient → searching web...")
        web_text = web_search(question)

        if is_web_data_useless(web_text):
            response = llm_client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": question}]
            )
            answer = response.choices[0].message.content.strip()
        else:
            answer = answer_from_web(llm_client, question, web_text)

    # 🚫 4️⃣ FAIL-FAST OUTPUT GUARDRAIL (MOST IMPORTANT)
    if contains_contact_pii(answer):
        log("🚫 Contact PII detected in model output")
        return "I can’t share personal contact details."


    # 5️⃣ Save safe history
    history.append({
        "user": question,
        "bot": redact_pii(answer)
    })
    save_history(history)

    # 6️⃣ Learn only if safe + non-document
    if not is_doc_question:
        memory = extract_memory(llm_client, question, answer)
        if memory:
            save_memory_to_store(memory, store)

    return answer

# =====================================================
# GUARDRAILS – PII DETECTION (FAIL-FAST)
# =====================================================

PII_INTENT_PATTERN = re.compile(
    r"\b(phone number|mobile number|contact number|email id|address|aadhaar|pan)\b",
    re.IGNORECASE
)


def contains_contact_pii(text: str) -> bool:
    if not text:
        return False

    # Strong signals only
    patterns = [
        PII_PATTERNS["PHONE"],
        PII_PATTERNS["EMAIL"],
        PII_PATTERNS["AADHAAR"],
        PII_PATTERNS["PAN"]
    ]

    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


# =====================================================
# GUARDRAILS – PII REDACTION
# =====================================================

PII_PATTERNS = {
    "PHONE": r"(\+?\d{1,3}[- ]?)?\d{10}",
    "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "AADHAAR": r"\b\d{4}\s\d{4}\s\d{4}\b",
    "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
}

def redact_pii(text: str) -> str:
    if not text:
        return text

    redacted = text
    for label, pattern in PII_PATTERNS.items():
        redacted = re.sub(pattern, f"[REDACTED_{label}]", redacted, flags=re.IGNORECASE)

    return redacted

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
