# agents.py
import os
import openai
from pydantic_ai import Agent, Prompt
from memory import MemoryStore
from rag import VectorStore

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    openai.api_key = openai_api_key

# Light-weight Pydantic-AI usage to ensure typed I/O (keeps code small)
research_agent = Agent()
synth_agent = Agent()

# Memory agent uses MemoryStore class
memory = MemoryStore()
vectorstore = memory.vs  # shared vectorstore

def research(query: str) -> str:
    """
    Research agent: ask the LLM to produce concise bullet points relevant to the query.
    """
    prompt = f"Act as a researcher. Give up to 5 concise bullet points on: {query}"
    if openai_api_key:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return resp.choices[0].message.content
    else:
        return "SIMULATED: research results about " + query

def synth(user_query: str, findings: str, retrieved_context: list) -> str:
    """
    Synthesis agent: combines user query + retrieved context + findings to produce a final answer.
    """
    context_text = "\n\n".join([r.get("text", "") for r in retrieved_context]) if retrieved_context else ""
    prompt = f"""
You are a synthesis agent. User query: {user_query}

Context (retrieved from memory / notes):
{context_text}

Research findings:
{findings}

Provide a clear, concise answer and recommended next steps (2-4 bullets).
"""
    if openai_api_key:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        return resp.choices[0].message.content
    else:
        return "SIMULATED: synthesized answer for " + user_query

def memory_store(user: str, query: str, answer: str):
    memory.add_interaction(user, query, answer)
