# second_brain.py

# second_brain.py

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document

import re
import os
from typing import List

# -------------------
# Guardrail function
# -------------------
def remove_pii(text: str) -> str:
    text = re.sub(r'\S+@\S+\.\S+', '[REDACTED_EMAIL]', text)
    text = re.sub(r'\b\d{10}\b', '[REDACTED_PHONE]', text)
    return text

# -------------------
# Agents
# -------------------
class ResearchAgent:
    def fetch(self, sources: List[str]) -> List[str]:
        data = []
        for s in sources:
            if os.path.exists(s):
                with open(s, 'r', encoding='utf-8') as f:
                    data.append(f.read())
        return data

class SynthesisAgent:
    def summarize(self, texts: List[str]) -> str:
        combined = " ".join(texts)
        combined = remove_pii(combined)
        return combined[:1000]

class LearningAgent:
    def __init__(self, vector_store: FAISS, llm: ChatOpenAI):
        self.vector_store = vector_store
        self.llm = llm

    def add_to_memory(self, text: str):
        doc = Document(page_content=text)
        self.vector_store.add_documents([doc])

    def query(self, question: str) -> str:
        docs = self.vector_store.similarity_search(question, k=3)
        context = " ".join([d.page_content for d in docs])
        return self.llm(f"Answer using this context: {context}\nQuestion: {question}")

# -------------------
# Main Second Brain
# -------------------
class SecondBrain:
    def __init__(self):
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")

        self.llm = ChatOpenAI(temperature=0)
        self.embeddings = OpenAIEmbeddings()

        # Initialize FAISS vector store from empty list
        self.vector_store = FAISS.from_documents([], self.embeddings)

        self.research_agent = ResearchAgent()
        self.synthesis_agent = SynthesisAgent()
        self.learning_agent = LearningAgent(self.vector_store, self.llm)

    def ingest_sources(self, sources: List[str]):
        raw_data = self.research_agent.fetch(sources)
        if raw_data:
            summary = self.synthesis_agent.summarize(raw_data)
            self.learning_agent.add_to_memory(summary)

    def ask(self, question: str) -> str:
        return self.learning_agent.query(question)

# -------------------
# CLI Interface
# -------------------
if __name__ == "__main__":
    brain = SecondBrain()
    sources = ["notes.txt", "meeting_transcript.txt"]
    brain.ingest_sources(sources)

    print("Welcome to your Second Brain! Type 'exit' to quit.")
    while True:
        user_input = input(">> ")
        if user_input.lower() == "exit":
            break
        response = brain.ask(user_input)
        print("📌 Answer:", response)
