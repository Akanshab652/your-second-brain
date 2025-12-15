from evals.run_evals import run_evals
from app import chat_with_brain, init_gemini_client
from rag_pipeline.vector_store import FaissVectorStore

llm = init_gemini_client()
store = FaissVectorStore("faiss_store")
store.load()

run_evals(chat_with_brain, llm, store)
