# ingest.py
import os
from typing import List

from rag_pipeline.loader import load_file
from rag_pipeline.vector_store import FaissVectorStore


def ingest_paths(paths: List[str], persist_dir: str = "faiss_store"):
    documents = []

    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    documents.extend(load_file(os.path.join(root, file)))
        else:
            documents.extend(load_file(path))

    if not documents:
        raise ValueError("No documents found for ingestion")

    vector_store = FaissVectorStore(persist_dir=persist_dir)

    # Chunking + Embedding + Storing happens INSIDE this call
    vector_store.build_from_documents(documents)

    return {
        "documents_loaded": len(documents),
        "vector_store": persist_dir
    }
