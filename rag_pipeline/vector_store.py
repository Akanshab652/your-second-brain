import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from rag_pipeline.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store",
                 embedding_model: str = "all-MiniLM-L12-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 log_func=print):  # allow logging function (Streamlit or print)
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.log = log_func

        self.log(f"[INFO] Creating embedding model '{embedding_model}' (may take a while)...")
        try:
            self.model = SentenceTransformer(embedding_model)
            self.log(f"[INFO] Loaded embedding model: {embedding_model}")
        except Exception as e:
            self.log(f"[ERROR] Failed to load embedding model: {e}")
            raise e

    def build_from_documents(self, documents: List[Any]):
        self.log(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model,
                                     chunk_size=self.chunk_size,
                                     chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk} for chunk in chunks]

        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        self.log(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D numpy array")
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
            self.log(f"[INFO] Initialized FAISS index with dimension {dim}")
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        self.log(f"[INFO] Added {embeddings.shape[0]} vectors to FAISS index")

    def save(self):
        if self.index is None:
            self.log("[WARNING] No index to save")
            return
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        self.log(f"[INFO] Saved FAISS index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        if not os.path.exists(faiss_path) or not os.path.exists(meta_path):
            self.log(f"[WARNING] FAISS index or metadata not found in {self.persist_dir}")
            self.index = None
            self.metadata = []
            return

        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
            print(len(self.metadata))
        self.log(f"[INFO] Loaded FAISS index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None:
            self.log("[WARNING] Search called on empty index")
            return []
        if query_embedding.ndim != 2:
            query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')

        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        self.log(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)
