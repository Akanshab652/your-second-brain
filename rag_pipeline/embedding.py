# rag_pipeline/embedding.py
from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: List):
        chunks = []

        for doc in documents:
            text = doc.page_content if hasattr(doc, "page_content") else str(doc)
            start = 0

            while start < len(text):
                end = start + self.chunk_size
                chunks.append(text[start:end])
                start = end - self.chunk_overlap

        return chunks

    def embed_chunks(self, chunks: List[str]):
        return self.model.encode(chunks, show_progress_bar=True)
