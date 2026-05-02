"""
BIS RAG Indexer
Embeds chunks using sentence-transformers and builds a FAISS index for retrieval.
"""

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path


MODEL_NAME = "all-MiniLM-L6-v2"


def build_chunk_text(chunk: dict) -> str:
    """Create rich text for embedding: combine title and content."""
    title = chunk.get("title", "")
    content = chunk.get("content", "")
    # Weight title heavily by repeating it
    return f"{title}\n{title}\n{content[:1500]}"


def build_index(chunks_path: str, index_path: str, embeddings_path: str):
    print(f"[1/4] Loading chunks from {chunks_path}")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"      -> {len(chunks)} chunks loaded")

    print(f"[2/4] Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [build_chunk_text(c) for c in chunks]

    print("[3/4] Generating embeddings (this may take a minute)...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"      -> Embeddings shape: {embeddings.shape}")

    # Save embeddings
    np.save(embeddings_path, embeddings)

    print("[4/4] Building FAISS index (IndexFlatIP for cosine similarity)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)

    print(f"      -> Index saved to {index_path}")
    print(f"      -> Total vectors in index: {index.ntotal}")
    return index, chunks, embeddings


if __name__ == "__main__":
    build_index(
        chunks_path="data/chunks.pkl",
        index_path="data/faiss_index.bin",
        embeddings_path="data/embeddings.npy",
    )
