"""
BIS RAG Retriever
Hybrid BM25 + TF-IDF retrieval for IS standard recommendation.
No external model downloads required - fully offline.
"""

import pickle
import re
import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


# ─── Text preprocessing ───────────────────────────────────────────────────────

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "on", "at", "by", "from", "with", "and", "or", "but", "if",
    "as", "it", "its", "this", "that", "we", "our", "their", "which",
    "who", "what", "how", "where", "when", "any", "all", "each", "no",
    "not", "only", "also", "than", "more", "most", "such", "other",
    "into", "over", "under", "about", "after", "before", "between", "through",
}

# IS number boost: if query contains "IS XXXX" pattern, boost exact matches
IS_QUERY_PATTERN = re.compile(r'\bIS\s+(\d+(?:\s*\(Part\s*\d+\))?)\b', re.IGNORECASE)

# Domain-specific synonyms to expand queries
DOMAIN_SYNONYMS = {
    "portland cement": ["OPC", "ordinary portland", "pozzolana"],
    "opc": ["ordinary portland cement", "IS 269", "IS 8112", "IS 12269"],
    "steel": ["reinforcement", "TMT", "structural steel", "rebar"],
    "aggregate": ["coarse aggregate", "fine aggregate", "sand", "gravel"],
    "concrete": ["RCC", "reinforced concrete", "precast"],
    "asbestos": ["AC sheets", "corrugated sheets", "roofing sheets"],
    "masonry": ["brick", "block", "mortar"],
    "lime": ["hydrated lime", "building lime", "quicklime"],
    "waterproofing": ["damp proofing", "sealing", "moisture"],
    "gypsum": ["plaster of paris", "plasterboard"],
    "bitumen": ["tar", "asphalt", "bituminous"],
    "glass": ["glazing", "float glass", "window glass"],
    "timber": ["wood", "plywood", "hardwood"],
    "pipe": ["conduit", "piping", "water main"],
}


def tokenize(text: str) -> list[str]:
    """Tokenize and normalize text."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens


def expand_query(query: str) -> str:
    """Expand query with domain synonyms."""
    q_lower = query.lower()
    expansions = []
    for key, synonyms in DOMAIN_SYNONYMS.items():
        if key in q_lower:
            expansions.extend(synonyms)
    if expansions:
        return query + " " + " ".join(expansions)
    return query


def build_chunk_text(chunk: dict) -> str:
    """Build rich text representation of a chunk."""
    title = chunk.get("title", "")
    content = chunk.get("content", "")
    std_id = chunk.get("std_id_raw", chunk.get("std_id", ""))
    # Repeat title and IS ID for better matching
    return f"{std_id} {std_id} {title} {title} {content}"


# ─── Index builder ────────────────────────────────────────────────────────────

class BISRetriever:
    def __init__(self):
        self.chunks = []
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.corpus_tokens = []

    def build(self, chunks: list[dict]):
        self.chunks = chunks
        corpus_texts = [build_chunk_text(c) for c in chunks]

        # Tokenized corpus for BM25
        print("  Building BM25 index...")
        self.corpus_tokens = [tokenize(t) for t in corpus_texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)

        # TF-IDF matrix
        print("  Building TF-IDF index...")
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=50000,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus_texts)
        print(f"  TF-IDF matrix: {self.tfidf_matrix.shape}")

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "bm25": self.bm25,
                "corpus_tokens": self.corpus_tokens,
                "tfidf_vectorizer": self.tfidf_vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
            }, f)
        print(f"  Retriever saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BISRetriever":
        with open(path, "rb") as f:
            data = pickle.load(f)
        r = cls()
        r.chunks = data["chunks"]
        r.bm25 = data["bm25"]
        r.corpus_tokens = data["corpus_tokens"]
        r.tfidf_vectorizer = data["tfidf_vectorizer"]
        r.tfidf_matrix = data["tfidf_matrix"]
        return r

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Hybrid BM25 + TF-IDF retrieval with query expansion and IS-number boosting."""
        expanded = expand_query(query)

        # BM25 scores
        q_tokens = tokenize(expanded)
        bm25_scores = np.array(self.bm25.get_scores(q_tokens))

        # TF-IDF scores
        q_vec = self.tfidf_vectorizer.transform([expanded])
        tfidf_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        # Normalize scores to [0,1]
        def normalize(arr):
            mx = arr.max()
            if mx == 0:
                return arr
            return arr / mx

        bm25_norm = normalize(bm25_scores)
        tfidf_norm = normalize(tfidf_scores)

        # Hybrid: 60% BM25 + 40% TF-IDF
        combined = 0.6 * bm25_norm + 0.4 * tfidf_norm

        # IS-number exact match boost: if the query itself contains "IS XXXX",
        # penalize chunks whose IS number doesn't match (prevents IS 12269 > IS 269)
        is_nums_in_query = IS_QUERY_PATTERN.findall(query)
        if is_nums_in_query:
            query_is_nums = set(n.replace(" ", "").lower() for n in is_nums_in_query)
            for idx, chunk in enumerate(self.chunks):
                chunk_num = re.search(r'\b(\d+(?:\(Part\d+\))?)\b', chunk["std_id"].replace(" ", ""))
                if chunk_num:
                    cn = chunk_num.group(1).lower()
                    if cn in query_is_nums:
                        combined[idx] += 0.3  # strong boost for exact IS number match

        # Get top indices
        top_indices = np.argsort(combined)[::-1][:top_k * 2]

        # Deduplicate by std_id
        seen_ids = set()
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            std_id = chunk["std_id"]
            if std_id not in seen_ids:
                seen_ids.add(std_id)
                results.append({
                    **chunk,
                    "score": float(combined[idx]),
                    "bm25_score": float(bm25_norm[idx]),
                    "tfidf_score": float(tfidf_norm[idx]),
                })
            if len(results) >= top_k:
                break

        return results


# ─── Build index script ───────────────────────────────────────────────────────

def build_retriever(chunks_path: str, retriever_path: str):
    print(f"[1/3] Loading chunks...")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"      -> {len(chunks)} chunks")

    print("[2/3] Building retriever...")
    r = BISRetriever()
    r.build(chunks)

    print("[3/3] Saving retriever...")
    r.save(retriever_path)
    print("Done!")
    return r


if __name__ == "__main__":
    build_retriever(
        chunks_path="data/chunks.pkl",
        retriever_path="data/retriever.pkl",
    )
