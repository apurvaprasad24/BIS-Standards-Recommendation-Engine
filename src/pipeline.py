"""
BIS RAG Pipeline
Retriever + Claude API for re-ranking and rationale generation.
"""

import json
import time
import pickle
import re
import urllib.request
import urllib.error
from pathlib import Path

from src.retriever import BISRetriever


# ─── Formatting helpers ───────────────────────────────────────────────────────

def format_std_id(raw_id: str) -> str:
    """
    Convert normalized IS id back to the expected output format.
    e.g. "IS 383: 1970" -> "IS 383: 1970"
         "IS 1489 (Part 2): 1991" -> "IS 1489 (Part 2): 1991"
    """
    # Normalize spacing around colon
    s = re.sub(r'\s*:\s*', ': ', raw_id).strip()
    return s


def normalize_for_eval(s: str) -> str:
    """Match the eval_script normalization: remove spaces, lowercase."""
    return str(s).replace(" ", "").lower()


# ─── Anthropic API call ───────────────────────────────────────────────────────

def call_claude(prompt: str, system: str = "", max_tokens: int = 800) -> str:
    """Call Anthropic API and return text response."""
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"]
    except Exception as e:
        return f"ERROR: {e}"


# ─── Core pipeline ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a BIS (Bureau of Indian Standards) compliance expert specializing in building materials under SP 21 (2005).

Your task: Given a product description and a list of retrieved IS standards (with their summaries), select and rank the TOP 5 most relevant standards.

RULES:
1. Only recommend standards from the provided list. NEVER invent or hallucinate new IS numbers.
2. Return ONLY a JSON array of IS standard IDs in ranked order (most relevant first).
3. The IDs must exactly match the format given in the list (e.g., "IS 383: 1970", "IS 1489 (Part 2): 1991").
4. If fewer than 5 are relevant, return only the relevant ones.
5. Do not add any explanation outside the JSON array.

Output format (strictly):
["IS XXX: YYYY", "IS XXX: YYYY", ...]
"""


def rerank_with_llm(query: str, candidates: list[dict]) -> list[str]:
    """Use Claude to re-rank retrieved candidates and return ordered IS IDs."""
    # Build candidate list for prompt
    cand_str = ""
    for i, c in enumerate(candidates, 1):
        cand_str += f"\n{i}. {c['std_id_raw']}\n   Title: {c['title'][:120]}\n   Summary: {c['content'][:400]}\n"

    prompt = f"""Product Query: {query}

Retrieved IS Standards (candidates):
{cand_str}

Return a JSON array of the top 5 most relevant IS standard IDs from the list above, ranked by relevance.
Only use IDs from the list. Output only the JSON array."""

    response = call_claude(prompt, system=SYSTEM_PROMPT, max_tokens=300)

    # Parse the JSON array from response
    try:
        # Find JSON array in response
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            ids = json.loads(match.group(0))
            return [str(i).strip() for i in ids]
    except Exception:
        pass

    # Fallback: extract IS IDs from text
    ids = re.findall(r'IS\s+[\d]+(?:\s*\(Part\s*\d+\))?\s*:\s*\d{4}', response)
    return ids


def run_pipeline(
    query: str,
    retriever: BISRetriever,
    top_k: int = 5,
    use_llm_rerank: bool = True,
) -> dict:
    """Full RAG pipeline for a single query."""
    t0 = time.time()

    # Step 1: Retrieve candidates (get more than needed for re-ranking)
    candidates = retriever.retrieve(query, top_k=10)

    # Step 2: LLM re-ranking (optional, adds quality but costs latency)
    if use_llm_rerank and candidates:
        ranked_ids = rerank_with_llm(query, candidates)
        # Build ordered result, filling from retrieval if LLM gives fewer
        seen = set()
        final_ids = []
        for rid in ranked_ids:
            norm = normalize_for_eval(rid)
            if norm not in seen:
                seen.add(norm)
                final_ids.append(rid)

        # Fill remaining from retrieval order
        for c in candidates:
            if len(final_ids) >= top_k:
                break
            norm = normalize_for_eval(c["std_id_raw"])
            if norm not in seen:
                seen.add(norm)
                final_ids.append(c["std_id_raw"])
    else:
        # Pure retrieval
        final_ids = [format_std_id(c["std_id_raw"]) for c in candidates[:top_k]]

    latency = time.time() - t0
    return {
        "retrieved_standards": final_ids[:top_k],
        "latency_seconds": round(latency, 3),
    }


# ─── Load retriever singleton ─────────────────────────────────────────────────

_retriever = None


def get_retriever(path: str = "data/retriever.pkl") -> BISRetriever:
    global _retriever
    if _retriever is None:
        _retriever = BISRetriever.load(path)
    return _retriever


if __name__ == "__main__":
    # Quick smoke test
    r = get_retriever()
    query = "We manufacture 33 Grade Ordinary Portland Cement. Which BIS standard applies?"
    result = run_pipeline(query, r, use_llm_rerank=True)
    print("Query:", query)
    print("Result:", json.dumps(result, indent=2))
