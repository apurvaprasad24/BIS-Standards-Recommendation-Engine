# 🏗️ BIS Standards Recommendation Engine

**AI-powered RAG system for BIS SP 21 (2005) — Building Materials Compliance**

> Hackathon Submission · Bureau of Indian Standards × Sigma Squad · May 2026

---

## 🎯 Overview

Indian MSEs often spend weeks identifying which BIS standards apply to their products. This engine turns a plain-language product description into accurate IS standard recommendations **in under 0.1 seconds**, using a hybrid BM25 + TF-IDF retrieval pipeline grounded entirely in the official BIS SP 21 (2005) document.

### ✅ Public Test Set Results
| Metric | Score | Target |
|--------|-------|--------|
| Hit Rate @3 | **100.00%** | >80% |
| MRR @5 | **0.8500** | >0.7 |
| Avg Latency | **0.01s** | <5s |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Query Expansion (domain synonyms)
    │
    ├──► BM25 Retrieval (60%)  ──┐
    │                             ├──► Hybrid Score → Top-K Results
    └──► TF-IDF Retrieval (40%) ─┘
                │
                ▼
       IS Number Exact-Match Boost
                │
                ▼
      (Optional) Claude LLM Re-ranking
                │
                ▼
       Top 5 BIS Standards + Rationale
```

### Key Design Decisions
- **Hybrid BM25 + TF-IDF**: BM25 handles sparse keyword matching (IS numbers, specific terms); TF-IDF captures n-gram semantic similarity. Combined score outperforms either alone.
- **IS Number Boosting**: When a query mentions a specific IS number, exact numeric matches are boosted to prevent partial-number confusion (e.g., IS 269 vs IS 12269).
- **Domain Query Expansion**: Material-specific synonym mapping expands queries with relevant domain terms before retrieval.
- **Full Content Chunking**: Each IS standard is extracted as a self-contained chunk with its title repeated for keyword weighting, achieving better MRR without neural embeddings.

---

## 📦 Setup

### Requirements
```bash
pip install -r requirements.txt
```

### One-time Index Build
```bash
# Put dataset.pdf in the data/ folder, then:
python setup.py
```

### Run Inference (for judges)
```bash
python inference.py --input hidden_private_dataset.json --output team_results.json
```

### Run Evaluation
```bash
python eval_script.py --results team_results.json
```

### Launch Web UI
```bash
python app.py
# Open http://localhost:5000
```

---

## 📁 Repository Structure

```
├── inference.py         
├── setup.py              # parsing + index build
├── app.py                # Flask web UI
├── eval_script.py        # evaluation script
├── requirements.txt      # Dependencies
├── README.md             # This file
├── src/
│   ├── parse_pdf.py      # BIS SP 21 PDF parser & chunker
│   ├── retriever.py      # Hybrid BM25+TF-IDF retriever
│   └── pipeline.py       # End-to-end RAG pipeline
└── data/
    ├── dataset.pdf        # BIS SP 21 source document
    ├── public_test_set.json
    ├── results_v2.json    # Public test set results
    └── chunks.pkl         # Parsed standard chunks
    └── retriever.pkl      # Built retrieval index 
```

---

## 🔧 Technical Details

### Chunking Strategy
- Each IS standard summary extracted as one chunk using `SUMMARY OF IS XXXX` page headers as delimiters
- Chunk text = `IS_ID (×2) + Title (×2) + Full summary content (up to 2500 chars)`
- Title repetition increases TF-IDF weight for standard name keywords
- Table of Contents mined for any standards missed by summary extraction

### Retrieval Pipeline
1. Query expansion via domain synonym dictionary
2. BM25Okapi scoring (60% weight)
3. TF-IDF cosine similarity (40% weight, trigram features)
4. Score normalization + hybrid fusion
5. IS-number exact match boosting (+0.3 for numeric match)
6. Deduplication and top-K selection
---

## 🎯 Impact on MSEs

- **Time savings**: Weeks → Seconds for standard identification
- **Zero hallucinations**: All output grounded in official SP 21 document
- **Fully offline core**: BM25+TF-IDF runs on CPU, no GPU required
- **Low cost**: Retrieval step uses no external APIs
