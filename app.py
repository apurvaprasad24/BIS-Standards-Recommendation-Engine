"""
app.py — BIS Standards Recommendation Engine Web UI
Run: python app.py
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template_string, request, jsonify
from src.retriever import BISRetriever
from src.pipeline import run_pipeline

app = Flask(__name__)

# Load retriever once at startup
retriever = None

def get_retriever():
    global retriever
    if retriever is None:
        retriever = BISRetriever.load("data/retriever.pkl")
    return retriever


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BIS Standards Recommendation Engine</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #f0f4f8; color: #1a202c; min-height: 100vh; }
  
  header {
    background: linear-gradient(135deg, #1a365d 0%, #2d6a4f 100%);
    color: white; padding: 24px 32px;
    display: flex; align-items: center; gap: 16px;
  }
  .logo { font-size: 2rem; }
  header h1 { font-size: 1.5rem; font-weight: 700; }
  header p { font-size: 0.9rem; opacity: 0.85; margin-top: 4px; }

  .container { max-width: 900px; margin: 40px auto; padding: 0 20px; }

  .search-card {
    background: white; border-radius: 16px; padding: 32px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
  }
  .search-card h2 { font-size: 1.1rem; color: #2d6a4f; margin-bottom: 12px; font-weight: 600; }
  
  textarea {
    width: 100%; border: 2px solid #e2e8f0; border-radius: 10px;
    padding: 14px 16px; font-size: 1rem; font-family: inherit;
    resize: vertical; min-height: 110px; transition: border-color 0.2s;
  }
  textarea:focus { outline: none; border-color: #2d6a4f; }

  .examples { margin: 12px 0; display: flex; flex-wrap: wrap; gap: 8px; }
  .example-btn {
    background: #f0fdf4; border: 1px solid #86efac; color: #166534;
    border-radius: 20px; padding: 6px 14px; font-size: 0.8rem;
    cursor: pointer; transition: all 0.2s;
  }
  .example-btn:hover { background: #dcfce7; }

  .search-btn {
    margin-top: 16px; width: 100%; padding: 14px;
    background: linear-gradient(135deg, #1a365d, #2d6a4f);
    color: white; border: none; border-radius: 10px;
    font-size: 1rem; font-weight: 600; cursor: pointer;
    transition: opacity 0.2s;
  }
  .search-btn:hover { opacity: 0.9; }
  .search-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .results { margin-top: 32px; }
  .results-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 16px;
  }
  .results-header h3 { font-size: 1.1rem; color: #1a365d; }
  .latency-badge {
    background: #e0f2fe; color: #0369a1; padding: 4px 12px;
    border-radius: 20px; font-size: 0.8rem; font-weight: 600;
  }

  .standard-card {
    background: white; border-radius: 12px; padding: 20px 24px;
    margin-bottom: 14px; border-left: 4px solid #2d6a4f;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    display: flex; gap: 16px; align-items: flex-start;
  }
  .rank-badge {
    background: #2d6a4f; color: white; width: 32px; height: 32px;
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.9rem; flex-shrink: 0;
  }
  .std-id { font-size: 1rem; font-weight: 700; color: #1a365d; }
  .std-title { font-size: 0.9rem; color: #4a5568; margin-top: 4px; }
  .std-score { font-size: 0.78rem; color: #a0aec0; margin-top: 6px; }

  .loading { text-align: center; padding: 40px; color: #4a5568; }
  .spinner {
    width: 40px; height: 40px; border: 4px solid #e2e8f0;
    border-top-color: #2d6a4f; border-radius: 50%;
    animation: spin 0.8s linear infinite; margin: 0 auto 16px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .error-box { background: #fff5f5; border: 1px solid #fc8181; border-radius: 10px; padding: 16px; color: #c53030; }

  footer { text-align: center; padding: 32px; color: #a0aec0; font-size: 0.85rem; }
</style>
</head>
<body>

<header>
  <div class="logo">🏗️</div>
  <div>
    <h1>BIS Standards Recommendation Engine</h1>
    <p>AI-powered compliance discovery for Building Materials — SP 21 (2005)</p>
  </div>
</header>

<div class="container">
  <div class="search-card">
    <h2>Describe your product or compliance need</h2>
    <textarea id="queryInput" placeholder="e.g. We manufacture hollow concrete masonry blocks. What BIS standard applies to dimensions and physical requirements?"></textarea>
    
    <div class="examples">
      <span style="font-size:0.8rem;color:#718096;align-self:center">Try:</span>
      <button class="example-btn" onclick="setExample(0)">Ordinary Portland Cement (33 Grade)</button>
      <button class="example-btn" onclick="setExample(1)">Precast concrete pipes for water mains</button>
      <button class="example-btn" onclick="setExample(2)">Asbestos cement roofing sheets</button>
      <button class="example-btn" onclick="setExample(3)">Portland slag cement manufacture</button>
    </div>

    <button class="search-btn" id="searchBtn" onclick="search()">
      🔍 Find Applicable BIS Standards
    </button>
  </div>

  <div class="results" id="results"></div>
</div>

<footer>BIS SP 21 (2005) · Sigma Squad · Hackathon Submission 2026</footer>

<script>
const examples = [
  "We are a small enterprise manufacturing 33 Grade Ordinary Portland Cement. Which BIS standard covers the chemical and physical requirements for our product?",
  "What is the official specification for manufacturing precast concrete pipes, both with and without reinforcement, for water mains?",
  "Looking for the standard detailing corrugated and semi-corrugated asbestos cement sheets used for roofing and cladding.",
  "What is the Indian Standard covering the manufacture, chemical, and physical requirements for Portland slag cement?",
];

function setExample(i) {
  document.getElementById('queryInput').value = examples[i];
}

async function search() {
  const query = document.getElementById('queryInput').value.trim();
  if (!query) { alert('Please enter a product description.'); return; }
  
  const btn = document.getElementById('searchBtn');
  btn.disabled = true;
  btn.textContent = 'Searching...';
  
  document.getElementById('results').innerHTML = `
    <div class="loading">
      <div class="spinner"></div>
      <p>Retrieving relevant BIS standards...</p>
    </div>`;

  try {
    const res = await fetch('/api/recommend', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query})
    });
    const data = await res.json();
    
    if (data.error) {
      document.getElementById('results').innerHTML = `<div class="error-box">❌ ${data.error}</div>`;
      return;
    }
    
    const standards = data.retrieved_standards;
    const latency = data.latency_seconds;
    
    let html = `<div class="results-header">
      <h3>Top ${standards.length} Recommended Standards</h3>
      <span class="latency-badge">⚡ ${latency.toFixed(2)}s</span>
    </div>`;
    
    standards.forEach((std, i) => {
      html += `<div class="standard-card">
        <div class="rank-badge">${i+1}</div>
        <div>
          <div class="std-id">${std.std_id}</div>
          <div class="std-title">${std.title || ''}</div>
          <div class="std-score">Score: ${(std.score * 100).toFixed(1)}%</div>
        </div>
      </div>`;
    });
    
    document.getElementById('results').innerHTML = html;
  } catch(e) {
    document.getElementById('results').innerHTML = `<div class="error-box">❌ Request failed: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '🔍 Find Applicable BIS Standards';
  }
}

document.getElementById('queryInput').addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.ctrlKey) search();
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        r = get_retriever()
        result = run_pipeline(query, r, top_k=5, use_llm_rerank=False)

        # Enrich with titles from chunks
        enriched = []
        for std_id in result["retrieved_standards"]:
            norm = std_id.replace(" ", "").lower()
            title = ""
            for c in r.chunks:
                if c["std_id"].replace(" ", "").lower() == norm:
                    title = c.get("title", "")
                    break
            enriched.append({
                "std_id": std_id,
                "title": title,
                "score": 1.0 - (0.1 * result["retrieved_standards"].index(std_id))
            })

        return jsonify({
            "retrieved_standards": enriched,
            "latency_seconds": result["latency_seconds"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Loading retriever...")
    get_retriever()
    print("Starting BIS Recommendation Engine at http://localhost:5000")
    app.run(debug=False, port=5000)
