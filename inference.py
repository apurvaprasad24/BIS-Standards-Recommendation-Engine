import argparse
import json
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retriever import BISRetriever
from src.pipeline import run_pipeline


def load_retriever():
    retriever_path = os.path.join(os.path.dirname(__file__), "data", "retriever.pkl")
    if not os.path.exists(retriever_path):
        print(f"[ERROR] Retriever not found at {retriever_path}")
        print("Please run: python setup.py  (to build the index from the PDF)")
        sys.exit(1)
    print(f"[INFO] Loading retriever from {retriever_path}...")
    return BISRetriever.load(retriever_path)


def process_dataset(input_path: str, output_path: str):
    # Load input
    print(f"[INFO] Loading input: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[INFO] {len(dataset)} queries to process")

    # Load retriever
    retriever = load_retriever()

    results = []
    total_start = time.time()

    for i, item in enumerate(dataset):
        query_id = item.get("id", f"Q-{i+1}")
        query = item.get("query", "")
        expected = item.get("expected_standards", [])  # may not be present in private set

        print(f"[{i+1}/{len(dataset)}] Processing: {query_id}")

        result = run_pipeline(
            query=query,
            retriever=retriever,
            top_k=5,
            use_llm_rerank=True,
        )

        out_item = {
            "id": query_id,
            "query": query,
            "retrieved_standards": result["retrieved_standards"],
            "latency_seconds": result["latency_seconds"],
        }

        # Include expected_standards if present (for eval_script.py compatibility)
        if expected:
            out_item["expected_standards"] = expected

        results.append(out_item)
        print(f"      -> Standards: {result['retrieved_standards'][:3]}... | Latency: {result['latency_seconds']:.2f}s")

    total_time = time.time() - total_start
    avg_latency = sum(r["latency_seconds"] for r in results) / len(results)

    print(f"\n[INFO] Done! Total time: {total_time:.1f}s | Avg latency: {avg_latency:.2f}s")

    # Save output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="BIS Standards Recommendation Engine — Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON results")
    args = parser.parse_args()

    process_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
