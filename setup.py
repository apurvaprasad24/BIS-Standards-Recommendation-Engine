"""
setup.py — BIS RAG Engine Setup
Run this once to parse the PDF and build the retrieval index.

Usage:
    python setup.py
    python setup.py --pdf path/to/dataset.pdf
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parse_pdf import run_parsing
from src.retriever import build_retriever


def main():
    parser = argparse.ArgumentParser(description="Setup: Parse PDF and build retrieval index")
    parser.add_argument("--pdf", type=str, default="data/dataset.pdf",
                        help="Path to BIS SP 21 dataset PDF (default: data/dataset.pdf)")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"[ERROR] PDF not found: {args.pdf}")
        print("Please ensure data/dataset.pdf exists.")
        sys.exit(1)

    os.makedirs("data", exist_ok=True)

    print("=" * 60)
    print("  BIS RAG Engine — Setup")
    print("=" * 60)

    # Step 1: Parse PDF
    print("\n[STEP 1] Parsing PDF...")
    chunks = run_parsing(pdf_path=args.pdf, output_path="data/chunks.pkl")

    # Step 2: Build retrieval index
    print("\n[STEP 2] Building retrieval index...")
    build_retriever(chunks_path="data/chunks.pkl", retriever_path="data/retriever.pkl")

    print("\n" + "=" * 60)
    print("  Setup complete! You can now run:")
    print("  python inference.py --input data/public_test_set.json --output data/results.json")
    print("  python eval_script.py --results data/results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
