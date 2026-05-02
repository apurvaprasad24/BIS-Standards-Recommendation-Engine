"""
BIS SP 21 PDF Parser (v2)
Extracts individual IS standards with full content across page boundaries.
"""

import fitz
import re
import json
import pickle
from pathlib import Path


SUMMARY_START = re.compile(r'SUMMARY\s+OF\s*\n(IS\s+.+)', re.IGNORECASE)


def clean_text(text: str) -> str:
    text = re.sub(r'\d+\.\d+\s*\nSP\s+21\s*:\s*2005\n?', ' ', text)
    text = re.sub(r'SP\s+21\s*:\s*2005\n?', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def normalize_std_id(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s*:\s*', ': ', s)
    s = re.sub(r'\(\s*PART\s*', '(Part ', s, flags=re.IGNORECASE)
    return s


def parse_is_number(std_id: str) -> str:
    m = re.search(r'IS\s+(\d+)', std_id, re.IGNORECASE)
    return m.group(1) if m else ""


def extract_chunks_from_pdf(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    print(f"  -> {n_pages} pages")

    page_texts = [doc[i].get_text() for i in range(n_pages)]
    page_offsets = {}
    full_text = ""
    for i, text in enumerate(page_texts):
        page_offsets[len(full_text)] = i + 1
        full_text += text + "\n"

    chunks = []
    seen_ids = set()
    matches = list(SUMMARY_START.finditer(full_text))

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else start + 4000
        block_text = clean_text(full_text[start:end])
        header_line = match.group(1).strip()

        id_match = re.match(
            r'(IS\s+[\d]+(?:\s*\(PART\s*\d+\))?\s*:\s*\d{4})\s*(.*)',
            header_line, re.IGNORECASE
        )
        if not id_match:
            id_match = re.match(r'(IS\s+[\d]+(?:\s*\(PART\s*\d+\))?)\s*(.*)', header_line, re.IGNORECASE)

        if not id_match:
            continue

        std_id_raw = id_match.group(1).strip()
        title_suffix = id_match.group(2).strip() if id_match.lastindex >= 2 else ""
        std_id = normalize_std_id(std_id_raw)
        is_num = parse_is_number(std_id)

        if std_id in seen_ids:
            continue
        seen_ids.add(std_id)

        pg = 1
        for off in sorted(page_offsets.keys()):
            if off <= start:
                pg = page_offsets[off]

        full_title = f"{std_id} {title_suffix}".strip()
        chunk_content = f"{std_id}\n{full_title}\n{full_title}\n{block_text[:2500]}"

        chunks.append({
            "std_id": std_id,
            "std_id_raw": std_id_raw,
            "is_num": is_num,
            "title": full_title[:200],
            "content": chunk_content,
            "page": pg,
        })

    # Mine ToC for any missed standards
    toc_pattern = re.compile(
        r'^\s*IS\s+([\d]+(?:\s*\(Part\s*\d+\))?)\s*:\s*(\d{4})\s+(.+?)(?:\s+[\d\.]+)?\s*$',
        re.MULTILINE | re.IGNORECASE
    )
    toc_text = "".join(page_texts[:30])
    for m in toc_pattern.finditer(toc_text):
        is_num = m.group(1).strip()
        year = m.group(2).strip()
        title = m.group(3).strip()
        raw = f"IS {is_num} : {year}"
        std_id = normalize_std_id(raw)
        if std_id not in seen_ids:
            seen_ids.add(std_id)
            chunks.append({
                "std_id": std_id,
                "std_id_raw": raw,
                "is_num": is_num.replace(" ", ""),
                "title": f"{std_id} {title}",
                "content": f"{std_id} {std_id} {title} {title}",
                "page": 0,
            })

    return chunks


def run_parsing(pdf_path: str, output_path: str):
    print(f"[1/3] Parsing PDF: {pdf_path}")
    chunks = extract_chunks_from_pdf(pdf_path)
    print(f"      -> {len(chunks)} standards extracted")

    print(f"[2/3] Saving chunks...")
    with open(output_path, "wb") as f:
        pickle.dump(chunks, f)

    json_path = output_path.replace(".pkl", "_sample.json")
    with open(json_path, "w") as f:
        json.dump(chunks[:30], f, indent=2)

    print("[3/3] Done!")
    return chunks


if __name__ == "__main__":
    run_parsing(pdf_path="data/dataset.pdf", output_path="data/chunks.pkl")
