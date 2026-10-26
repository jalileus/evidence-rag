import argparse
import hashlib
import json
import pickle
import re
import sys
import time

from datetime import datetime, timezone
from pathlib import Path

from rank_bm25 import BM25Okapi

REQUIRED_FIELDS = {"block_id", "page", "section", "text"}

MATH_SYMBOLS = "+\-−*/=<>≤≥≠≈∂∆△∇∑∏√∞∈∉⊂⊃⊆⊇∩∪∅" "∧∨∀∃→↔ℂℝℤℕ∫∮×·⋅±"

WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
MATH_PATTERN = re.compile(
    r"[^\W_]+|[" + re.escape(MATH_SYMBOLS) + r"]",
    re.UNICODE,
)


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    retrieval = config.get("retrieval", {})
    tokenization = config.get("tokenization", {})

    if retrieval.get("method", "bm25").lower() != "bm25":
        raise ValueError("This version supports only BM25.")

    k1 = float(retrieval.get("k1", 1.5))
    b = float(retrieval.get("b", 0.75))

    if k1 <= 0:
        raise ValueError("BM25 k1 must be greater than zero.")

    if not 0 <= b <= 1:
        raise ValueError("BM25 b must be between 0 and 1.")

    min_token_length = int(tokenization.get("min_token_length", 1))

    if min_token_length < 1:
        raise ValueError("min_token_length must be at least 1.")

    return {
        "k1": k1,
        "b": b,
        "lowercase": bool(tokenization.get("lowercase", True)),
        "include_math_symbols": bool(tokenization.get("include_math_symbols", True)),
        "min_token_length": min_token_length,
    }


def load_corpus(path):
    records = []
    seen_ids = set()

    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object.")

            if set(record.keys()) != REQUIRED_FIELDS:
                raise ValueError(
                    f"Line {line_number} must contain exactly these fields: "
                    f"{sorted(REQUIRED_FIELDS)}. "
                    f"Found: {sorted(record.keys())}"
                )

            block_id = record["block_id"]

            if not isinstance(block_id, str) or not block_id.strip():
                raise ValueError(
                    f"Line {line_number}: block_id must be a non-empty string."
                )

            if block_id in seen_ids:
                raise ValueError(
                    f"Duplicate block_id on line {line_number}: {block_id}"
                )

            page = record["page"]

            if type(page) is not int or page < 1:
                raise ValueError(
                    f"Line {line_number}: page must be a positive integer."
                )

            if not isinstance(record["section"], str):
                raise ValueError(f"Line {line_number}: section must be a string.")

            if not isinstance(record["text"], str) or not record["text"].strip():
                raise ValueError(f"Line {line_number}: text must not be empty.")

            seen_ids.add(block_id)
            records.append(record)

    if not records:
        raise ValueError("The corpus is empty.")

    return records


def tokenize(text, settings):
    if settings["include_math_symbols"]:
        tokens = MATH_PATTERN.findall(text)
    else:
        tokens = WORD_PATTERN.findall(text)

    if settings["lowercase"]:
        tokens = [token.casefold() for token in tokens]

    min_length = settings["min_token_length"]

    return [token for token in tokens if len(token) >= min_length]


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Build a BM25 index for the textbook corpus."
    )
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="Path to corpus.jsonl",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where the index will be saved",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to config.json",
    )
    args = parser.parse_args()

    if not args.corpus.is_file():
        parser.error(f"Corpus file not found: {args.corpus}")

    if not args.config.is_file():
        parser.error(f"Config file not found: {args.config}")

    start_time = time.perf_counter()

    settings = load_config(args.config)
    records = load_corpus(args.corpus)

    tokenized_documents = [tokenize(record["text"], settings) for record in records]

    empty_documents = sum(1 for tokens in tokenized_documents if not tokens)

    if empty_documents == len(records):
        raise ValueError("No searchable tokens were found in the corpus.")

    if empty_documents:
        print(
            f"Warning: {empty_documents} documents contain no searchable " "tokens.",
            file=sys.stderr,
        )

    bm25 = BM25Okapi(
        tokenized_documents,
        k1=settings["k1"],
        b=settings["b"],
    )

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    documents_path = output_dir / "documents.jsonl"
    index_path = output_dir / "bm25.pkl"
    manifest_path = output_dir / "manifest.json"

    with documents_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with index_path.open("wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed_seconds = time.perf_counter() - start_time

    manifest = {
        "format_version": 1,
        "index_type": "BM25Okapi",
        "document_count": len(records),
        "empty_token_documents": empty_documents,
        "corpus_sha256": sha256_file(args.corpus),
        "bm25": {
            "k1": settings["k1"],
            "b": settings["b"],
        },
        "tokenization": {
            "lowercase": settings["lowercase"],
            "include_math_symbols": settings["include_math_symbols"],
            "min_token_length": settings["min_token_length"],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_seconds": round(elapsed_seconds, 3),
        "python_version": sys.version.split()[0],
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_size = sum(
        path.stat().st_size for path in (documents_path, index_path, manifest_path)
    )

    print("\nBM25 index built successfully.")
    print(f"Documents:   {len(records)}")
    print(f"Unique IDs:  {len({r['block_id'] for r in records})}")
    print(f"Build time:  {elapsed_seconds:.2f} seconds")
    print(f"Index size:  {total_size / (1024 * 1024):.2f} MB")
    print(f"Saved to:    {output_dir.resolve()}")
    print(f"Manifest:    {manifest_path}")


if __name__ == "__main__":
    main()
