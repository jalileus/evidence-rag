import json
import math
import re
from collections import Counter
from pathlib import Path

QA_FILE = Path("data/tfkp_15_qa.md")
CORPUS_FILE = Path("data/corpus.jsonl")
QUESTIONS_FILE = Path("data/dev_questions.jsonl")
GOLD_FILE = Path("data/dev_gold.jsonl")


QUESTION_CATEGORY = {
    1: "условия теорем",
    2: "условия теорем",
    3: "сопоставление фрагментов и контрпримеры",
    4: "сопоставление фрагментов и контрпримеры",
    5: "условия теорем",
    6: "условия теорем",
    7: "короткие вычисления и выводы",
    8: "условия теорем",
    9: "сопоставление фрагментов и контрпримеры",
    10: "короткие вычисления и выводы",
    11: "короткие вычисления и выводы",
    12: "сопоставление фрагментов и контрпримеры",
    13: "недостаточность источника",
    14: "недостаточность источника",
    15: "недостаточность источника",
}

TYPE_LINE_MAP = {
    "условия": "условия теорем",
    "сопоставление": "сопоставление фрагментов и контрпримеры",
    "применение": "короткие вычисления и выводы",
    "недостаточность данных": "недостаточность источника",
}


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    return text.strip()


def tokens(text):
    return re.findall(r"[a-zа-яё0-9]+", text.lower())


def parse_qa(text):
    sections = re.split(r"(?m)^##\s+(\d+)\.\s+", text)
    result = []

    for i in range(1, len(sections), 2):
        number = int(sections[i])
        body = sections[i + 1]

        title = body.splitlines()[0].strip()

        # "Тип:" in the .md questions
        category = QUESTION_CATEGORY.get(number, "unknown")

        type_match = re.search(r"\*\*Тип:\*\*\s*([^\.\n]+)", body)
        if type_match:
            type_raw = type_match.group(1).strip().lower()
            for keyword, cat in TYPE_LINE_MAP.items():
                if keyword in type_raw:
                    if cat != category:
                        print(
                            f"  WARNING q_{number:03d}: hardcoded={category!r} "
                            f"but Тип line says {cat!r}. Using hardcoded."
                        )
                    break

        q_match = re.search(
            r"\*\*Вопрос\.\*\*\s*(.*?)(?=\n\s*\n-\s*\*\*A\.\*\*)", body, re.S
        )
        if not q_match:
            raise ValueError(f"Cannot parse question {number}")
        question = clean(q_match.group(1))

        options = []
        option_matches = re.findall(r"(?m)^-\s*\*\*([A-D])\.\*\*\s*(.*?)(?=\n|$)", body)
        for option_id, option_text in option_matches:
            options.append({"id": option_id, "text": clean(option_text)})
        if len(options) != 4:
            raise ValueError(
                f"Question {number}: expected 4 options, found {len(options)}"
            )

        answer_match = re.search(r"\*\*Ответ:\*\*\s*(.*)", body)
        if not answer_match:
            raise ValueError(f"Cannot parse answer for question {number}")
        answer_text = answer_match.group(1).strip()
        abstain = "отказ" in answer_text.lower()
        answer_ids = [] if abstain else re.findall(r"\b[A-D]\b", answer_text)

        source_match = re.search(r"\*\*Источник:\*\*\s*(.*)", body)
        if not source_match:
            raise ValueError(f"Cannot parse source for question {number}")
        source_line = source_match.group(1)
        pages_match = re.search(r"с\.\s*([\d,\s]+)", source_line)
        if not pages_match:
            raise ValueError(f"Cannot parse source pages for question {number}")
        pages = [int(x) for x in re.findall(r"\d+", pages_match.group(1))]

        explanation_start = answer_match.end()
        explanation_end = source_match.start()
        rationale = clean(body[explanation_start:explanation_end])
        rationale = re.sub(r"\n+", " ", rationale).strip()

        result.append(
            {
                "id": f"q_{number:03d}",
                "number": number,
                "title": title,
                "question": question,
                "options": options,
                "answer_ids": answer_ids,
                "abstain": abstain,
                "rationale": rationale[:1200],
                "pages": pages,
                "category": category,
            }
        )

    if len(result) != 15:
        raise ValueError(f"Expected 15 questions, parsed {len(result)}")
    return result


def choose_block(page_blocks, query_tokens, idf):
    query_counts = Counter(query_tokens)
    best_block, best_score = None, -1.0
    for block in page_blocks:
        block_tokens = tokens(block.get("text", ""))
        counts = Counter(block_tokens)
        # TF-IDF score to find the most relevant block for the query
        score = sum(
            idf.get(term, 0.0) * min(freq, counts[term])
            for term, freq in query_counts.items()
        )
        score += min(len(block_tokens), 500) * 1e-7
        if score > best_score:
            best_score = score
            best_block = block
    return best_block


def main():
    qa_text = QA_FILE.read_text(encoding="utf-8")
    questions = parse_qa(qa_text)
    corpus = read_jsonl(CORPUS_FILE)

    required_fields = {"block_id", "page", "text"}
    for i, block in enumerate(corpus, 1):
        missing = required_fields - block.keys()
        if missing:
            raise ValueError(f"Corpus line {i} missing fields: {sorted(missing)}")

    by_page = {}
    for block in corpus:
        by_page.setdefault(int(block["page"]), []).append(block)

    document_frequency = Counter()
    for block in corpus:
        document_frequency.update(set(tokens(block.get("text", ""))))

    total_docs = max(len(corpus), 1)
    idf = {
        term: math.log(1 + total_docs / (1 + df))
        for term, df in document_frequency.items()
    }

    dev_questions = []
    dev_gold = []

    for item in questions:
        dev_questions.append(
            {
                "id": item["id"],
                "question": item["question"],
                "options": item["options"],
            }
        )

        if item["abstain"]:
            citation_sets = []
        else:
            correct_option_text = " ".join(
                option["text"]
                for option in item["options"]
                if option["id"] in item["answer_ids"]
            )
            query = " ".join(
                [
                    item["title"],
                    item["question"],
                    correct_option_text,
                    item["rationale"],
                ]
            )
            query_tokens = tokens(query)

            selected_ids = []
            for page in item["pages"]:
                page_blocks = by_page.get(page, [])
                if not page_blocks:
                    raise ValueError(
                        f"{item['id']}: no corpus blocks found for PDF page {page}"
                    )
                block = choose_block(page_blocks, query_tokens, idf)
                selected_ids.append(block["block_id"])

            citation_sets = [list(dict.fromkeys(selected_ids))]

        dev_gold.append(
            {
                "id": item["id"],
                "answer_ids": item["answer_ids"],
                "abstain": item["abstain"],
                "rationale": item["rationale"]
                or (
                    "The provided source does not contain enough information."
                    if item["abstain"]
                    else "See the cited textbook passages."
                ),
                "citation_sets": citation_sets,
                "category": item["category"],
            }
        )

    write_jsonl(QUESTIONS_FILE, dev_questions)
    write_jsonl(GOLD_FILE, dev_gold)

    print(f"Created {QUESTIONS_FILE} ({len(dev_questions)} questions)")
    print(f"Created {GOLD_FILE} ({len(dev_gold)} gold records)")

    from collections import Counter as C

    cat_counts = C(item["category"] for item in dev_gold)
    print("\nCategory distribution:")
    for cat, n in sorted(cat_counts.items()):
        print(f"  {cat:<30} {n}")

    print("\nCitation IDs selected per question:")
    for item in dev_gold:
        print(f"  {item['id']}  [{item['category']}]  {item['citation_sets']}")


if __name__ == "__main__":
    main()
