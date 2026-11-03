import json
import re
import sys
from pathlib import Path


INPUT_PATH = Path("data/native_text.md")
OUTPUT_PATH = Path("data/corpus.jsonl")

MIN_CHARS = 80
DISCARD_CHARS = 30
MAX_CHARS = 1200


THEOREM_LABEL = re.compile(
    r"^(Теорема|Определение|Лемма|Следствие|Утверждение|"
    r"Замечание|Пример|Задача)\s+(\d+(?:\.\d+)?)\b",
    re.MULTILINE,
)

SECTION_HEADING = re.compile(
    r"^(\d{1,2}(?:\.\d+)?)\.\s+"
    r"([А-ЯA-Z][^\n]{3,80}?)(?:\.|$)",
    re.MULTILINE,
)


HEADING_START = re.compile(
    r"(?m)^(?=(?:"
    r"Теорема|Определение|Лемма|Следствие|Утверждение|"
    r"Замечание|Пример|Задача)\s+\d+(?:\.\d+)?\b|"
    r"\d{1,2}(?:\.\d+)?\.\s+[А-ЯA-Z])"
)

PAGE_MARKER = re.compile(r"<!--\s*PAGE\s*:\s*(\d+)\s*-->")

HEADER_LINE = re.compile(
    r"^\s*(?:"
    r"ЛЕКЦИИ\s+ПО\s+ТФКП"
    r"|В\.\s*В\.\s*ГОРЯЙНОВ\s*,?\s*"
    r"Е\.\s*С\.\s*ПОЛОВИНКИН"
    r")\s*$",
    re.IGNORECASE,
)

TOC_LINE = re.compile(r"\.{4,}", re.MULTILINE)


def split_pages(text: str) -> list[tuple[int, str]]:
    """Return [(page_number, page_text), ...] in document order."""
    parts = PAGE_MARKER.split(text)
    result = []

    i = 1
    while i + 1 < len(parts):
        page_number = int(parts[i])
        page_text = parts[i + 1]
        result.append((page_number, page_text))
        i += 2

    return result


def clean_page(text: str) -> str:
    """Remove running headers/footers and TOC dotted lines."""
    text = HEADER_LINE.sub("", text)
    text = TOC_LINE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_heading(text: str) -> str | None:
    """Return the normalized heading if text starts with one."""
    stripped = text.strip()

    match = THEOREM_LABEL.match(stripped)
    if match:
        return f"{match.group(1)} {match.group(2)}"

    match = SECTION_HEADING.match(stripped)
    if match:
        number = match.group(1)
        title = match.group(2).strip()
        return f"{number}. {title}"

    return None


def split_paragraphs(text: str) -> list[str]:
    """Split text on blank lines."""
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n{2,}", text)
        if paragraph.strip()
    ]


def split_at_headings(paragraphs: list[str]) -> list[str]:
    """
    Split paragraphs at headings that start on a new line.
    """
    result = []

    for paragraph in paragraphs:
        parts = re.split(HEADING_START, paragraph)
        result.extend(
            part.strip()
            for part in parts
            if part.strip()
        )

    return result


def merge_short(paras: list[str]) -> list[str]:
    """
    Merge short paragraphs
    """
    merged = []
    buf = ""

    for paragraph in paras:
        is_heading = extract_heading(paragraph) is not None

        if is_heading and buf:
            merged.append(buf)
            buf = ""

        if buf:
            buf += "\n\n" + paragraph
        elif len(paragraph) < MIN_CHARS:
            buf = paragraph
        else:
            merged.append(paragraph)
            continue

        if len(buf) >= MIN_CHARS:
            merged.append(buf)
            buf = ""

    if buf:
        merged.append(buf)

    return merged


def split_long(text: str) -> list[str]:
    """
    Split long blocks near sentence boundaries.
    """
    if len(text) <= MAX_CHARS:
        return [text]

    sentences = re.split(r"(?<=[.!?;])\s+", text)

    chunks = []
    current = ""

    for sentence in sentences:
        candidate = (
            (current + " " + sentence).strip()
            if current
            else sentence
        )

        if len(candidate) > MAX_CHARS and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def is_noise(text: str) -> bool:
    """Return True for short or mostly punctuation-only blocks."""
    if len(text) < DISCARD_CHARS:
        return True

    printable = re.sub(r"[\s.\-–—]", "", text)
    return len(printable) < DISCARD_CHARS // 2


def build_corpus(input_path: Path, output_path: Path) -> None:
    raw = input_path.read_text(encoding="utf-8")
    pages = split_pages(raw)

    if not pages:
        print(
            "ERROR: No <!-- PAGE: N --> markers found in the file.",
            file=sys.stderr,
        )
        sys.exit(1)

    block_counter = 0
    current_section = ""

    # Skip page 1 (table of contents).
    skip_pages = {1}

    with output_path.open("w", encoding="utf-8") as out:
        for page_number, page_text in pages:
            if page_number in skip_pages:
                continue

            page_text = clean_page(page_text)
            paragraphs = split_paragraphs(page_text)

            paragraphs = split_at_headings(paragraphs)
            paragraphs = merge_short(paragraphs)

            for paragraph in paragraphs:
                heading = extract_heading(paragraph)
                if heading:
                    current_section = heading

                sub_blocks = split_long(paragraph)

                for block_text in sub_blocks:
                    block_text = block_text.strip()

                    if is_noise(block_text):
                        continue

                    block_counter += 1

                    record = {
                        "block_id": f"b{block_counter:07d}",
                        "page": page_number,
                        "section": current_section,
                        "text": block_text,
                    }

                    out.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

    print(f"Native corpus saved : {output_path}")
    print(f"Source pages        : {len(pages)} (skipped {len(skip_pages)})")
    print(f"Blocks written      : {block_counter}")



def main():
    build_corpus(INPUT_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
