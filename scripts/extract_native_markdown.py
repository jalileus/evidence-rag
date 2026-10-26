import re
import sys
from pathlib import Path
import pymupdf  

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "data" / "textbook.pdf"
OUTPUT_PATH = ROOT / "data" / "native_text.md"


LATIN_EXTENDED = re.compile(r"[\u00c0-\u00ff]")
CYRILLIC = re.compile(r"[\u0400-\u052f]")

TOC_LINE = re.compile(
    r"(?:\.\s*){3,}\d{1,3}\s*$|" r"…{2,}\s*\d{1,3}\s*$|" r"_{3,}\s*\d{1,3}\s*$"
)

THEOREM_LABEL = re.compile(
    r"^(Теорема|Определение|Лемма|Следствие|Утверждение|"
    r"Замечание|Пример|Задача)\s+\d+(?:\.\d+)*\.?",
    re.IGNORECASE,
)

NUMBERED_HEADING = re.compile(
    r"^\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z]",
    re.IGNORECASE,
)

CHAPTER_HEADING = re.compile(
    r"^Глава\s+\S+",
    re.IGNORECASE,
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


def repair_word(word: str) -> str:
    """
    Attempt to repair a word that appears to be Russian text decoded
    with the wrong encoding.
    """
    if not LATIN_EXTENDED.search(word) or CYRILLIC.search(word):
        return word

    try:
        candidate = word.encode("latin-1").decode("cp1251")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return word

    letters = [char for char in candidate if char.isalpha()]
    if not letters:
        return word

    russian_count = sum(bool(CYRILLIC.fullmatch(char)) for char in letters)

    if russian_count / len(letters) >= 0.65:
        return candidate

    return word


def repair_russian(text: str) -> str:
    """
    Repair suspiciously misdecoded Russian words.
    """
    LOG_ENCODING_REPAIRS = False

    def replace_match(match: re.Match) -> str:
        original = match.group()
        repaired = repair_word(original)

        if LOG_ENCODING_REPAIRS and repaired != original:
            print(
                f"Encoding repair: {original!r} -> {repaired!r}",
                file=sys.stderr,
            )

        return repaired

    return re.sub(r"\S+", replace_match, text)


def heading_candidate(line: str) -> str:
    """Remove leading symbols sometimes found before extracted headings."""
    return re.sub(r"^[^\w]+", "", line, flags=re.UNICODE).strip()


def is_chapter_one(line: str) -> bool:
    candidate = heading_candidate(line)

    return bool(
        re.match(
            r"^1\.\s*Комплексные\s+числа\b",
            candidate,
            re.IGNORECASE,
        )
    )


def is_heading(line: str) -> bool:
    candidate = heading_candidate(line)

    if len(candidate) > 180:
        return False

    return bool(
        THEOREM_LABEL.match(candidate)
        or NUMBERED_HEADING.match(candidate)
        or CHAPTER_HEADING.match(candidate)
    )


def make_paragraph(lines: list[str]) -> str:
    text = "\n".join(lines)

    text = re.sub(
        r"(?<=[А-Яа-яЁё])-\s*\n\s*(?=[А-Яа-яЁё])",
        "",
        text,
    )

    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def clean_lines(raw_text: str) -> list[str]:
    raw_text = repair_russian(raw_text)
    lines = [line.strip() for line in raw_text.splitlines()]

    cleaned = []
    for line in lines:
        if HEADER_LINE.match(line):
            continue

        if TOC_LINE.search(line):
            continue

        cleaned.append(line)

    return cleaned


def split_inline_headings(lines: list[str]) -> list[str]:
    result = []

    heading_pattern = re.compile(
        r"(?<!\S)"
        r"(?=(?:Теорема|Определение|Лемма|Следствие|Утверждение|"
        r"Замечание|Пример|Задача)\s+\d+(?:\.\d+)*\.?)"
    )

    for line in lines:
        if is_heading(line):
            result.append(line)
            continue

        parts = heading_pattern.split(line)

        if len(parts) == 1:
            result.append(line)
            continue

        result.extend(part.strip() for part in parts if part.strip())

    return result


def extract_page_text(raw_text: str) -> str:
    lines = clean_lines(raw_text)

    chapter_start = next(
        (index for index, line in enumerate(lines) if is_chapter_one(line)),
        None,
    )

    if chapter_start is not None:
        lines = lines[chapter_start:]
    else:
        toc_count = sum(bool(TOC_LINE.search(line)) for line in raw_text.splitlines())

        starts_with_contents = any(
            line.casefold() in {"содержание", "оглавление"} for line in lines[:5]
        )

        if toc_count >= 3 or starts_with_contents:
            return "<!-- SKIP_TOC_PAGE -->"

    lines = split_inline_headings(lines)

    paragraphs = []
    current_lines = []

    def flush() -> None:
        if current_lines:
            paragraph = make_paragraph(current_lines)
            if paragraph:
                paragraphs.append(paragraph)
            current_lines.clear()

    for line in lines:
        if not line:
            flush()
        elif is_heading(line):
            flush()
            paragraphs.append(line)
        else:
            current_lines.append(line)

    flush()

    return "\n\n".join(paragraphs)


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    page_outputs = []
    skipped_pages = []

    with pymupdf.open(PDF_PATH) as pdf:
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1

            raw_text = page.get_text("text", sort=True)
            text = extract_page_text(raw_text)

            if "<!-- SKIP_TOC_PAGE -->" in text:
                skipped_pages.append(page_number)
                page_outputs.append(
                    f"<!-- PAGE: {page_number} -->\n" "<!-- SKIP_TOC_PAGE -->"
                )
                continue

            page_outputs.append(f"<!-- PAGE: {page_number} -->\n{text}")

    OUTPUT_PATH.write_text(
        "\n\n".join(page_outputs) + "\n",
        encoding="utf-8",
    )

    print(f"Saved Markdown : {OUTPUT_PATH}")
    print(f"PDF pages      : {len(page_outputs)}")
    print(f"Skipped TOC    : {skipped_pages}")


if __name__ == "__main__":
    main()
