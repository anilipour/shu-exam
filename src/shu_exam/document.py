"""Find the questions and reference answers in the exam PDF.

Questions are bold, 12 pt, numbered headings in the left margin. A question's
reference answer is everything between its heading and the next one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

_FOOTER_TOP = 700  # Page numbers sit below this y coordinate.
_RUNNING_HEADER = re.compile(r"^Chapter \d+\. ")


@dataclass(frozen=True)
class Question:
    number: int
    chapter: str
    text: str
    first_page: int  # 1-indexed PDF page holding the heading.
    last_page: int  # 1-indexed PDF page where the reference answer ends.
    has_reference: bool


@dataclass
class _Line:
    spans: list[dict]
    y: float


def clean_text(text: str) -> str:
    """Undo line-break and accent artifacts from PDF text extraction."""
    text = re.sub(r"(\w)- (\w)", r"\1\2", text)
    text = re.sub(r"([?.])([A-Z])", r"\1 \2", text)
    text = text.replace("Str¨omgren", "Strömgren").replace("B −V", "B − V")
    return re.sub(r"\s+", " ", text).strip()


def _page_lines(page: pymupdf.Page) -> list[_Line]:
    """Group spans into baselines, keeping sub/superscripts on their line."""
    spans = [
        span
        for block in page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
        if span["text"].strip() or span["text"] == " "
    ]
    spans.sort(key=lambda s: (round(s["origin"][1]), s["bbox"][0]))
    lines: list[_Line] = []
    for span in spans:
        if lines and abs(span["origin"][1] - lines[-1].y) < 6:
            lines[-1].spans.append(span)
        else:
            lines.append(_Line([span], span["origin"][1]))
    for line in lines:
        line.spans.sort(key=lambda s: s["bbox"][0])
    return lines


def _page_content(page: pymupdf.Page) -> list[_Line | None]:
    """Text lines and figures top to bottom; ``None`` stands for a figure."""
    lines = [(line.y, line) for line in _page_lines(page)]
    images = [(image["bbox"][1], None) for image in page.get_image_info()]
    return [item for _, item in sorted(lines + images, key=lambda pair: pair[0])]


def _text(spans: list[dict]) -> str:
    return "".join(s["text"] for s in spans)


def _is_bold(span: dict) -> bool:
    return any(tag in span["font"] for tag in ("BX", "MIB", "BSY"))


def _is_heading_start(line: _Line, expected: int) -> bool:
    first = line.spans[0]
    return (
        _is_bold(first)
        and 11 < first["size"] < 13
        and first["bbox"][0] < 75
        and first["text"].strip() == str(expected)
    )


def _is_heading_continuation(line: _Line, indent: float) -> bool:
    """True for a wrapped heading line, which lines up with the heading's text."""
    first = line.spans[0]
    bold = sum(len(s["text"]) for s in line.spans if _is_bold(s))
    total = sum(len(s["text"]) for s in line.spans)
    return (
        abs(first["bbox"][0] - indent) < 3
        and 11 < max(s["size"] for s in line.spans) < 13
        and bold / total > 0.5
    )


def _is_page_furniture(line: _Line) -> bool:
    """True for running headers, page numbers, and chapter titles."""
    text = _text(line.spans).strip()
    return (
        bool(_RUNNING_HEADER.match(text))
        or (line.y > _FOOTER_TOP and text.isdigit())
        or max(s["size"] for s in line.spans) > 13
    )


class ExamDocument:
    """An opened exam PDF and the questions found in it."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._doc = pymupdf.open(self.path)
        self.questions = self._parse()
        if not self.questions:
            raise ValueError(f"No numbered questions found in {self.path}")

    def _chapter_for(self, page_number: int) -> str:
        chapter = ""
        for _, title, start in self._doc.get_toc():
            if start <= page_number:
                chapter = title
        return chapter

    def _parse(self) -> list[Question]:
        found: list[dict] = []
        current: dict | None = None  # The question whose answer is being read.
        in_heading = False

        for page_number, page in enumerate(self._doc, start=1):
            for line in _page_content(page):
                if line is not None and _is_heading_start(line, len(found) + 1):
                    current = {
                        "number": len(found) + 1,
                        "chapter": self._chapter_for(page_number),
                        "heading": [_text(line.spans[1:])],
                        # Wrapped lines align with the text after the number.
                        "indent": line.spans[1]["bbox"][0],
                        "first_page": page_number,
                        "last_page": page_number,
                        "has_reference": False,
                    }
                    found.append(current)
                    in_heading = True
                elif current is None or (line is not None and _is_page_furniture(line)):
                    continue
                elif (
                    in_heading
                    and line is not None
                    and _is_heading_continuation(line, current["indent"])
                ):
                    current["heading"].append(_text(line.spans))
                else:
                    in_heading = False
                    current["has_reference"] = True
                    current["last_page"] = page_number

        return [
            Question(
                number=q["number"],
                chapter=q["chapter"],
                text=clean_text(" ".join(q["heading"])),
                first_page=q["first_page"],
                last_page=q["last_page"],
                has_reference=q["has_reference"],
            )
            for q in found
        ]

    def get(self, number: int) -> Question:
        return self.questions[number - 1]

    def filter_by_chapter(self, query: str) -> list[Question]:
        """Questions whose chapter title contains ``query``, ignoring case."""
        query = query.casefold()
        return [q for q in self.questions if query in q.chapter.casefold()]

    def reference_pdf(self, question: Question) -> bytes:
        """Return a small PDF with only the pages holding this answer."""
        subset = pymupdf.open()
        subset.insert_pdf(
            self._doc,
            from_page=question.first_page - 1,
            to_page=question.last_page - 1,
        )
        return subset.tobytes(garbage=4, deflate=True)

    def reference_images(self, question: Question, dpi: int = 150) -> list[bytes]:
        """Return a PNG of each page holding this answer."""
        return [
            self._doc[index].get_pixmap(dpi=dpi).tobytes("png")
            for index in range(question.first_page - 1, question.last_page)
        ]
