import os
from pathlib import Path

import pymupdf
import pytest

from shu_exam.document import ExamDocument, clean_text

PDF = Path(os.environ.get("SHU_EXAM_PDF") or "Shu_Exam.pdf")


def test_clean_text():
    assert clean_text("two applica- tions") == "two applications"
    assert clean_text("the Belt?What is") == "the Belt? What is"
    assert clean_text("Str¨omgren  spheres") == "Strömgren spheres"


@pytest.fixture(scope="module")
def exam():
    if not PDF.is_file():
        pytest.skip("exam PDF is not available")
    return ExamDocument(PDF)


def test_finds_all_questions_in_order(exam):
    assert [q.number for q in exam.questions] == list(range(1, 91))
    assert all(q.text and q.chapter for q in exam.questions)


def test_questions_without_written_answers(exam):
    assert {q.number for q in exam.questions if not q.has_reference} == set(range(79, 91))


def test_chapter_boundary_excludes_next_chapter_page(exam):
    assert (exam.get(10).first_page, exam.get(10).last_page) == (11, 12)
    assert exam.get(11).first_page == 13


def test_answer_spills_onto_page_of_next_heading(exam):
    assert (exam.get(1).first_page, exam.get(1).last_page) == (4, 5)


def test_figure_after_last_answer_belongs_to_it(exam):
    assert exam.get(78).last_page == 71


def test_chapter_filter(exam):
    assert len(exam.filter_by_chapter("cosmology")) == 14


def test_reference_pdf_has_only_answer_pages(exam):
    subset = pymupdf.open(stream=exam.reference_pdf(exam.get(1)), filetype="pdf")
    assert subset.page_count == 2
