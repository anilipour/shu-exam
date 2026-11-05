import io
import os
from pathlib import Path

import pytest

from shu_exam.cli import main, read_answer
from shu_exam.examiner import ExaminerError, ExaminerTurn, Review

PDF = Path(os.environ.get("SHU_EXAM_PDF") or "Shu_Exam.pdf")
needs_pdf = pytest.mark.skipif(not PDF.is_file(), reason="exam PDF is not available")

REVIEW = Review(
    score=8,
    summary="Solid.",
    correct=["Tides"],
    missing=["Rigid limit"],
    errors=[],
)


def _reader(lines):
    items = iter(lines)

    def read():
        try:
            return next(items)
        except StopIteration:
            raise EOFError from None

    return read


def test_read_answer_stops_at_lone_period():
    assert read_answer(_reader(["one", "", "two.", " . ", "ignored"])) == "one\n\ntwo."


def test_read_answer_stops_at_end_of_input():
    assert read_answer(_reader(["only line"])) == "only line"


def test_read_answer_empty():
    assert read_answer(_reader(["."])) == ""


def test_read_answer_done_ends_without_a_period():
    assert read_answer(_reader(["Done", "ignored"])) == "done"


def test_read_answer_keeps_done_inside_an_answer():
    assert read_answer(_reader(["first", "done", "."])) == "first\ndone"


class FakeExaminer:
    """Replaces the backend: plays scripted turns and records each transcript."""

    def __init__(self, monkeypatch, turns):
        self.turns = list(turns)
        self.transcripts = []
        self.reviewed = None
        self.backend = None
        monkeypatch.setattr("shu_exam.cli.next_turn", self.next_turn)
        monkeypatch.setattr("shu_exam.cli.final_review", self.final_review)
        monkeypatch.setattr("shu_exam.cli.backend_problem", lambda backend: None)
        monkeypatch.delenv("SHU_EXAM_BACKEND", raising=False)

    def next_turn(self, exam, question, transcript, *, backend, model):
        self.transcripts.append([(m.role, m.text) for m in transcript])
        self.backend = backend
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn

    def final_review(self, exam, question, transcript, *, backend, model):
        self.reviewed = [(m.role, m.text) for m in transcript]
        return REVIEW


def _run(monkeypatch, args, typed):
    monkeypatch.setattr("sys.stdin", io.StringIO(typed))
    return main(["--pdf", str(PDF), *args])


def _turn(message, finished=False):
    return ExaminerTurn(message=message, finished=finished)


@needs_pdf
def test_session_follows_up_until_student_says_done(monkeypatch, capsys):
    examiner = FakeExaminer(monkeypatch, [_turn("Are you sure?"), _turn("Why?")])

    code = _run(monkeypatch, ["-q", "2"], "my answer\n.\nmy reply\nmore\n.\ndone\n")

    assert code == 0
    assert examiner.backend == "claude-code"
    assert examiner.transcripts == [
        [("student", "my answer")],
        [
            ("student", "my answer"),
            ("examiner", "Are you sure?"),
            ("student", "my reply\nmore"),
        ],
    ]
    assert examiner.reviewed[-1] == ("examiner", "Why?")
    output = capsys.readouterr().out
    assert output.index("Are you sure?") < output.index("Why?") < output.index("Score: 8/10")
    assert "Reference answer: PDF pages 5-6." in output


@needs_pdf
def test_session_ends_when_examiner_finishes(monkeypatch, capsys):
    examiner = FakeExaminer(monkeypatch, [_turn("Well argued.", finished=True)])

    assert _run(monkeypatch, ["-q", "2"], "my answer\n.\n") == 0

    assert len(examiner.transcripts) == 1
    assert examiner.reviewed[-1] == ("examiner", "Well argued.")
    assert "Score: 8/10" in capsys.readouterr().out


@needs_pdf
def test_no_score_before_the_session_ends(monkeypatch, capsys):
    FakeExaminer(monkeypatch, [_turn("Are you sure?")])
    assert _run(monkeypatch, ["-q", "2"], "my answer\n.\ndone\n") == 0
    output = capsys.readouterr().out
    assert output.index("Are you sure?") < output.index("Score:")


@needs_pdf
def test_empty_first_answer_sends_nothing(monkeypatch, capsys):
    examiner = FakeExaminer(monkeypatch, [])
    assert _run(monkeypatch, ["-q", "2"], ".\n") == 0
    assert examiner.transcripts == []
    assert "nothing was sent" in capsys.readouterr().out


@needs_pdf
def test_failed_request_can_be_retried_without_losing_the_session(monkeypatch, capsys):
    examiner = FakeExaminer(
        monkeypatch, [ExaminerError("Rate limited"), _turn("Back again.", finished=True)]
    )

    assert _run(monkeypatch, ["-q", "2"], "my answer\n.\ny\n") == 0

    assert len(examiner.transcripts) == 2
    assert examiner.transcripts[0] == examiner.transcripts[1]
    assert "Rate limited" in capsys.readouterr().err


@needs_pdf
def test_declining_retry_still_gives_a_review(monkeypatch, capsys):
    examiner = FakeExaminer(monkeypatch, [ExaminerError("Rate limited")])
    assert _run(monkeypatch, ["-q", "2"], "my answer\n.\nn\n") == 0
    assert examiner.reviewed == [("student", "my answer")]


@needs_pdf
def test_gemini_warning_is_shown(monkeypatch, capsys):
    FakeExaminer(monkeypatch, [])
    assert _run(monkeypatch, ["-q", "2", "--backend", "gemini"], ".\n") == 0
    assert "human reviewers" in capsys.readouterr().out


@needs_pdf
def test_stops_before_typing_if_backend_unavailable(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert _run(monkeypatch, ["-q", "2", "--backend", "gemini"], "my answer\n.\n") == 2
    captured = capsys.readouterr()
    assert "GEMINI_API_KEY" in captured.err
    assert "Type your answer" not in captured.out
