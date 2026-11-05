import io
import os
from pathlib import Path

import pytest

from shu_exam.cli import main
from shu_exam.examiner import ExaminerTurn, Review
from shu_exam.voice import VoiceError, read_spoken_answer

PDF = Path(os.environ.get("SHU_EXAM_PDF") or "Shu_Exam.pdf")


class Script:
    """Plays typed lines and transcripts, recording what the student saw."""

    def __init__(self, lines, transcripts=()):
        self.lines = list(lines)
        self.transcripts = list(transcripts)
        self.shown = []
        self.recordings = 0

    def read_line(self, prompt=""):
        if not self.lines:
            raise EOFError
        return self.lines.pop(0)

    def record(self):
        self.recordings += 1
        return "audio"

    def transcribe(self, audio):
        result = self.transcripts.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def run(self):
        return read_spoken_answer(
            self.record, self.transcribe, self.shown.append, self.read_line
        )


def test_accepts_transcript_on_enter():
    script = Script(["", ""], ["tides stretch bodies"])
    assert script.run() == "tides stretch bodies"
    assert any("tides stretch bodies" in text for text in script.shown)


def test_re_record_replaces_transcript():
    script = Script(["", "r", "", ""], ["first try", "second try"])
    assert script.run() == "second try"
    assert script.recordings == 2


def test_typed_correction_replaces_transcript():
    script = Script(["", "the Roche limit"], ["the roach limit"])
    assert script.run() == "the Roche limit"


def test_typing_instead_of_speaking_skips_the_mic():
    script = Script(["done"])
    assert script.run() == "done"
    assert script.recordings == 0


def test_empty_recording_asks_again():
    script = Script(["", "", ""], ["", "got it"])
    assert script.run() == "got it"
    assert any("didn't catch" in text for text in script.shown)


def test_voice_failure_lets_student_try_again():
    script = Script(["", "", ""], [VoiceError("mic unplugged"), "got it"])
    assert script.run() == "got it"
    assert any("mic unplugged" in text for text in script.shown)


def test_end_of_input_ends_cleanly():
    assert Script([]).run() == ""


@pytest.mark.skipif(not PDF.is_file(), reason="exam PDF is not available")
def test_voice_session_passes_examiner_context_to_transcriber(monkeypatch, capsys):
    contexts = []

    class FakeTranscriber:
        def __init__(self, model_name):
            pass

        def transcribe(self, audio, context=""):
            contexts.append(context)
            return f"spoken {len(contexts)}"

    transcripts = []

    def next_turn(exam, question, transcript, *, backend, model):
        transcripts.append([m.text for m in transcript])
        finished = len(transcripts) == 2
        return ExaminerTurn(message="Are you sure?", finished=finished)

    def final_review(exam, question, transcript, *, backend, model):
        return Review(score=5, summary="Okay.", correct=[], missing=[], errors=[])

    monkeypatch.setattr("shu_exam.cli.check_microphone", lambda: None)
    monkeypatch.setattr("shu_exam.cli.Transcriber", FakeTranscriber)
    monkeypatch.setattr("shu_exam.cli.record_until_enter", lambda: "audio")
    monkeypatch.setattr("shu_exam.cli.backend_problem", lambda backend: None)
    monkeypatch.setattr("shu_exam.cli.next_turn", next_turn)
    monkeypatch.setattr("shu_exam.cli.final_review", final_review)
    monkeypatch.delenv("SHU_EXAM_BACKEND", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO("\n\n\n\n"))

    assert main(["--pdf", str(PDF), "-q", "2", "--voice"]) == 0

    assert transcripts == [["spoken 1"], ["spoken 1", "Are you sure?", "spoken 2"]]
    assert "Roche" in contexts[0] and "Are you sure?" not in contexts[0]
    assert "Are you sure?" in contexts[1]
    assert "Score: 5/10" in capsys.readouterr().out


@pytest.mark.skipif(not PDF.is_file(), reason="exam PDF is not available")
def test_missing_voice_setup_stops_before_the_session(monkeypatch, capsys):
    def no_mic():
        raise VoiceError("No microphone was found.")

    monkeypatch.setattr("shu_exam.cli.check_microphone", no_mic)
    monkeypatch.setattr("shu_exam.cli.backend_problem", lambda backend: None)
    assert main(["--pdf", str(PDF), "-q", "2", "--voice"]) == 2
    assert "No microphone" in capsys.readouterr().err
