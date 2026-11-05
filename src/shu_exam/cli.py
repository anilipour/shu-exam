"""Command-line interface: ask a question, then run a practice oral exam."""

from __future__ import annotations

import argparse
import os
import random
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from dotenv import find_dotenv, load_dotenv

from shu_exam.document import ExamDocument, Question
from shu_exam.examiner import (
    BACKENDS,
    CLAUDE_CODE,
    EXAMINER,
    GEMINI,
    STUDENT,
    ExaminerError,
    Message,
    Review,
    backend_problem,
    final_review,
    next_turn,
)
from shu_exam.voice import (
    DEFAULT_WHISPER_MODEL,
    Transcriber,
    VoiceError,
    check_microphone,
    read_spoken_answer,
    record_until_enter,
)

WIDTH = 78
END_OF_ANSWER = "."
END_OF_SESSION = "done"

T = TypeVar("T")


def _wrap(text: str, prefix: str = "") -> str:
    return "\n".join(
        textwrap.fill(
            paragraph,
            width=WIDTH,
            initial_indent=prefix,
            subsequent_indent=" " * len(prefix),
        )
        for paragraph in text.splitlines()
        if paragraph.strip()
    )


def read_answer(read_line: Callable[[], str] = input) -> str:
    """Read lines until a lone '.', a first line of 'done', or end of input."""
    lines: list[str] = []
    while True:
        try:
            line = read_line()
        except EOFError:
            break
        if line.strip() == END_OF_ANSWER:
            break
        if not lines and line.strip().lower() == END_OF_SESSION:
            return END_OF_SESSION
        lines.append(line)
    return "\n".join(lines).strip()


def _pages(question: Question) -> str:
    if question.first_page == question.last_page:
        return f"PDF page {question.first_page}"
    return f"PDF pages {question.first_page}-{question.last_page}"


def _print_section(title: str, items: list[str]) -> None:
    if items:
        print(f"\n{title}")
        for item in items:
            print(_wrap(item, prefix="  - "))


def print_review(review: Review, question: Question) -> None:
    print(f"\nScore: {review.score}/10\n")
    print(_wrap(review.summary))
    _print_section("What you showed you understand", review.correct)
    _print_section("What to study", review.missing)
    _print_section("What to correct", review.errors)
    if question.has_reference:
        print(f"\nReference answer: {_pages(question)}.")
    else:
        print("\nThe PDF has no written answer for this question.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shu-exam",
        description="Practice a Shu Exam question with an AI oral examiner.",
    )
    parser.add_argument("-q", "--question", type=int, help="question number to ask")
    parser.add_argument(
        "-c", "--chapter", help="pick from chapters containing this text, e.g. stars"
    )
    parser.add_argument(
        "--pdf", type=Path, help="exam PDF (default: $SHU_EXAM_PDF or ./Shu_Exam.pdf)"
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        help=f"who examines you (default: $SHU_EXAM_BACKEND or {CLAUDE_CODE})",
    )
    parser.add_argument(
        "--model", help="model for the backend (default: $SHU_EXAM_MODEL or its default)"
    )
    parser.add_argument(
        "--voice", action="store_true", help="speak your answers instead of typing"
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help=f"speech model for --voice (default: {DEFAULT_WHISPER_MODEL})",
    )
    parser.add_argument(
        "--no-review", action="store_true", help="only print a question"
    )
    return parser


def _pick_question(
    exam: ExamDocument, args: argparse.Namespace, parser: argparse.ArgumentParser
) -> Question:
    if args.question is not None and args.chapter:
        parser.error("use either --question or --chapter, not both")
    if args.question is not None:
        if not 1 <= args.question <= len(exam.questions):
            parser.error(f"--question must be from 1 to {len(exam.questions)}")
        return exam.get(args.question)
    if args.chapter:
        matches = exam.filter_by_chapter(args.chapter)
        if not matches:
            chapters = sorted({q.chapter for q in exam.questions})
            parser.error(f"no chapter matches. Chapters: {'; '.join(chapters)}")
        return random.choice(matches)
    return random.choice(exam.questions)


def _with_retry(request: Callable[[], T]) -> T | None:
    """Run a request, offering retries so a failure doesn't lose the session."""
    while True:
        try:
            return request()
        except ExaminerError as e:
            print(f"\n{e}", file=sys.stderr)
        try:
            again = input("Try again? [Y/n] ").strip().lower()
        except EOFError:
            return None
        if again not in ("", "y", "yes"):
            return None


def _spoken_reader(
    transcriber: Transcriber, question: Question
) -> Callable[[str], str]:
    """Return a reader that takes what was last said to the student as context."""

    def read(examiner_said: str) -> str:
        context = f"{question.text} {examiner_said}".strip()
        return read_spoken_answer(
            record=record_until_enter,
            transcribe=lambda audio: transcriber.transcribe(audio, context),
            show=lambda text: print(
                "\n".join(textwrap.fill(line, WIDTH) for line in text.split("\n"))
            ),
        )

    return read


def _run_session(
    exam: ExamDocument,
    question: Question,
    answer: str,
    backend: str,
    model: str | None,
    read_spoken: Callable[[str], str] | None = None,
) -> int:
    transcript = [Message(STUDENT, answer)]
    explained_replies = False
    while True:
        print("\nThinking...")
        turn = _with_retry(
            lambda: next_turn(exam, question, transcript, backend=backend, model=model)
        )
        if turn is None:
            break
        print(f"\nExaminer:\n{_wrap(turn.message, prefix='  ')}\n")
        transcript.append(Message(EXAMINER, turn.message))
        if turn.finished:
            break
        if read_spoken:
            reply = read_spoken(turn.message)
        else:
            if not explained_replies:
                print(
                    f"Reply and end with a line with only {END_OF_ANSWER} "
                    f"Type {END_OF_SESSION} to finish and get your review.\n"
                )
                explained_replies = True
            reply = read_answer()
        if not reply or reply.lower() == END_OF_SESSION:
            break
        transcript.append(Message(STUDENT, reply))

    print("\nWriting your review...")
    review = _with_retry(
        lambda: final_review(exam, question, transcript, backend=backend, model=model)
    )
    if review is None:
        print("\nSession ended without a review.", file=sys.stderr)
        return 1
    print_review(review, question)
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(find_dotenv(usecwd=True))
    parser = _parser()
    args = parser.parse_args(argv)

    pdf = args.pdf or Path(os.environ.get("SHU_EXAM_PDF") or "Shu_Exam.pdf")
    if not pdf.is_file():
        print(
            f"Can't find the exam PDF at {pdf}. "
            "Use --pdf or set SHU_EXAM_PDF in .env.",
            file=sys.stderr,
        )
        return 2
    exam = ExamDocument(pdf)
    question = _pick_question(exam, args, parser)

    print(f"\nQuestion {question.number} ({question.chapter})\n")
    print(_wrap(question.text))
    print()
    if args.no_review:
        return 0

    backend = args.backend or os.environ.get("SHU_EXAM_BACKEND") or CLAUDE_CODE
    if backend not in BACKENDS:
        parser.error(f"SHU_EXAM_BACKEND must be one of: {', '.join(BACKENDS)}")
    problem = backend_problem(backend)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if backend == GEMINI:
        print(
            "Note: on Gemini's free tier, Google may use what you send to improve "
            "its products,\nand human reviewers may read it. That includes your "
            "answers and these PDF pages.\n"
        )
    if not question.has_reference:
        print(
            "Note: the PDF has no written answer for this question, so the examiner "
            "will use the\nmodel's own knowledge, not your notes.\n"
        )
    read_spoken = None
    if args.voice:
        try:
            check_microphone()
            print("Loading the speech model (the first time downloads it)...\n")
            read_spoken = _spoken_reader(Transcriber(args.whisper_model), question)
        except VoiceError as e:
            print(e, file=sys.stderr)
            return 2
    else:
        print(f"Type your answer. Enter a line with only {END_OF_ANSWER} to finish.\n")
    try:
        answer = read_spoken("") if read_spoken else read_answer()
        if not answer or answer.lower() == END_OF_SESSION:
            print("\nNo answer entered, so nothing was sent.")
            if question.has_reference:
                print(f"Reference answer: {_pages(question)}.")
            return 0
        model = args.model or os.environ.get("SHU_EXAM_MODEL") or None
        return _run_session(exam, question, answer, backend, model, read_spoken)
    except KeyboardInterrupt:
        print("\nSession cancelled.")
        return 130
