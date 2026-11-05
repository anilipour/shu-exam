import base64
import json
import subprocess
from pathlib import Path

import httpx
import pytest
from google import genai
from google.genai import types

from shu_exam.document import Question
from shu_exam.examiner import (
    EXAMINER,
    EXAMINER_PROMPT,
    STUDENT,
    WRAP_UP_PROMPT,
    ExaminerError,
    ExaminerTurn,
    Message,
    Review,
    _ask_claude_code,
    _ask_gemini,
    backend_problem,
)

TURN = {"message": "Are you sure about that?", "finished": False}
REVIEW = {
    "score": 7,
    "summary": "Mostly right.",
    "correct": ["Kepler's second law"],
    "missing": ["Bound orbits have negative energy."],
    "errors": [],
}
TRANSCRIPT = [
    Message(STUDENT, "my answer"),
    Message(EXAMINER, "Are you sure?"),
    Message(STUDENT, "my reply"),
]


class FakeExam:
    def reference_pdf(self, question):
        return b"%PDF-fake"

    def reference_images(self, question):
        return [b"png-1", b"png-2"]


def _question(has_reference=True):
    return Question(1, "General Physics", "State Kepler's Laws.", 4, 5, has_reference)


# Claude Code backend


class FakeRun:
    def __init__(self, stdout=None, returncode=0, stderr="", error=None, output=TURN):
        success = {"is_error": False, "result": "", "structured_output": output}
        self.stdout = json.dumps(success) if stdout is None else stdout
        self.returncode = returncode
        self.stderr = stderr
        self.error = error

    def __call__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.files = {p.name: p.read_bytes() for p in Path(kwargs["cwd"]).iterdir()}
        if self.error:
            raise self.error
        return subprocess.CompletedProcess(
            command, self.returncode, self.stdout, self.stderr
        )


def _claude(run, has_reference=True, model=None, schema=ExaminerTurn):
    return _ask_claude_code(
        FakeExam(),
        _question(has_reference),
        TRANSCRIPT,
        EXAMINER_PROMPT,
        schema,
        model,
        run=run,
    )


def test_claude_code_sends_page_images_and_whole_conversation():
    run = FakeRun()

    assert _claude(run) == ExaminerTurn(**TURN)

    assert run.files == {"page_1.png": b"png-1", "page_2.png": b"png-2"}
    prompt = run.kwargs["input"]
    assert "page_1.png, page_2.png" in prompt
    assert "<student>\nmy answer\n</student>" in prompt
    assert "<examiner>\nAre you sure?\n</examiner>" in prompt
    assert prompt.index("my answer") < prompt.index("my reply")
    assert run.command[run.command.index("--tools") + 1] == "Read"
    assert run.command[run.command.index("--system-prompt") + 1] == EXAMINER_PROMPT
    schema = json.loads(run.command[run.command.index("--json-schema") + 1])
    assert set(schema["properties"]) == {"message", "finished"}
    assert "--model" not in run.command


def test_claude_code_returns_requested_schema():
    assert _claude(FakeRun(output=REVIEW), schema=Review).score == 7


def test_claude_code_uses_subscription_not_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "secret")
    run = FakeRun()
    _claude(run)
    assert "ANTHROPIC_API_KEY" not in run.kwargs["env"]
    assert "ANTHROPIC_AUTH_TOKEN" not in run.kwargs["env"]


def test_claude_code_without_reference_gets_no_files_or_tools():
    run = FakeRun()
    _claude(run, has_reference=False)
    assert run.files == {}
    assert run.command[run.command.index("--tools") + 1] == ""
    assert "no reference answer" in run.kwargs["input"]


def test_claude_code_passes_model():
    run = FakeRun()
    _claude(run, model="sonnet")
    assert run.command[run.command.index("--model") + 1] == "sonnet"


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (FakeRun(stdout="not json", returncode=1, stderr="boom"), "boom"),
        (
            FakeRun(stdout=json.dumps({"is_error": True, "result": "Not logged in"})),
            "Not logged in",
        ),
        (FakeRun(stdout=json.dumps({"is_error": False, "result": "hi"})), "cut off"),
        (FakeRun(error=FileNotFoundError()), "Can't find"),
        (FakeRun(error=subprocess.TimeoutExpired("claude", 1)), "too long"),
    ],
)
def test_claude_code_failures_raise_clear_errors(run, message):
    with pytest.raises(ExaminerError, match=message):
        _claude(run)


# Gemini backend


def _gemini_client(status, body, requests=None):
    def handler(request):
        if requests is not None:
            requests.append(request)
        return httpx.Response(
            status, headers={"content-type": "application/json"}, json=body
        )

    return genai.Client(
        api_key="test",
        http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _interaction(text=json.dumps(TURN), status="completed"):
    return {
        "id": "i1",
        "status": status,
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
    }


def _gemini(client, has_reference=True, system=EXAMINER_PROMPT, schema=ExaminerTurn):
    return _ask_gemini(
        FakeExam(),
        _question(has_reference),
        TRANSCRIPT,
        system,
        schema,
        None,
        client=client,
    )


def test_gemini_sends_reference_and_whole_conversation():
    requests = []
    turn = _gemini(_gemini_client(200, _interaction(), requests))

    assert turn == ExaminerTurn(**TURN)
    body = json.loads(requests[0].content)
    assert body["model"] == "gemini-3.8-flash"
    assert body["store"] is False
    assert body["system_instruction"] == EXAMINER_PROMPT
    assert body["response_format"]["mime_type"] == "application/json"
    assert set(body["response_format"]["schema"]["properties"]) == {"message", "finished"}
    document, text = body["input"][0]["content"]
    assert base64.b64decode(document["data"]) == b"%PDF-fake"
    assert "<student>\nmy answer\n</student>" in text["text"]
    assert "<student>\nmy reply\n</student>" in text["text"]


def test_gemini_returns_requested_schema():
    client = _gemini_client(200, _interaction(text=json.dumps(REVIEW)))
    assert _gemini(client, system=WRAP_UP_PROMPT, schema=Review).score == 7


def test_gemini_without_reference_sends_no_document():
    requests = []
    _gemini(_gemini_client(200, _interaction(), requests), has_reference=False)
    content = json.loads(requests[0].content)["input"][0]["content"]
    assert [c["type"] for c in content] == ["text"]
    assert "no reference answer" in content[0]["text"]


@pytest.mark.parametrize(
    ("status", "message"),
    [(401, "API key"), (404, "model"), (429, "quota"), (500, "failed")],
)
def test_gemini_api_errors_raise_clear_errors(status, message):
    client = _gemini_client(status, {"error": {"code": status, "message": "x"}})
    with pytest.raises(ExaminerError, match=message):
        _gemini(client)


@pytest.mark.parametrize(
    "body",
    [_interaction(text='{"message": "Are yo'), _interaction(status="incomplete")],
)
def test_gemini_incomplete_response_raises_clear_error(body):
    with pytest.raises(ExaminerError, match="cut off"):
        _gemini(_gemini_client(200, body))


# Checks before the student starts typing


def test_backend_problem_reports_missing_gemini_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert "GEMINI_API_KEY" in backend_problem("gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert backend_problem("gemini") is None


def test_backend_problem_reports_missing_claude_cli(monkeypatch):
    monkeypatch.setattr("shu_exam.examiner.shutil.which", lambda name: None)
    assert "Claude Code CLI" in backend_problem("claude-code")
    monkeypatch.setattr("shu_exam.examiner.shutil.which", lambda name: "/bin/claude")
    assert backend_problem("claude-code") is None
