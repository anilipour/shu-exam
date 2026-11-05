# shu-exam

Practice Shu Exam questions in the terminal with an AI oral examiner. It asks a
question, you type an answer, and Claude or Gemini questions you on it using the
reference answer in your exam notes.

## Setup

You need Python 3.10+ and one of the two options below.

```powershell
git clone <this repo> shu_exam
cd shu_exam
python -m venv .venv
.venv\Scripts\Activate.ps1      # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

The exam notes, `Shu_Exam.pdf`, are included in the repo. To use a different copy,
set `SHU_EXAM_PDF` in `.env`.

## Choose your examiner

| Option | What you need | Cost | Your notes and answers |
|---|---|---|---|
| `claude-code` (default) | [Claude Code](https://code.claude.com/) installed and logged in | Counts toward your subscription limits | Handled under your Claude account's data settings |
| `gemini` | A key from [Google AI Studio](https://aistudio.google.com/api-keys), set as `GEMINI_API_KEY` in `.env` | A free tier is available | On the free tier, Google may use them to improve its products, and human reviewers may read them |

Google's terms say not to submit sensitive, confidential, or personal information
to the free tier. Check that you're comfortable sending your notes before using
it. Paid Gemini API use isn't used to improve Google's products.

To change the default, set `SHU_EXAM_BACKEND` in `.env`.

## Usage

```
shu-exam                    # random question, then a practice oral exam
shu-exam -c stars           # random question from a matching chapter
shu-exam -q 27              # a specific question
shu-exam --backend gemini   # use Gemini as the examiner
shu-exam --model sonnet     # choose a model for the backend
shu-exam --no-review        # just print a question
```

Type each answer over as many lines as you like, then enter a line containing
only `.` to submit. Submitting nothing as your first answer sends nothing.

## Speaking your answers

```powershell
pip install -e ".[voice]"
shu-exam --voice
```

Press Enter to start speaking and Enter again to stop. You then see the transcript
and can send it, re-record it, or type a correction. You can also type `done` at
any prompt to finish.

Transcription runs on your computer with Whisper, so your voice is never uploaded
and only the text goes to the examiner. The first run downloads the speech model
from Hugging Face. To trade accuracy for speed, use a smaller model, such as
`--whisper-model base.en`.

## How a session works

The examiner asks one follow-up question at a time, based on your answers so far:

- **If you got something wrong,** it challenges you without giving the correction.
- **If you left something out,** it guides you toward it with a question or hint.
- **If your answer is solid,** it asks harder questions, such as derivations,
  limiting cases, or when an assumption breaks down.

If you're stuck, it gives a stronger hint, then explains the point and moves on.

The session ends when the examiner is satisfied or when you type `done`. You then
get a score out of 10, what you showed you understand, what to study, what to
correct, and where to find the reference answer. This is where the correct answers
are stated outright.

Each turn sends the question's PDF pages and the conversation so far, so the model
can read the equations and figures. Nothing else leaves your machine.

- `claude-code` runs `claude -p` with only its file-reading tool enabled. It
  removes any `ANTHROPIC_API_KEY` from that command so it uses your login.
- `gemini` uses `gemini-3.8-flash` by default.

Questions 79-90 have no written answer in the current PDF. For those, the tool
says so before you start, and the model uses its own knowledge.

## Development

```
pytest
```

Examiner tests use fakes, so they cost nothing and don't need a login or key.
