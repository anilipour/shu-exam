"""Speak answers instead of typing them, using local Whisper transcription.

Audio is held in memory only and never leaves the machine; only the transcribed
text goes to the examiner. The audio libraries are an optional install, so they
are imported here only when voice mode is used.
"""

from __future__ import annotations

from collections.abc import Callable

DEFAULT_WHISPER_MODEL = "small.en"
_WHISPER_RATE = 16000
_INSTALL_HINT = 'Voice mode needs extra packages. Run: pip install -e ".[voice]"'

# Whisper uses this to spell field-specific terms correctly.
_GLOSSARY = (
    "Astronomy oral exam. Kepler, virial theorem, Roche limit, Jeans mass, "
    "Rayleigh-Jeans, Wien, Planck function, Schwarzschild radius, Strömgren sphere, "
    "Hertzsprung-Russell diagram, Lyman-alpha forest, Kuiper Belt, Oort cloud, ISM, "
    "IGM, CGM, parsec, redshift, metallicity, Friedmann-Robertson-Walker."
)


class VoiceError(Exception):
    """Voice input failed; the message is safe to show."""


def check_microphone() -> None:
    """Raise VoiceError if voice packages or an input device are missing."""
    try:
        import sounddevice
    except (ImportError, OSError) as e:
        raise VoiceError(_INSTALL_HINT) from e
    try:
        sounddevice.query_devices(kind="input")
    except (ValueError, sounddevice.PortAudioError) as e:
        raise VoiceError("No microphone was found.") from e


def _open_stream(sounddevice, on_audio):
    """Open the mic at Whisper's rate, or at the device's own if it refuses."""
    try:
        return sounddevice.InputStream(
            samplerate=_WHISPER_RATE, channels=1, dtype="float32", callback=on_audio
        )
    except sounddevice.PortAudioError:
        rate = sounddevice.query_devices(kind="input")["default_samplerate"]
        return sounddevice.InputStream(
            samplerate=rate, channels=1, dtype="float32", callback=on_audio
        )


def record_until_enter(read_line: Callable[[], str] = input):
    """Record from the mic until Enter, returning 16 kHz mono float32 samples."""
    import numpy
    import sounddevice

    chunks = []

    def on_audio(data, frames, time, status):
        chunks.append(data.copy())

    try:
        stream = _open_stream(sounddevice, on_audio)
        with stream:
            read_line()
    except sounddevice.PortAudioError as e:
        raise VoiceError(f"Couldn't record from the microphone: {e}") from e
    if not chunks:
        return numpy.zeros(0, dtype="float32")

    audio = numpy.concatenate(chunks).flatten()
    if stream.samplerate == _WHISPER_RATE:
        return audio
    count = int(len(audio) * _WHISPER_RATE / stream.samplerate)
    positions = numpy.linspace(0, len(audio) - 1, count)
    return numpy.interp(positions, numpy.arange(len(audio)), audio).astype("float32")


class Transcriber:
    """Local speech-to-text. Runs on the CPU, which needs no GPU libraries."""

    def __init__(self, model_name: str = DEFAULT_WHISPER_MODEL):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise VoiceError(_INSTALL_HINT) from e
        try:
            self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
        except Exception as e:
            raise VoiceError(
                f"Couldn't load the speech model {model_name}: {e}"
            ) from e

    def transcribe(self, audio, context: str = "") -> str:
        """Return the text spoken in the audio; ``context`` helps with terms."""
        try:
            segments, _ = self._model.transcribe(
                audio,
                language="en",
                vad_filter=True,
                initial_prompt=f"{_GLOSSARY} {context}".strip(),
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as e:
            raise VoiceError(f"Couldn't transcribe the recording: {e}") from e


def read_spoken_answer(
    record: Callable[[], object],
    transcribe: Callable[[object], str],
    show: Callable[[str], None],
    read_line: Callable[[str], str] = input,
) -> str:
    """Record until the student accepts a transcript; returns the final text."""
    try:
        while True:
            command = read_line(
                "Press Enter to start speaking, type your answer, or type done: "
            ).strip()
            if command:
                return command

            show("\nRecording... press Enter when you're finished.")
            try:
                audio = record()
                show("Transcribing...")
                text = transcribe(audio)
            except VoiceError as e:
                show(f"\n{e}\n")
                continue
            if not text:
                show("\nI didn't catch anything. Let's try that again.\n")
                continue

            show(f"\nYou said:\n{text}\n")
            choice = read_line(
                "Press Enter to send, r to re-record, or type a corrected version: "
            ).strip()
            if not choice:
                return text
            if choice.lower() != "r":
                return choice
            show("")
    except EOFError:
        return ""
