#!/usr/bin/env python3
"""
voice — dictation for the composer: recorded audio in, text out.

    python3 voice.py             # what's installed, and whether it's ready
    python3 voice.py clip.wav    # transcribe a file, to test without a browser

Deliberately DICTATION, not conversation. You press record, say a question, press
stop; the words land in the prompt box and you send them yourself. There is no
voice-activity detection here, nothing that starts on its own, and nothing that
answers what it hears — that is the other application's job. Here the microphone is
open only between two explicit clicks.

The transcription is entirely local: faster-whisper running in this process, with the
model cached under ~/.cache/huggingface. No audio leaves the machine, and the network
is never touched after the first download.

The model is loaded ONCE and kept resident: a dictated question is a few seconds of
audio, and reloading a model per click would cost more than the transcription.
"""

import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = os.environ.get("RESEARCH_STT_MODEL", "base.en")
COMPUTE = os.environ.get("RESEARCH_STT_COMPUTE", "int8")
# Half the cores at most: dictation runs while the LLM may be generating, and the
# transcription of a ten-second clip is not worth slowing an answer down for.
THREADS = min(8, os.cpu_count() or 4)
LANGUAGE = "en"

# Words whisper should EXPECT. It conditions on this prompt, so names and jargon here
# are heard correctly instead of mangled into common words — the cheapest accuracy
# lever available, and a misheard name costs a confidently wrong answer.
HINT = os.environ.get("RESEARCH_STT_HINT",
                      "Ollama, gpt-oss, gemma, qwen, arXiv, Jetson, Strix Halo")

ENABLED = os.environ.get("RESEARCH_VOICE", "1") != "0"

_model = None
_load_lock = threading.Lock()     # loading is not reentrant
_run_lock = threading.Lock()      # a WhisperModel is not safe from two threads


def available():
    if not ENABLED:
        return False
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def load(verbose=False):
    """Load and cache. The first ever call downloads ~150 MB."""
    global _model
    with _load_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            if verbose:
                print(f"loading {MODEL} ({COMPUTE}, {THREADS} threads)…", flush=True)
            _model = WhisperModel(MODEL, device="cpu", compute_type=COMPUTE,
                                  cpu_threads=THREADS)
    return _model


# Whisper narrates non-speech rather than staying quiet: a near-silent clip comes back
# as "[BLANK_AUDIO]" or "(wind blowing)". A line that is entirely a bracketed stage
# direction was never something a person said.
_STAGE_DIRECTION = re.compile(r"^\s*[\[(][^\])]*[\])]\s*$")


def _clean(text):
    kept = [ln for ln in text.splitlines()
            if ln.strip() and not _STAGE_DIRECTION.match(ln)]
    return " ".join(" ".join(kept).split())


def _confident(seg):
    """Reject what the model itself doubts. Two seconds of silence transcribes as
    "you" with high confidence of NOT being speech — a word blacklist would be the
    wrong tool, since "you" is also a real word."""
    if getattr(seg, "no_speech_prob", 0.0) > 0.6:
        return False
    return getattr(seg, "avg_logprob", 0.0) >= -1.0


def transcribe(audio):
    """Return what was said. `audio` is 16 kHz mono S16_LE WAV bytes, or a file path."""
    model = load()
    if isinstance(audio, (bytes, bytearray)):
        import io
        audio = io.BytesIO(bytes(audio))
    with _run_lock:
        segments, _info = model.transcribe(
            audio,
            language=LANGUAGE,
            initial_prompt=HINT or None,
            beam_size=1,               # greedy: dictation wants fast over perfect
            # Without this, whisper feeds each result into the next, and one misheard
            # phrase can send it into a repeating loop. Each dictation stands alone.
            condition_on_previous_text=False,
        )
        return _clean(" ".join(s.text for s in segments if _confident(s)))


# ======================= CLI =========================

def main(argv):
    if not available():
        print("dictation is off or faster-whisper is missing:")
        print("  pip install faster-whisper        (or RESEARCH_VOICE=0 to hide the mic)")
        return 1
    if argv:
        path = argv[0]
        if not os.path.exists(path):
            print(f"no such file: {path}")
            return 1
        t0 = time.time()
        load(verbose=True)
        t1 = time.time()
        text = transcribe(path)
        print(f"\nload {t1 - t0:.1f}s · transcribe {time.time() - t1:.2f}s")
        print(f"heard: {text!r}" if text else "heard: (nothing)")
        return 0
    print(f"model    {MODEL} ({COMPUTE}, {THREADS} threads)")
    print(f"hint     {HINT}")
    t0 = time.time()
    load(verbose=True)
    print(f"ready in {time.time() - t0:.1f}s "
          f"(cached under ~/.cache/huggingface after the first run)")
    print("\ntest it:  arecord -f S16_LE -r 16000 -c 1 -d 4 /tmp/t.wav && "
          "python3 webapp/voice.py /tmp/t.wav")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
