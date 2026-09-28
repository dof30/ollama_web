#!/usr/bin/env python3
"""
speech — the answer, spoken: text in, WAV out.

    python3 speech.py                 # what's installed, and whether it's ready
    python3 speech.py "say this"      # synthesise to /tmp, to test without a browser
    python3 speech.py --voices        # list the voices you can pick from

The mirror of voice.py. That one turns your recording into a question; this one turns
the model's answer into sound, sentence by sentence while it is still being written —
so listening starts on the first sentence instead of after the last.

This module is handed ONE ready-to-speak sentence at a time and knows nothing about
where it came from. Deciding what is speakable — stripping markdown, skipping code
blocks, and above all never speaking the model's thinking — belongs to the page, which
is the only place that can tell an answer from a research step as it streams.

Kokoro-82M, run locally through onnxruntime on the CPU, measured here at 6-7x realtime:
comfortably faster than the model writes, which is the whole requirement. The weights
are the same ~350 MB pair the speak-it-out application downloaded; this reads them
where they already are rather than keeping a second copy.
"""

import io
import os
import re
import sys
import threading
import time
import wave

# The speak-it-out application downloaded these; a 325 MB file is not worth a second
# copy. If it is ever uninstalled, available() goes False and the app simply loses its
# voice rather than breaking — same contract as faster-whisper in voice.py.
CANDIDATES = [
    os.path.expanduser("~/.local/share/research-web/kokoro"),
    os.path.expanduser("~/.local/share/speak-it-out/kokoro"),
]


def _find(name, env):
    override = os.environ.get(env)
    if override:
        return override
    for d in CANDIDATES:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(CANDIDATES[0], name)


MODEL = _find("kokoro-v1.0.onnx", "RESEARCH_TTS_MODEL")
VOICES = _find("voices-v1.0.bin", "RESEARCH_TTS_VOICES")
VOICE = os.environ.get("RESEARCH_TTS_VOICE", "af_heart")
SPEED = float(os.environ.get("RESEARCH_TTS_SPEED", "1.0"))
RATE = 24000                      # what Kokoro emits; not a preference

ENABLED = os.environ.get("RESEARCH_SPEECH", "1") != "0"

# One sentence, capped. The page sends a sentence at a time, so this is a guard against
# a pathological one (a table row, a minified URL) rather than a normal limit.
MAX_CHARS = 1200

# ONE model for the whole process, guarded. Learned the hard way in the other app:
# building a synthesiser per request loaded 310 MB each time AND raced inside the
# phonemiser, which surfaced as "number of lines in input and output must be equal".
_kokoro = None
_load_lock = threading.Lock()
_run_lock = threading.Lock()


def available():
    if not ENABLED:
        return False
    if not (os.path.exists(MODEL) and os.path.exists(VOICES)):
        return False
    try:
        import kokoro_onnx  # noqa: F401
        return True
    except Exception:
        return False


def load(verbose=False):
    """Load and cache. ~0.6s, once per server run."""
    global _kokoro
    with _load_lock:
        if _kokoro is None:
            from kokoro_onnx import Kokoro
            if verbose:
                print(f"loading {os.path.basename(MODEL)}…", flush=True)
            _kokoro = Kokoro(MODEL, VOICES)
    return _kokoro


def voices():
    return sorted(load().get_voices())


# Kokoro takes a fixed number of phoneme tokens and quietly truncates past it, which
# loses the end of a long sentence with no error at all. Splitting at clause boundaries
# keeps every word AND puts the seams where a speaker would breathe anyway.
_CLAUSE = re.compile(r"(?<=[,;:—])\s+")


def _pieces(text, max_chars=220):
    out, buf = [], ""
    for part in _CLAUSE.split(text):
        if buf and len(buf) + len(part) + 1 > max_chars:
            out.append(buf)
            buf = part
        else:
            buf = f"{buf} {part}".strip()
    if buf:
        out.append(buf)
    # A single clause longer than the cap (a URL, a pasted table row) still has to go
    # somewhere: hard-wrap it rather than hand Kokoro something it will silently trim.
    wrapped = []
    for p in out:
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars) or max_chars
            wrapped.append(p[:cut])
            p = p[cut:].lstrip()
        if p:
            wrapped.append(p)
    return wrapped


def synth(text, voice=None):
    """Return one sentence as 24 kHz mono 16-bit WAV bytes, or b"" for nothing to say."""
    import numpy as np

    text = (text or "").strip()[:MAX_CHARS]
    if not text:
        return b""
    k = load()
    chosen = voice or VOICE
    chunks = []
    for piece in _pieces(text):
        with _run_lock:
            samples, rate = k.create(piece, voice=chosen, speed=SPEED, lang="en-us")
        chunks.append((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    if not chunks:
        return b""

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(chunks))
    return buf.getvalue()


# ======================= CLI =========================

def main(argv):
    if not available():
        print("speech is off, or Kokoro is missing:")
        print(f"  model  {MODEL}  {'✓' if os.path.exists(MODEL) else '✗ missing'}")
        print(f"  voices {VOICES}  {'✓' if os.path.exists(VOICES) else '✗ missing'}")
        print("  pip install kokoro-onnx      (or RESEARCH_SPEECH=0 to hide the switch)")
        return 1
    if argv and argv[0] == "--voices":
        names = voices()
        print(f"{len(names)} voices (RESEARCH_TTS_VOICE=... to choose):")
        for i in range(0, len(names), 6):
            print("  " + "  ".join(names[i:i + 6]))
        return 0
    if argv:
        t0 = time.time()
        load(verbose=True)
        t1 = time.time()
        wav = synth(" ".join(argv))
        el = time.time() - t1
        secs = (len(wav) - 44) / 2 / RATE
        out = "/tmp/research-speech.wav"
        with open(out, "wb") as f:
            f.write(wav)
        print(f"\nload {t1 - t0:.1f}s · synth {el:.2f}s -> {secs:.2f}s audio "
              f"({secs / el:.1f}x realtime)")
        print(f"wrote {out} — play it:  aplay {out}")
        return 0
    print(f"model  {MODEL}")
    print(f"voice  {VOICE} at {SPEED}x  ({RATE} Hz)")
    t0 = time.time()
    load(verbose=True)
    print(f"ready in {time.time() - t0:.1f}s")
    print('\ntest it:  python3 webapp/speech.py "the answer, spoken aloud"')
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
