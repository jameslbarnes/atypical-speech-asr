"""Transcribe the usable APROCSA utterances with one ASR model (CPU only).

Usage:
  python run_asr.py <model_key> <utterances.jsonl> <out_dir> [--threads 8] [--limit N]

Runs at idle priority so it only uses CPU the live stack leaves free. Resumable: ids
already present in <out_dir>/<model_key>.jsonl are skipped. Besides the transcript,
it stores whatever confidence signal the runtime exposes (word/token probabilities)
for a later confidence-guided LLM correction layer.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

GGUF_DIR = Path(os.environ.get("GGUF_DIR", "models/gguf"))

MODELS = {
    # key: (runtime, model id / file). transcribe.cpp GGUFs are from huggingface.co/handy-computer.
    "whisper-large-v3": ("transcribe.cpp", "whisper-large-v3-Q8_0.gguf"),
    "whisper-large-v3-turbo": ("transcribe.cpp", "whisper-large-v3-turbo-Q8_0.gguf"),
    "whisper-large-v3-turbo-fw": ("faster-whisper", "large-v3-turbo"),  # runtime cross-check (CTranslate2 int8)
    "parakeet-tdt-0.6b-v2": ("transcribe.cpp", "parakeet-tdt-0.6b-v2-F16.gguf"),
    "parakeet-tdt-0.6b-v3": ("transcribe.cpp", "parakeet-tdt-0.6b-v3-F16.gguf"),
    "parakeet-ultra-0.6b": ("transcribe.cpp", "parakeet-ultra-F16.gguf"),  # Moondream post-train of v3
    "canary-1b-v2": ("transcribe.cpp", "canary-1b-v2-F16.gguf"),
    "cohere-transcribe-2b": ("transcribe.cpp", "cohere-transcribe-03-2026-Q8_0.gguf"),
    "canary-qwen-2.5b": ("transcribe.cpp", "canary-qwen-2.5b-Q8_0.gguf"),
    "granite-speech-4.1-2b": ("transcribe.cpp", "granite-speech-4.1-2b-Q8_0.gguf"),
    "nemotron-streaming-0.6b": ("transcribe.cpp", "nemotron-speech-streaming-en-0.6b-F16.gguf"),
    "voxtral-mini-3b": ("transcribe.cpp", "Voxtral-Mini-3B-2507-Q8_0.gguf"),
}

PAD = 0.1  # seconds of context on each side of the CHAT bullet
# Streaming models expect audio to keep coming; without a short silent tail the last chunk is never
# flushed and short utterances come back empty (54 of 130 empties recovered in a test). Live use has that tail.
STREAMING_TAIL = {"nemotron-streaming-0.6b": 0.5}


def usable(rows, min_dur=0.3, max_dur=30.0):
    return [r for r in rows
            if not r["unintelligible"] and not r["overlap"] and r.get("timing_ok", True) and r["refs"]["L0"]
            and min_dur <= r["dur"] <= max_dur]


class Audio:
    def __init__(self):
        self.cache = {}

    def clip(self, row):
        if row["wav"] not in self.cache:
            pcm, sr = sf.read(row["wav"], dtype="float32")
            assert sr == 16000 and pcm.ndim == 1
            self.cache[row["wav"]] = pcm
        pcm = self.cache[row["wav"]]
        a = max(0, int((row["start"] - PAD) * 16000))
        b = min(len(pcm), int((row["end"] + PAD) * 16000))
        return np.ascontiguousarray(pcm[a:b])


def make_transcriber(key, threads):
    runtime, name = MODELS[key]
    if runtime == "faster-whisper":
        from faster_whisper import WhisperModel
        model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads)

        def run(pcm):
            segs, info = model.transcribe(pcm, language="en", beam_size=5, word_timestamps=True)
            segs = list(segs)
            words = [{"w": w.word.strip(), "p": round(w.probability, 4)} for s in segs for w in (s.words or [])]
            meta = [{"avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3),
                     "compression_ratio": round(s.compression_ratio, 2)} for s in segs]
            return " ".join(s.text.strip() for s in segs).strip(), {"words": words, "segments": meta}
        return run

    if runtime == "transcribe.cpp":
        import transcribe_cpp
        model = transcribe_cpp.Model(str(GGUF_DIR / name), backend="cpu")
        session = model.session(n_threads=threads)
        lang = "en" if "en" in (model.capabilities.languages or ()) else None

        def run(pcm):
            try:
                res = session.run(pcm, language=lang) if lang else session.run(pcm)
            except transcribe_cpp.InvalidArgument:
                res = session.run(pcm)
            toks = [{"t": t.text, "p": round(t.p, 4)} for t in (res.tokens or [])]
            return res.text.strip(), {"tokens": toks}
        return run

    raise ValueError(runtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("utterances")
    ap.add_argument("out_dir")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)

    rows = usable([json.loads(l) for l in open(args.utterances, encoding="utf-8")])
    rows = [r for r in rows if Path(r["wav"]).exists()]
    if args.limit:
        rows = rows[: args.limit]
    out_path = Path(args.out_dir) / f"{args.model}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{args.model}: {len(todo)} to do ({len(done)} already done)", flush=True)
    if not todo:
        return

    t_load = time.time()
    run = make_transcriber(args.model, args.threads)
    print(f"loaded in {time.time() - t_load:.1f}s", flush=True)

    audio = Audio()
    tail = np.zeros(int(STREAMING_TAIL.get(args.model, 0) * 16000), dtype=np.float32)
    t0, audio_s = time.time(), 0.0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            pcm = audio.clip(r)
            if len(tail):
                pcm = np.concatenate([pcm, tail])
            t = time.time()
            try:
                hyp, extra = run(pcm)
                err = None
            except Exception as e:  # keep going; record the failure
                hyp, extra, err = "", {}, f"{type(e).__name__}: {e}"
            dt = time.time() - t
            audio_s += len(pcm) / 16000
            f.write(json.dumps({"id": r["id"], "hyp": hyp, "sec": round(dt, 3), "audio_sec": round(len(pcm) / 16000, 3),
                                "error": err, **extra}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  RTF {el / audio_s:.3f}  elapsed {el / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
