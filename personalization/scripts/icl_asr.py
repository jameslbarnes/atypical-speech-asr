"""In-context speaker personalization for speech recognition with Phi-4 Multimodal.

Follows Roll et al. 2025 (arXiv 2505.14887): k audio clips from a speaker with their
transcripts go in the prompt before the clip to transcribe; no training.

Sources:
  hf:<dataset>[:split]      a Hugging Face dataset with audio, speaker and transcript fields
  manifest:<file.jsonl>     rows {"id","speaker","wav","text", ...}; wav paths relative to the file

Conditions per test clip:
  zero           no examples
  same           k examples from the same speaker
  diff           k examples from one other speaker (controls for prompt format)
  text           the same-speaker examples' transcripts with no audio (controls for vocabulary leak)

Output: one JSON line per (clip, condition, k). Resumable.
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

MODEL = "microsoft/Phi-4-multimodal-instruct"
U, A, E = "<|user|>", "<|assistant|>", "<|end|>"


def norm(t):
    t = (t or "").lower()
    t = re.sub(r"[.,?!;:\"'()\[\]]", " ", t)
    return " ".join(t.split())


def load_model():
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except Exception:
        attn = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype="auto", _attn_implementation=attn
    ).cuda().eval()
    gen = GenerationConfig.from_pretrained(MODEL)
    print("attention:", attn, flush=True)
    return model, proc, gen


def build_prompt(k, cond, examples, frame):
    """Returns (prompt, list of example indexes whose audio is included)."""
    if cond == "zero":
        return f"{U}<|audio_1|>Transcribe the audio clip into text.{E}{A}", []
    ref = "the same speaker" if cond in ("same", "text", "samev") else "a different speaker"
    n = "an example" if k == 1 else f"{k} examples"
    if cond == "text":
        p = (f"{U}I'll provide {n} of correct transcriptions of speech from {frame}. "
             f"Then I'll give you a new audio from {ref} to transcribe.{E}"
             f"{A}I understand. I'll read the examples and use them to accurately transcribe the final audio.{E}")
        for ex in examples[:k]:
            p += f"{U}Here is a transcription:{E}{A}{ex['text']}{E}"
        p += f"{U}<|audio_1|>Please transcribe this audio from {ref}:{E}{A}"
        return p, []
    p = (f"{U}I'll provide {n} of speech from {frame}, followed by the correct transcription. "
         f"Then I'll give you a new audio from {ref} to transcribe.{E}"
         f"{A}I understand. I'll listen to the {'example' if k == 1 else 'examples'} and use "
         f"{'it' if k == 1 else 'them'} to accurately transcribe the final audio.{E}")
    for i, ex in enumerate(examples[:k]):
        p += f"{U}<|audio_{i+1}|>Transcribe this audio:{E}{A}{ex['text']}{E}"
    if cond == "samev":
        p += (f"{U}<|audio_{k+1}|>This is a new recording from {ref}. It says something different from every example above. "
              f"Write only the words spoken in this last recording, even if it is a single word:{E}{A}")
    else:
        p += f"{U}<|audio_{k+1}|>Please transcribe this audio from {ref}:{E}{A}"
    return p, list(range(k))


def to16k(x, sr):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return x


def decode_bytes(b):
    """Decode with ffmpeg's libraries (PyAV): some FLAC files here have headers libsndfile cannot read."""
    import io
    if len(b) < 2000:
        raise ValueError("empty audio file")
    try:
        x, sr = sf.read(io.BytesIO(b), dtype="float32")
        if len(x):
            return x, sr
    except Exception:
        pass
    import av
    out = []
    with av.open(io.BytesIO(b)) as c:
        st = c.streams.audio[0]
        sr = st.rate
        rs = av.AudioResampler(format="flt", layout="mono", rate=sr)
        for fr in c.decode(st):
            for o in rs.resample(fr):
                out.append(o.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.float32), sr


class Source:
    def __init__(self, spec):
        kind, rest = spec.split(":", 1)
        self.rows = []
        if kind == "hf":
            from datasets import load_dataset, Audio
            parts = rest.split(":")
            ds = load_dataset(parts[0], split=parts[1] if len(parts) > 1 else "train")
            cols = ds.column_names
            tf = next(c for c in ("transcript", "text", "sentence", "transcription") if c in cols)
            sp = next(c for c in ("speaker", "speaker_id", "speakerId", "speaker_ID") if c in cols)
            self.ds = ds.cast_column("audio", Audio(decode=False))
            meta = self.ds.remove_columns([c for c in cols if c not in (tf, sp)])
            for i, r in enumerate(meta):
                self.rows.append({"id": f"{i}", "speaker": str(r[sp]), "text": r[tf], "i": i})
            self.kind = "hf"
        else:
            base = Path(rest).parent
            for line in open(rest, encoding="utf-8"):
                r = json.loads(line)
                r["wav"] = str((base / r["wav"]).resolve()) if not os.path.isabs(r["wav"]) else r["wav"]
                self.rows.append(r)
            self.kind = "manifest"
        self.cache = {}

    def audio(self, row):
        if row["id"] in self.cache:
            return self.cache[row["id"]]
        if self.kind == "hf":
            import io
            a = self.ds[row["i"]]["audio"]
            x, sr = decode_bytes(a["bytes"]) if a.get("bytes") else sf.read(a["path"])
        else:
            x, sr = sf.read(row["wav"])
        x = to16k(x, sr)
        if len(self.cache) < 4000:
            self.cache[row["id"]] = x
        return x


def plan(src, args):
    """Deterministic trial plan: test clips and their example sets."""
    rng = random.Random(args.seed)
    by = {}
    for r in src.rows:
        if "dur" not in r:
            try:
                r["dur"] = len(src.audio(r)) / 16000 if args.measure_dur else 99
            except Exception:
                r["dur"] = 0  # unreadable or empty clip: excluded by the duration filters
        by.setdefault(r["speaker"], []).append(r)
    kmax = max(args.shots)
    speakers = sorted(s for s, v in by.items() if len(v) >= kmax + 2)
    if args.max_speakers:
        speakers = rng.sample(speakers, min(args.max_speakers, len(speakers)))
    trials = []
    for s in sorted(speakers):
        ex_ok = [r for r in by[s] if args.min_ex_dur <= r["dur"] <= args.max_ex_dur and norm(r["text"]) and r.get("ex_ok", True)]
        tests = [r for r in by[s] if args.min_test_dur <= r["dur"] <= args.max_test_dur and norm(r["text"]) and r.get("test_ok", True)]
        rng.shuffle(tests)
        for t in tests[: args.trials]:
            pool = [r for r in ex_ok if r["id"] != t["id"] and norm(r["text"]) != norm(t["text"])]
            if len(pool) < kmax:
                continue
            same = rng.sample(pool, kmax)
            others = [o for o in speakers if o != s]
            diff = []
            rng.shuffle(others)
            for o in others:
                opool = [r for r in by[o] if args.min_ex_dur <= r["dur"] <= args.max_ex_dur
                         and norm(r["text"]) and norm(r["text"]) != norm(t["text"]) and r.get("ex_ok", True)]
                if len(opool) >= kmax:
                    diff = rng.sample(opool, kmax)
                    break
            trials.append({"test": t, "same": same, "diff": diff})
    return trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shots", type=lambda s: [int(x) for x in s.split(",")], default=[12])
    ap.add_argument("--conds", type=lambda s: s.split(","), default=["zero", "same", "diff"])
    ap.add_argument("--trials", type=int, default=20, help="test clips per speaker")
    ap.add_argument("--max-speakers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frame", default="a non-native English speaker")
    ap.add_argument("--min-ex-dur", type=float, default=2.5)
    ap.add_argument("--max-ex-dur", type=float, default=15.0)
    ap.add_argument("--min-test-dur", type=float, default=2.5)
    ap.add_argument("--max-test-dur", type=float, default=30.0)
    ap.add_argument("--measure-dur", action="store_true", help="measure durations when the source has none")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-speakers", default="", help="run only these test speakers (examples may still come from others)")
    args = ap.parse_args()

    args.frame = args.frame.replace('_', ' ')
    src = Source(args.source)
    trials = plan(src, args)
    if args.only_speakers:
        only = set(args.only_speakers.split(","))
        trials = [t for t in trials if t["test"]["speaker"] in only]
    if args.limit:
        trials = trials[: args.limit]
    print("rows", len(src.rows), "trials", len(trials), flush=True)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["id"], r["cond"], r["k"]))
    model, proc, gen = load_model()
    out = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    n = 0
    for ti, tr in enumerate(trials):
        t = tr["test"]
        jobs = [("zero", 0)] if "zero" in args.conds else []
        for c in args.conds:
            if c != "zero":
                jobs += [(c, k) for k in args.shots]
        for cond, k in jobs:
            if (t["id"], cond, k) in done:
                continue
            ex = tr["diff"] if cond == "diff" else tr["same"]
            if cond != "zero" and len(ex) < k:
                continue
            prompt, idx = build_prompt(k, cond, ex, args.frame)
            audios = [(src.audio(ex[i]), 16000) for i in idx] + [(src.audio(t), 16000)]
            ts = time.time()
            try:
                inputs = proc(text=prompt, audios=audios, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                         num_beams=1, generation_config=gen, num_logits_to_keep=1)
                ids = ids[:, inputs["input_ids"].shape[1]:]
                hyp = proc.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                err = None
            except Exception as e:  # keep going; record the failure
                hyp, err = "", repr(e)[:300]
                torch.cuda.empty_cache()
            rec = {"id": t["id"], "speaker": t["speaker"], "cond": cond, "k": k, "ref": t["text"], "hyp": hyp,
                   "ex_ids": [e["id"] for e in ex[:k]] if cond != "zero" else [],
                   "ex_speaker": ex[0]["speaker"] if (cond != "zero" and ex) else None,
                   "ex_audio_s": round(sum(len(a) for a, _ in audios[:-1]) / 16000, 1),
                   "secs": round(time.time() - ts, 2), "err": err}
            for extra in ("refs", "meta", "kind", "dur"):
                if extra in t:
                    rec[extra] = t[extra]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            n += 1
        if ti % 10 == 0:
            print(f"trial {ti+1}/{len(trials)}  runs {n}  {time.time()-t0:.0f}s", flush=True)
    print("done", n, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
