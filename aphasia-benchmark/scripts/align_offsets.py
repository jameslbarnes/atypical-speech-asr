"""Re-time CHAT utterances against the published audio using ASR word anchors.

The published APROCSA videos were edited for de-identification, so CHAT bullets after an
edit can point at the wrong audio. This transcribes each full recording with Parakeet
(word timestamps, 30 s windows), aligns that word stream to the CHAT %wor tier, and uses
runs of 3+ matching words as anchors to estimate the local offset for every utterance.

Usage: python align_offsets.py <data_dir>
  caches <data_dir>/derived/longform/<spk>.json (ASR words with times)
  writes <data_dir>/derived/utterances_aligned.jsonl: utterances.jsonl plus offset / offset_n,
         with start / end corrected (original bullet times kept as chat_start / chat_end)
"""

import difflib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import BULLET, SPEAKERS  # noqa: E402

GGUF = Path(os.environ.get("GGUF_DIR", "models/gguf")) / "parakeet-tdt-0.6b-v2-F16.gguf"
WINDOW = 30.0


def w_norm(w):
    w = w.lower().replace("_", " ").replace("+", " ")
    w = re.sub(r"@[a-z:]+$", "", w)
    w = re.sub(r"[^a-z' ]", "", w).strip("' ")
    return w


def longform(wav, cache):
    if cache.exists():
        return json.loads(cache.read_text())
    import transcribe_cpp
    pcm, sr = sf.read(wav, dtype="float32")
    model = transcribe_cpp.Model(str(GGUF), backend="cpu")
    session = model.session(n_threads=4)
    words = []
    for k in range(0, len(pcm), int(WINDOW * sr)):
        seg = np.ascontiguousarray(pcm[k: k + int(WINDOW * sr)])
        if len(seg) < sr:
            continue
        res = session.run(seg, timestamps="word")
        for w in res.words:
            for part in w_norm(w.text).split():
                words.append([part, round(k / sr + w.t0_ms / 1000, 3)])
    cache.write_text(json.dumps(words))
    return words


def chat_words(cha):
    """(word, time) from %wor tiers: each word is followed by its \x15start_end\x15 bullet."""
    out = []
    lines = cha.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("%wor:"):
            text = lines[i][5:]
            while i + 1 < len(lines) and lines[i + 1].startswith("\t"):
                i += 1
                text += " " + lines[i].strip()
            for m in re.finditer(r"(\S+)\s*\x15(\d+)_(\d+)\x15", text):
                for part in w_norm(m.group(1)).split():
                    out.append([part, int(m.group(2)) / 1000])
        i += 1
    return out


def anchors(cw, aw):
    sm = difflib.SequenceMatcher(None, [w for w, _ in cw], [w for w, _ in aw], autojunk=False)
    pairs = []
    for a, b, size in sm.get_matching_blocks():
        if size >= 3:
            pairs += [(cw[a + k][1], aw[b + k][1] - cw[a + k][1]) for k in range(size)]
    return np.array(sorted(pairs))


def local_offset(pairs, start, end):
    inside = pairs[(pairs[:, 0] >= start - 2) & (pairs[:, 0] <= end + 2)]
    if len(inside) >= 3:
        return float(np.median(inside[:, 1])), len(inside)
    near = pairs[np.argsort(np.abs(pairs[:, 0] - (start + end) / 2))[:15]]
    return float(np.median(near[:, 1])), 0


def main(data_dir):
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    data = Path(data_dir)
    (data / "derived" / "longform").mkdir(exist_ok=True)
    rows = [json.loads(l) for l in open(data / "derived" / "utterances.jsonl", encoding="utf-8")]
    for spk in SPEAKERS:
        aw = longform(data / "wav" / f"{spk}.wav", data / "derived" / "longform" / f"{spk}.json")
        cw = chat_words(data / "cha" / f"aprocsa{spk}a.cha")
        pairs = anchors(cw, aw)
        mine = [r for r in rows if r["speaker"] == spk]
        for r in mine:
            r.setdefault("chat_start", r["start"])
            r.setdefault("chat_end", r["end"])
            off, n = local_offset(pairs, r["chat_start"], r["chat_end"])
            off = 0.0 if abs(off) < 0.25 else round(off, 2)  # ignore jitter from word-boundary differences
            r["offset"], r["offset_n"] = off, n
            r["start"], r["end"] = round(r["chat_start"] + off, 3), round(r["chat_end"] + off, 3)
        moved = [r for r in mine if r["offset"]]
        mins = {}
        for t, o in pairs:
            mins.setdefault(int(t // 60), []).append(o)
        timeline = " ".join(f"{m}m:{np.median(v):+.1f}" for m, v in sorted(mins.items()) if m % 3 == 0)
        print(f"{spk}: {len(cw)} chat words, {len(aw)} asr words, {len(pairs)} anchored; "
              f"{len(moved)}/{len(mine)} utterances shifted\n   {timeline}", flush=True)
    with open(data / "derived" / "utterances_aligned.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
