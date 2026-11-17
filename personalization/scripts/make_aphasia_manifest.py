"""Cut the scoreable APROCSA utterances into clips and write manifests for icl_asr.py.
Output stays under <APROCSA_DIR>/derived/icl/ (never in a repo).
  manifest_L0.jsonl  example transcripts are verbatim (L0)
  manifest_L3.jsonl  example transcripts are clean (L3)
Each row also carries every reference level in `refs` for scoring.
"""
import json
import os
from pathlib import Path

import soundfile as sf

ROOT = Path(os.environ.get("APROCSA_DIR", "data/aprocsa"))
OUT = ROOT / "derived" / "icl"
(OUT / "clips").mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(ROOT / "derived" / "utterances.jsonl", encoding="utf-8")]
keep = [r for r in rows if not r["unintelligible"] and not r["overlap"] and r["timing_ok"] and 0.3 <= r["dur"] <= 30]
print("utterances", len(rows), "kept", len(keep), "ref levels", sorted(keep[0]["refs"].keys()))
handles = {}
m0, m3, n = [], [], 0
for r in keep:
    f = handles.get(r["wav"]) or handles.setdefault(r["wav"], sf.SoundFile(r["wav"]))
    sr = f.samplerate
    a, b = max(0, int((r["start"] - 0.10) * sr)), int((r["end"] + 0.10) * sr)   # 0.1 s of margin each side
    f.seek(a)
    x = f.read(b - a, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    name = f"clips/{r['id']}.wav"
    sf.write(OUT / name, x, sr, subtype="PCM_16")
    base = {"id": r["id"], "speaker": r["speaker"], "wav": name, "dur": round(len(x) / sr, 2), "refs": r["refs"],
            "meta": {"n_paraphasia": r["n_paraphasia"], "impairment": r["impairment"]}}
    for lvl, dest in (("L0", m0), ("L3", m3)):
        t = r["refs"].get(lvl) or ""
        dest.append({**base, "text": t})
    n += 1
for name, m in (("manifest_L0.jsonl", m0), ("manifest_L3.jsonl", m3)):
    with open(OUT / name, "w", encoding="utf-8", newline="\n") as fh:
        for row in m:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
import collections
print("clips", n, "per speaker", dict(collections.Counter(r["speaker"] for r in keep)))
print("clips >= 1.5 s per speaker", dict(collections.Counter(r["speaker"] for r in keep if r["dur"] >= 1.5)))
print("empty L0", sum(1 for r in m0 if not r["text"].strip()), "empty L3", sum(1 for r in m3 if not r["text"].strip()))
