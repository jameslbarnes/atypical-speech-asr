"""Build a manifest for icl_asr.py from the TORGO dysarthric speakers (runs on the pod).
Usage: python make_torgo_manifest.py <torgo_root> <out.jsonl> [sentences_per_speaker] [words_per_speaker]
Keeps prompts that are plain text (drops instructions in brackets, picture prompts and 'xxx').
Head microphone when present, else the array microphone. Examples come from sentences only.
"""
import json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import soundfile as sf

root, out = Path(sys.argv[1]), sys.argv[2]
NS, NW = int(sys.argv[3]) if len(sys.argv) > 3 else 120, int(sys.argv[4]) if len(sys.argv) > 4 else 60
rows, skipped = [], Counter()
for spk in sorted(p for p in list(root.glob("*")) + list(root.glob("*/*")) if p.is_dir() and re.fullmatch(r"[FM]C?0\d", p.name)):
    for ses in sorted(spk.glob("Session*")):
        pr = ses / "prompts"
        if not pr.is_dir():
            continue
        for f in sorted(pr.glob("*.txt")):
            try:
                text = " ".join(f.read_text(encoding="utf-8", errors="ignore").split())
            except Exception:
                skipped["unreadable prompt"] += 1; continue
            if not text or "[" in text or "]" in text or ".jpg" in text.lower() or "xxx" in text.lower() or "/" in text:
                skipped["not plain text"] += 1; continue
            wav = next((w for w in (ses / "wav_headMic" / (f.stem + ".wav"), ses / "wav_arrayMic" / (f.stem + ".wav")) if w.exists()), None)
            if wav is None:
                skipped["no wav"] += 1; continue
            try:
                info = sf.info(str(wav)); dur = info.frames / info.samplerate
            except Exception:
                skipped["unreadable wav"] += 1; continue
            if not 0.3 <= dur <= 30:
                skipped["duration"] += 1; continue
            n = len(re.sub(r"[^\w' ]", " ", text).split())
            rows.append({"id": f"{spk.name}_{ses.name}_{f.stem}", "speaker": spk.name, "wav": str(wav), "text": text,
                         "dur": round(dur, 2), "kind": "sentence" if n >= 3 else "word", "mic": wav.parent.name})
rng = random.Random(7)
by = defaultdict(list)
for r in rows:
    by[r["speaker"]].append(r)
for s, v in sorted(by.items()):
    sent = [r for r in v if r["kind"] == "sentence"]; word = [r for r in v if r["kind"] == "word"]
    rng.shuffle(sent); rng.shuffle(word)
    pick = {r["id"] for r in sent[:NS]} | {r["id"] for r in word[:NW]}
    for r in v:
        r["test_ok"] = r["id"] in pick
        r["ex_ok"] = r["kind"] == "sentence"
    print(s, "sentences", len(sent), "words", len(word), "distinct sentence texts", len({r["text"].lower() for r in sent}),
          "tests", len(pick), "mics", dict(Counter(r["mic"] for r in v)))
with open(out, "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print("rows", len(rows), "skipped", dict(skipped))
