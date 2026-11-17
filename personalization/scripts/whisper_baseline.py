"""Whisper large-v3 on the test clips of a manifest, for a reference point. Usage: whisper_baseline.py <manifest> <out.jsonl>"""
import json, sys
import numpy as np, soundfile as sf, torch, librosa
from transformers import pipeline
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
rows = [r for r in rows if r.get("test_ok", True)]
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16, device="cuda")
out = open(sys.argv[2], "w", encoding="utf-8")
for i, r in enumerate(rows):
    x, sr = sf.read(r["wav"], dtype="float32")
    if x.ndim > 1: x = x.mean(axis=1)
    if sr != 16000: x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    try:
        hyp = pipe({"raw": x, "sampling_rate": 16000}, generate_kwargs={"language": "en", "task": "transcribe"})["text"]
    except Exception as e:
        hyp = ""; print("err", r["id"], repr(e)[:120], flush=True)
    out.write(json.dumps({"id": r["id"], "speaker": r["speaker"], "cond": "whisper", "k": 0, "ref": r["text"], "hyp": hyp, "kind": r["kind"], "dur": r["dur"]}) + "\n"); out.flush()
    if i % 100 == 0: print(i, len(rows), flush=True)
print("done", flush=True)
