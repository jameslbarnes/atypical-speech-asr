"""Untuned Nemotron 3.5 (320 ms lookahead; English fixed and auto) and Whisper large-v3 on the TORGO test clips.
Usage: python eval_base.py <manifest> <out_prefix>"""
import json, sys, time
import librosa, numpy as np, soundfile as sf, torch
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]; rows = [r for r in rows if r.get("test_ok")]
def audio(r):
    x, sr = sf.read(r["wav"], dtype="float32")
    if x.ndim > 1: x = x.mean(axis=1)
    if sr != 16000: x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return x.astype(np.float32)[: 30 * 16000]
def dump(name, hyps):
    with open(f"{sys.argv[2]}_{name}.jsonl", "w", encoding="utf-8") as f:
        for r, h in zip(rows, hyps):
            f.write(json.dumps({"id": r["id"], "speaker": r["speaker"], "cond": name, "ref": r["text"], "hyp": h, "kind": r["kind"], "dur": r["dur"]}, ensure_ascii=False) + "\n")
    print(name, "done", flush=True)
from transformers import AutoModelForRNNT, AutoProcessor
M, R = "nvidia/nemotron-3.5-asr-streaming-0.6b", "ea30d66debe3740a08b573244286791d423d6b3e"
proc = AutoProcessor.from_pretrained(M, revision=R)
model = AutoModelForRNNT.from_pretrained(M, revision=R, dtype=torch.float32).cuda().eval()
for look, lang, name in ((3, "en-US", "nemotron_320_en"), (3, "auto", "nemotron_320_auto"), (13, "en-US", "nemotron_1120_en")):
    proc.set_num_lookahead_tokens(look); hyps = []
    with torch.no_grad():
        for i in range(0, len(rows), 16):
            inp = proc([audio(r) for r in rows[i:i + 16]], sampling_rate=16000, language=lang).to("cuda", dtype=torch.float32)
            o = model.generate(**inp, return_dict_in_generate=True)
            hyps += [t.strip() for t in proc.batch_decode(o.sequences, skip_special_tokens=True)]
    dump(name, hyps)
del model; torch.cuda.empty_cache()
from transformers import WhisperForConditionalGeneration, WhisperProcessor
wp = WhisperProcessor.from_pretrained("openai/whisper-large-v3", language="en", task="transcribe")
wm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float16).cuda().eval()
hyps = []
with torch.no_grad():
    for i in range(0, len(rows), 16):
        feats = wp.feature_extractor([audio(r) for r in rows[i:i + 16]], sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
        ids = wm.generate(input_features=feats, language="en", task="transcribe", max_new_tokens=120, num_beams=1)
        hyps += [t.strip() for t in wp.batch_decode(ids, skip_special_tokens=True)]
dump("whisper_large_v3", hyps)
