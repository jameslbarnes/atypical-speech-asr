"""Sound-level transcripts (IPA phones) for every utterance, on CPU.

Usage: python run_phones.py <derived_dir> [--threads 4]
Model: facebook/wav2vec2-lv-60-espeak-cv-ft (Apache-2.0), a CTC phone recognizer.
Writes <derived_dir>/hyps_phones/phones.jsonl (resumable). Output stays local.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC  # noqa: E402

MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("derived")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    derived = Path(args.derived)
    utts = [json.loads(l) for l in open(derived / "utterances.jsonl", encoding="utf-8")]
    out = derived / "hyps_phones" / "phones.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()

    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()
    from huggingface_hub import hf_hub_download
    vocab = json.load(open(hf_hub_download(MODEL, "vocab.json"), encoding="utf-8"))
    inv = {v: k for k, v in vocab.items()}
    blank = model.config.pad_token_id

    t0, n = time.time(), 0
    with open(out, "a", encoding="utf-8") as f:
        for u in utts:
            if u["id"] in done:
                continue
            wav = Path(u["wav"])
            if not wav.is_absolute():
                wav = derived / wav
            sr = sf.info(str(wav)).samplerate  # the wav is the whole recording; cut the utterance, 0.1 s pad
            a, b = max(0, int((u["start"] - 0.1) * sr)), int((u["end"] + 0.1) * sr)
            audio, sr = sf.read(str(wav), start=a, stop=b, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            assert sr == 16000, sr
            with torch.no_grad():
                ids = model(**fe(audio, sampling_rate=sr, return_tensors="pt")).logits[0].argmax(-1).tolist()
            phones, prev = [], None
            for i in ids:
                if i != prev and i != blank:
                    tok = inv.get(i, "")
                    if tok and not tok.startswith("<"):
                        phones.append(tok)
                prev = i
            f.write(json.dumps({"id": u["id"], "phones": " ".join(phones)}, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 100 == 0:
                print(f"  {n} done, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {n} new, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
