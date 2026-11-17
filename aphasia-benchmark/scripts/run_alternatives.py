"""Transcribe with Parakeet Ultra (ONNX, fp32) and keep the alternative tokens at every emitted step.

At each step the TDT decoder scores every token; greedy decoding keeps the best one. This records
the top 4 (blank included, i.e. "emit nothing here") with their probabilities, the same kind of
signal an on-device build exposes as alternate tokens.

Usage: python run_alternatives.py <utterances.jsonl> <out.jsonl> [--model-dir DIR] [--threads 4]
Model files: huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx (fp32/, config.json, vocab.txt),
with the two encoder data shards kept as separate files.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx_asr
import onnx_asr.asr
import onnxruntime as ort
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_asr import Audio, usable  # noqa: E402

TOP_K = 4
CAPTURE = []  # per recognize() call: list of (chosen_id, [(id, p), ...]) for each emitted token


def _decoding_with_alternatives(self, encoder_out, encoder_out_lens, /, **kwargs):
    """onnx-asr's greedy TDT loop, unchanged except that it records the top-k at each emitted token."""
    for encodings, encodings_len in zip(encoder_out, encoder_out_lens, strict=True):
        prev_state = self._create_state()
        tokens, timestamps, alts = [], [], []
        t = emitted = 0
        while t < encodings_len:
            logits, step, state = self._decode(tokens, prev_state, encodings[t])
            token = int(logits.argmax())
            if token != self._blank_idx:
                p = np.exp(logits - logits.max())
                p /= p.sum()
                top = np.argsort(-p)[:TOP_K]
                alts.append((token, [(int(i), round(float(p[i]), 4)) for i in top]))
                prev_state = state
                tokens.append(token)
                timestamps.append(t)
                emitted += 1
            if step > 0:
                t += step
                emitted = 0
            elif token == self._blank_idx or emitted == self._max_tokens_per_step:
                t += 1
                emitted = 0
        CAPTURE.append(alts)
        yield tokens, timestamps, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("utterances")
    ap.add_argument("out")
    ap.add_argument("--model-dir", default="models/parakeet-ultra-onnx")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    onnx_asr.asr._AsrWithTransducerDecoding._decoding = _decoding_with_alternatives
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    model = onnx_asr.load_model("nemo-conformer-tdt", path=args.model_dir, sess_options=so,
                                providers=["CPUExecutionProvider"])
    vocab = {}
    for line in open(Path(args.model_dir) / "vocab.txt", encoding="utf-8"):
        piece, idx = line.rstrip("\n").rsplit(" ", 1)
        vocab[int(idx)] = piece.replace("\u2581", " ")
    blank = next(i for i, piece in vocab.items() if piece == "<blk>")
    vocab[blank] = "<blank>"  # an alternative of <blank> means "emit nothing here"

    rows = [r for r in usable([json.loads(l) for l in open(args.utterances, encoding="utf-8")]) if Path(r["wav"]).exists()]
    if args.limit:
        rows = rows[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")} if out_path.exists() else set()
    audio = Audio()
    t0, audio_s = time.time(), 0.0
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(todo)} to do", flush=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for n, r in enumerate(todo, 1):
            pcm = audio.clip(r)
            CAPTURE.clear()
            t = time.time()
            text = str(model.recognize(pcm, sample_rate=16000)).strip()
            dt = time.time() - t
            audio_s += len(pcm) / 16000
            steps = CAPTURE[0] if CAPTURE else []
            tokens = [{"t": vocab.get(tok, "?"), "p": dict(top).get(tok, 0.0),
                       "alts": [[vocab.get(i, "?"), p] for i, p in top if i != tok]} for tok, top in steps]
            f.write(json.dumps({"id": r["id"], "hyp": text, "sec": round(dt, 3), "audio_sec": round(len(pcm) / 16000, 3),
                                "tokens": tokens}, ensure_ascii=False) + "\n")
            if n % 100 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  RTF {(time.time() - t0) / audio_s:.3f}", flush=True)


if __name__ == "__main__":
    main()
