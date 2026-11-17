"""Confucius4-R2T2 live loop on plain PyTorch/transformers, no vLLM engine.
The maker's streaming code calls the engine in one place: model.generate([{prompt, audio}], sampling_params).
This replaces that object with a shim over the Hugging Face model, so the same loop runs wherever PyTorch does (Windows included).
Usage: python eval_r2t2_torch.py <manifest> <ids.txt> <out.jsonl> [chunk_ms] [lookahead_ms] [unfixed]"""
import contextlib, io, json, sys, time, types, os
import librosa, numpy as np, soundfile as sf, torch
sys.path.insert(0, __import__("os").environ.get("R2T2_DIR", "r2t2"))  # checkout of the maker's Confucius4-R2T2 code
if os.environ.get("NO_VLLM"):  # prove nothing imports the real engine: stub the module the loop imports SamplingParams from
    class _SP:
        def __init__(self, **k): self.__dict__.update(k)
    sys.modules["vllm"] = types.SimpleNamespace(SamplingParams=_SP)

class Shim:
    def __init__(self, hf, proc): self.hf, self.proc, self.calls, self.secs = hf, proc, 0, 0.0
    @torch.no_grad()
    def generate(self, batch, sampling_params=None, use_tqdm=False):
        outs = []
        for inp in batch:
            a = inp["multi_modal_data"]["audio"][0]
            if isinstance(a, (tuple, list)): a = a[0]
            t = time.time()
            x = self.proc(text=[inp["prompt"]], audio=[np.asarray(a, dtype=np.float32)], return_tensors="pt", padding=True)
            x = x.to(self.hf.device).to(self.hf.dtype)
            ids = self.hf.generate(**x, max_new_tokens=int(getattr(sampling_params, "max_tokens", 4) or 4), do_sample=False)
            seq = ids.sequences if hasattr(ids, "sequences") else ids
            txt = self.proc.batch_decode(seq[:, x["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            self.calls += 1; self.secs += time.time() - t
            outs.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(text=txt)]))
        return outs

def main():
    from example import run_streaming
    from r2t2.r2t2_asr import R2T2ASRModel
    man, idf, out = sys.argv[1:4]
    chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 160
    look = int(sys.argv[5]) if len(sys.argv) > 5 else 160
    unfixed = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    ids = [l.strip() for l in open(idf) if l.strip()]
    rows = {}
    for l in open(man, encoding="utf-8"):
        r = json.loads(l); rows[r["id"]] = r
    done = set()
    try: done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")}
    except FileNotFoundError: pass
    asr = R2T2ASRModel.from_pretrained("netease-youdao/Confucius4-R2T2", dtype=torch.bfloat16, device_map="cuda:0", max_new_tokens=4)
    print("backend", asr.backend, type(asr.model).__name__, flush=True)
    shim = Shim(asr.model, asr.processor)
    asr.model, asr.backend = shim, "vllm"
    asr.sampling_params = types.SimpleNamespace(max_tokens=4)
    f = open(out, "a", encoding="utf-8"); t0 = time.time()
    for i, k in enumerate(ids):
        if k in done or k not in rows: continue
        r = rows[k]
        x, sr = sf.read(r["wav"], dtype="float32")
        if x.ndim > 1: x = x.mean(axis=1)
        if sr != 16000: x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        x = x.astype(np.float32); ts = time.time(); err = None; c0, s0 = shim.calls, shim.secs
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                hyp = run_streaming(asr, x, chunk, chunk / 1000.0, unfixed, look, language="English")
        except Exception as e:
            hyp, err = "", repr(e)[:300]
        n = max(shim.calls - c0, 1)
        f.write(json.dumps({"id": k, "speaker": r["speaker"], "cond": "r2t2_torch_stream", "ref": r["text"], "hyp": hyp or "", "kind": r["kind"], "dur": r["dur"],
                            "secs": round(time.time() - ts, 2), "calls": n, "ms_per_call": round(1000 * (shim.secs - s0) / n, 1), "err": err}, ensure_ascii=False) + "\n"); f.flush()
        if i % 50 == 0: print(i, len(ids), round(time.time() - t0), "peak_gpu_mb", torch.cuda.max_memory_allocated() // 2**20, err, flush=True)
    print("done peak_gpu_mb", torch.cuda.max_memory_allocated() // 2**20, flush=True)

if __name__ == "__main__":
    main()
