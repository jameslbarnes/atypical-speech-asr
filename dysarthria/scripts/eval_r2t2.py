"""Confucius4-R2T2 on the TORGO test clips. Usage (inside the r2t2 env, from the repo dir):
python eval_r2t2.py <manifest> <out.jsonl> stream|offline [chunk_ms] [lookahead_ms] [unfixed]"""
import contextlib, io, json, sys, time
import librosa, numpy as np, soundfile as sf
sys.path.insert(0, __import__("os").environ.get("R2T2_DIR", "r2t2"))  # checkout of the maker's Confucius4-R2T2 code

def main():
    import example as ex
    man, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 160
    look = int(sys.argv[5]) if len(sys.argv) > 5 else 160
    unfixed = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    rows = [json.loads(l) for l in open(man, encoding="utf-8")]
    rows = [r for r in rows if r.get("test_ok")]
    import os
    done = set()
    try:
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")}
    except FileNotFoundError:
        pass
    M = os.environ.get("R2_MODEL", "netease-youdao/Confucius4-R2T2")
    if os.environ.get("R2_IDS"):
        keep = set(json.load(open(os.environ["R2_IDS"]))); rows = [r for r in rows if r["id"] in keep]
    CTX = json.load(open(os.environ["R2_CTX"])) if os.environ.get("R2_CTX") else {}
    asr = ex.init_stream_asr_model(M) if mode == "stream" else ex.init_onetime_asr_model(M)
    f = open(out, "a", encoding="utf-8"); t0 = time.time()
    for i, r in enumerate(rows):
        if r["id"] in done: continue
        x, sr = sf.read(r["wav"], dtype="float32")
        if x.ndim > 1: x = x.mean(axis=1)
        if sr != 16000: x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        x = x.astype(np.float32); ts = time.time(); err = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                if mode == "stream":
                    hyp = ex.run_streaming(asr, x, chunk, chunk / 1000.0, unfixed, look, language="English", context=CTX.get(r["id"], ""))
                else:
                    hyp = ex.run_onetime(asr, x, language="English")
        except Exception as e:
            hyp, err = "", repr(e)[:200]
        f.write(json.dumps({"id": r["id"], "speaker": r["speaker"], "cond": f"r2t2_{mode}", "ref": r["text"], "hyp": hyp or "",
                            "kind": r["kind"], "dur": r["dur"], "secs": round(time.time() - ts, 2), "err": err}, ensure_ascii=False) + "\n"); f.flush()
        if i % 100 == 0: print(i, len(rows), round(time.time() - t0), flush=True)
    print("done", flush=True)

if __name__ == "__main__":
    main()
