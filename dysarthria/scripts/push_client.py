"""Play audio files into a recognizer sidecar at real-time pace and report what came back and when.
Usage: python push_client.py <url> <wav> [<wav>...] [--chunk-ms 160] [--language en-US] [--gap 1.0] [--quiet]"""
import argparse, json, sys, time, urllib.request
import numpy as np, soundfile as sf

def post(url, body=b""):
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, data=body, method="POST"), timeout=60).read())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("url"); ap.add_argument("wavs", nargs="+")
    ap.add_argument("--chunk-ms", type=int, default=160); ap.add_argument("--language", default=None)
    ap.add_argument("--gap", type=float, default=1.0, help="silence after each file, seconds"); ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--session", default="bench")
    a = ap.parse_args()
    q = f"session={a.session}"
    post(f"{a.url}/stream/reset?{q}" + (f"&language={a.language}" if a.language else ""))
    lat, rows, late = [], [], 0
    for w in a.wavs:
        x, sr = sf.read(w, dtype="float32")
        if x.ndim > 1: x = x.mean(axis=1)
        if sr != 16000:
            import librosa; x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        x = np.concatenate([x, np.zeros(int(a.gap * 16000), dtype=np.float32)]).astype(np.float32)
        n = int(16000 * a.chunk_ms / 1000); t0 = time.perf_counter(); finals = []; first = None
        for i in range(0, x.size, n):
            due = t0 + (i + n) / 16000  # a chunk can be sent once it has been "spoken"
            now = time.perf_counter()
            if now < due: time.sleep(due - now)
            else: late += 1
            ts = time.perf_counter()
            r = post(f"{a.url}/stream/push?{q}&final=0", x[i:i + n].tobytes())
            lat.append((time.perf_counter() - ts) * 1000)
            for e in r["events"]:
                at = time.perf_counter() - t0
                if first is None and e.get("text"): first = at
                if e["type"] == "final": finals.append((round(at, 2), e["text"], e.get("end")))
                if not a.quiet: print(f"  {at:6.2f}s {e['type']:8s} {e.get('text','')}")
        dur = (x.size / 16000) - a.gap
        rows.append({"wav": w.split("/")[-1], "dur": round(dur, 2), "first_text_s": first and round(first, 2),
                     "final_after_speech_s": finals and round(finals[-1][0] - dur, 2), "text": " ".join(f[1] for f in finals)})
        print(json.dumps(rows[-1]))
    lat.sort()
    print(json.dumps({"posts": len(lat), "post_ms_median": round(lat[len(lat)//2], 1), "post_ms_p90": round(lat[int(.9*(len(lat)-1))], 1),
                      "post_ms_max": round(lat[-1], 1), "late_posts": late}))
    print(json.dumps({k: v for k, v in json.loads(urllib.request.urlopen(a.url + "/health", timeout=10).read()).items()
                      if k in ("model", "decode_ms_median", "decode_ms_p90", "cuda_peak_gb", "cuda_allocated_gb", "gpu", "streaming_latency_ms")}))

if __name__ == "__main__":
    main()
