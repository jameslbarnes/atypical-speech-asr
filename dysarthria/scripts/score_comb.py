"""Hints and tuning together, live Confucius, 240 sentences from M01 M02 M04 M05. Guard: if the output repeats a
3-word run of the hint, use the same model's no-hint output."""
import json, sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "aphasia-benchmark" / "scripts"))
from score import norm as bnorm, counts, wer
N = lambda s: bnorm(s, "L3")
ids = json.load(open("logs/ids_all4.json"))
hint = {"hotright": json.load(open("logs/ctx3_hotright.json")), "hotwrong": json.load(open("logs/ctx3_hotwrong.json"))}
def load(paths):
    d = {}
    for p in glob.glob(paths):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); d[r["id"]] = r
    return d
A = {("untuned", "none"): load("results/cmp/cmp_r2t2_stream.jsonl"), ("untuned", "hotright"): load("results/ctx2/ctx2_hot.out.jsonl"),
     ("untuned", "hotwrong"): load("results/comb/untuned_hotwrong.jsonl")}
for c in ("none", "hotright", "hotwrong"):
    A[("tuned", c)] = load(f"results/comb/tuned_{c}_M0*.jsonl")
ok = [i for i in ids if all(i in d and not d[i].get("err") for d in A.values())]
def grams(t, n=3):
    w = t.split(); return {" ".join(w[j:j+n]) for j in range(len(w) - n + 1)}
spk = ["M01", "M02", "M04", "M05"]
print("sentence clips", len(ok))
for (m, c), d in A.items():
    raw, gd, echo, planted = [], [], 0, 0
    wl = {i: set(N(hint["hotwrong"][i]).split()) - {"hotwords"} for i in ok}
    for i in ok:
        h = d[i]["hyp"]; raw.append(counts(N(d[i]["ref"]), N(h)))
        if c != "none" and grams(N(h)) & grams(N(hint[c][i]).replace("hotwords ", "")):
            echo += 1; h = A[(m, "none")][i]["hyp"]
        gd.append(counts(N(d[i]["ref"]), N(h)))
        if c == "hotwrong":
            planted += len((set(N(h).split()) & wl[i]) - set(N(d[i]["ref"]).split()) - set(N(A[(m, "none")][i]["hyp"]).split()))
    ps = " ".join(f"{s}:{100*wer([gd[k] for k, i in enumerate(ok) if d[i]['speaker']==s]):.0f}" for s in spk)
    print(f"{m:8}{c:9} raw {100*wer(raw):6.1f}%  echoed {echo:3d}  guarded {100*wer(gd):5.1f}%  [{ps}]" + (f"  unsaid wrong words after guard {planted}" if c == "hotwrong" else ""))
