import json, sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "aphasia-benchmark" / "scripts"))
from score import norm as bnorm, counts, wer
N = lambda s: bnorm(s, "L3")
ids = json.load(open("logs/ctx2_ids.json"))
base = {json.loads(l)["id"]: json.loads(l) for l in open("results/cmp/cmp_r2t2_stream.jsonl", encoding="utf-8")}
wrongw = {i: set(N(v).split()) for i, v in json.load(open("logs/ctx2_wrongbare.json")).items()}
def grams(t, n=3):
    w = t.split(); return {" ".join(w[j:j+n]) for j in range(len(w) - n + 1)}
arms = {"no hint": base}
import os
for c in ("bare", "hot", "short", "wrongbare"):
    if not os.path.exists(f"results/ctx2/ctx2_{c}.out.jsonl"): continue
    arms[c] = {json.loads(l)["id"]: json.loads(l) for l in open(f"results/ctx2/ctx2_{c}.out.jsonl", encoding="utf-8")}
hints = {c: json.load(open(f"logs/ctx2_{c}.json")) for c in ("bare", "hot", "short", "wrongbare")}
ok = [i for i in ids if all(i in d and not d[i].get("err") for d in arms.values())]
spk = sorted({base[i]["speaker"] for i in ok})
print("sentence clips", len(ok))
for name, d in arms.items():
    raw, guarded, echo, planted = [], [], 0, 0
    for i in ok:
        h = d[i]["hyp"]; raw.append(counts(N(d[i]["ref"]), N(h)))
        if name != "no hint":
            e = bool(grams(N(h)) & grams(N(hints[name][i]).replace("hotwords ", "")))
            echo += e
            if e: h = base[i]["hyp"]
        guarded.append(counts(N(d[i]["ref"]), N(h)))
        planted += len((set(N(d[i]["hyp"]).split()) & wrongw[i]) - set(N(d[i]["ref"]).split()))
    ps = " ".join(f"{s}:{100*wer([raw[k] for k, i in enumerate(ok) if base[i]['speaker']==s]):.0f}" for s in spk)
    print(f"{name:10} raw WER {100*wer(raw):5.1f}%  echoed hint on {echo:3d}  with guard {100*wer(guarded):5.1f}%  unsaid wrong-list words {planted:4d}  [{ps}]")
