"""Score session-loop runs. Usage: <bench python> score_loop.py <level> <files...>   level: L3 normalizer; ref taken from row 'ref' unless --refs"""
import json, random, sys, statistics as st
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "aphasia-benchmark" / "scripts"))
from score import norm as bnorm, counts, wer
field = sys.argv[1]            # "ref" or a refs level such as L0 / L3 / L4
lvl = "L0" if field == "L0" else "L3"
N = lambda s: bnorm(s, lvl)
rows = []
for f in sys.argv[2:]:
    rows += [json.loads(l) for l in open(f, encoding="utf-8")]
import os
if os.environ.get("ARMS"):
    keep = set(os.environ["ARMS"].split(","))
    rows = [r for r in rows if r["arm"] in keep]
def ref(r): return r["ref"] if field == "ref" else r["refs"][field]
E = lambda x: x["S"] + x["D"] + x["I"]
by = defaultdict(dict)       # (speaker, seed, id) -> (arm, n) -> counts
info = {}
tsec = defaultdict(list); taud = defaultdict(list)
for r in rows:
    k = (r["speaker"], r["seed"], r["id"])
    by[k][(r["arm"], r["n"])] = counts(N(ref(r)), N(r["hyp"])); info[k] = r
    if "train_secs" in r: tsec[(r["arm"], r["n"])].append(r["train_secs"]); taud[(r["arm"], r["n"])].append(r["train_audio_s"])
arms = sorted({a for d in by.values() for a in d}, key=lambda a: (["base","loop","noise20","self","alt"].index(a[0]), a[1]))
spk = sorted({k[0] for k in by}); seeds = sorted({k[1] for k in by})
def table(label, filt, need_arms):
    keys = [k for k, d in by.items() if all(a in d for a in need_arms) and filt(info[k])]
    if not keys: return
    print(f"\n== {label}: clip-runs {len(keys)}, speakers {sorted({k[0] for k in keys})}")
    for a in need_arms:
        v = [by[k][a] for k in keys]
        per_seed = [100 * wer([by[k][a] for k in keys if k[1] == s]) for s in seeds if any(k[1] == s for k in keys)]
        ps = " ".join(f"{s}:{100*wer([by[k][a] for k in keys if k[0]==s]):.0f}" for s in spk if any(k[0] == s for k in keys))
        line = f"{a[0]:8}{a[1]:3} WER {100*wer(v):5.1f}%  by order [{' '.join(f'{x:.1f}' for x in per_seed)}]  [{ps}]"
        if a[0] != "base":
            rng = random.Random(4); bs = []
            for _ in range(2000):
                smp = [keys[rng.randrange(len(keys))] for _ in keys]
                n = sum(by[k][a]["N"] for k in smp) or 1
                bs.append(100 * (sum(E(by[k][a]) for k in smp) - sum(E(by[k][("base", 0)]) for k in smp)) / n)
            bs.sort()
            better = sum(1 for s in spk if any(k[0] == s for k in keys) and sum(E(by[k][a]) for k in keys if k[0] == s) < sum(E(by[k][("base", 0)]) for k in keys if k[0] == s))
            line += f"  vs untuned {100*(wer(v)-wer([by[k][('base',0)] for k in keys])):+.1f} ({bs[50]:+.1f} to {bs[1949]:+.1f}) better in {better}/{len({k[0] for k in keys})}"
        print(line)
kinds = {info[k].get("kind") for k in by}
subsets = [("all clips", lambda r: True)]
if "sentence" in kinds: subsets += [("sentences", lambda r: r.get("kind") == "sentence"), ("words", lambda r: r.get("kind") == "word")]
small = [a for a in arms if a[1] <= 10]
for nm, f in subsets:
    table(nm + " | arms up to 10 (all speakers)", f, small)
    table(nm + " | all arms (speakers with 30 arrivals)", f, arms)
print("\ntraining time (s, on a shared GPU) and audio used:")
for a in arms:
    if tsec[a]: print(f"  {a[0]:8}{a[1]:3} median train {st.median(tsec[a]):6.1f} s   median audio {st.median(taud[a]):6.1f} s")
