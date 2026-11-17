import json, sys, glob, random
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "aphasia-benchmark" / "scripts"))
from score import norm as bnorm, counts, wer
N = lambda s: bnorm(s, "L3")
data = {}
for f in sorted(glob.glob(sys.argv[1] + "/cmp_*.jsonl")):
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    if rows: data[rows[0]["cond"]] = {r["id"]: r for r in rows if not r.get("err")}
ids = set.intersection(*[set(d) for d in data.values()])
any_ = next(iter(data.values()))
print("systems", list(data), "clips scored on all:", len(ids))
def grp(i): return "control" if "C" in any_[i]["speaker"] else "dysarthric"
for g in ("dysarthric", "control"):
    for kind in ("sentence", "word"):
        sel = [i for i in ids if grp(i) == g and any_[i]["kind"] == kind]
        spk = sorted({any_[i]["speaker"] for i in sel})
        print(f"\n== {g} speakers, {kind}s: clips {len(sel)}")
        for name, d in data.items():
            cs = {i: counts(N(d[i]["ref"]), N(d[i]["hyp"])) for i in sel}
            v = list(cs.values()); n = sum(x["N"] for x in v) or 1
            ex = sum(1 for i in sel if N(d[i]["hyp"]) == N(d[i]["ref"]))
            emp = sum(1 for i in sel if not N(d[i]["hyp"]))
            ps = " ".join(f"{s}:{100*wer([cs[i] for i in sel if any_[i]['speaker']==s]):.0f}" for s in spk)
            print(f"{name:20} WER {100*wer(v):6.1f}%  ins {100*sum(x['I'] for x in v)/n:5.1f}  exact {100*ex/len(sel):4.1f}%  empty {emp:3d}  [{ps}]")
