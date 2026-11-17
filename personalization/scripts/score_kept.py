import json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "aphasia-benchmark" / "scripts"))
from score import norm as bnorm
N = lambda s: bnorm(s, "L3").split()
rows = []
for f in sys.argv[1:]: rows += [json.loads(l) for l in open(f, encoding="utf-8")]
agg = defaultdict(lambda: [0, 0, 0, 0, set()])
for r in rows:
    a, b = N(r["refs"]["L3"]), N(r["refs"]["L4"])
    if a == b: continue
    said, meant, h = set(a) - set(b), set(b) - set(a), set(N(r["hyp"]))
    g = agg[(r["arm"], r["n"])]
    g[0] += len(said & h); g[1] += len(said); g[2] += len(meant & h); g[3] += len(meant); g[4].add((r["id"], r["seed"]))
for k in sorted(agg, key=lambda k: (["base","loop","noise20","self","alt"].index(k[0]), k[1])):
    g = agg[k]
    print(f"{k[0]:8}{k[1]:3} clip-runs {len(g[4]):3d}  error words written as said {g[0]}/{g[1]} ({100*g[0]/max(1,g[1]):.0f}%)  replaced by the intended word {g[2]}/{g[3]} ({100*g[2]/max(1,g[3]):.0f}%)")
