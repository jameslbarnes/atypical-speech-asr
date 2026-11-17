"""Pooled word error rate per condition, with a speaker-level bootstrap interval.
Usage: python score.py <results.jsonl> [ref_field]
ref_field: 'ref' (default) or a key inside each row's 'refs' dict (e.g. L0, L3, L4).
"""
import json
import random
import re
import sys
from collections import defaultdict


def norm(t):
    t = (t or "").lower()
    t = re.sub(r"[.,?!;:\"'()\[\]]", " ", t)
    return t.split()


def edits(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)]


rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
field = sys.argv[2] if len(sys.argv) > 2 else "ref"
# only clips that have every condition, so conditions are compared on the same clips
conds = sorted({(r["cond"], r["k"]) for r in rows})
have = defaultdict(set)
for r in rows:
    if not r.get("err"):
        have[r["id"]].add((r["cond"], r["k"]))
full = {i for i, s in have.items() if len(s) == len(conds)}
stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # cond -> speaker -> [errors, words]
for r in rows:
    if r["id"] not in full:
        continue
    ref = r["ref"] if field == "ref" else r.get("refs", {}).get(field, "")
    rw, hw = norm(ref), norm(r["hyp"])
    if not rw:
        continue
    s = stats[(r["cond"], r["k"])][r["speaker"]]
    s[0] += edits(rw, hw)
    s[1] += len(rw)
print(f"clips with every condition: {len(full)} of {len(have)}; reference: {field}; errors: {sum(1 for r in rows if r.get('err'))}")
rng = random.Random(0)
for c in conds:
    sp = stats[c]
    if not sp:
        continue
    e = sum(v[0] for v in sp.values())
    w = sum(v[1] for v in sp.values())
    keys = list(sp)
    boots = []
    for _ in range(2000):
        pick = [sp[rng.choice(keys)] for _ in keys]
        boots.append(sum(v[0] for v in pick) / max(1, sum(v[1] for v in pick)))
    boots.sort()
    per = ", ".join(f"{k}:{100*v[0]/max(1,v[1]):.1f}" for k, v in sorted(sp.items())) if len(sp) <= 12 else f"{len(sp)} speakers"
    print(f"{c[0]:5} k={c[1]:2}  WER {100*e/w:5.1f}%  (speaker bootstrap {100*boots[50]:.1f}-{100*boots[1949]:.1f})  words {w}  [{per}]")
