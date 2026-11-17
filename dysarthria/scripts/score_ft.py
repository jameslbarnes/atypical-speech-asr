import json, random, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "aphasia-benchmark" / "scripts"))
from score import norm as bnorm, counts, wer
N = lambda s: bnorm(s, "L3")
rows = []
for f in sys.argv[1:]:
    rows += [json.loads(l) for l in open(f, encoding="utf-8")]
have = {r["speaker"] for r in rows if "c2" in ""}
# drop M03 rows from the crashed process (ft_all) in favour of the rerun
seen = {}
for r in rows:
    seen[(r["id"], r["cond"], r["k"], r["lr"])] = r   # later files overwrite earlier
rows = list(seen.values())
conds = [("base",0,0.0),("own",10,1e-4),("own",10,1e-3),("own",30,1e-4),("own",30,1e-3),("diff",10,1e-4),("diff",10,1e-3)]
by = defaultdict(dict)
for r in rows: by[r["id"]][(r["cond"], r["k"], r["lr"])] = r
E = lambda x: x["S"]+x["D"]+x["I"]
spk = sorted({r["speaker"] for r in rows})
def table(filt, label, need):
    ids = [i for i, d in by.items() if all(c in d for c in need) and filt(d[conds[0]])]
    print(f"\n== {label}: clips {len(ids)}, speakers {sorted({by[i][conds[0]]['speaker'] for i in ids})}")
    res = {}
    for c in need:
        cs = {i: counts(N(by[i][c]["ref"]), N(by[i][c]["hyp"])) for i in ids}; res[c] = cs
        v = list(cs.values()); n = sum(x["N"] for x in v) or 1
        ps = " ".join(f"{s}:{100*wer([cs[i] for i in ids if by[i][c]['speaker']==s]):.0f}" for s in spk if any(by[i][c]['speaker']==s for i in ids))
        ex = sum(1 for i in ids if N(by[i][c]['hyp']) == N(by[i][c]['ref']))
        ru = sum(1 for i in ids if len(N(by[i][c]['hyp']).split()) > 2*len(N(by[i][c]['ref']).split())+3)
        print(f"{c[0]:5}{c[1]:3} lr {c[2]:<7g} WER {100*wer(v):6.1f}%  ins {100*sum(x['I'] for x in v)/n:5.1f}  exact {100*ex/len(ids):4.1f}%  runaway {ru:3d}  [{ps}]")
    def diff(a, b):
        idl = sorted(ids); rng = random.Random(5); bs = []
        for _ in range(3000):
            s = [idl[rng.randrange(len(idl))] for _ in idl]
            n = sum(res[a][i]["N"] for i in s) or 1
            bs.append(100*(sum(E(res[a][i]) for i in s)-sum(E(res[b][i]) for i in s))/n)
        bs.sort(); sp = defaultdict(lambda: [0,0])
        for i in idl:
            sp[by[i][a]["speaker"]][0] += E(res[a][i]); sp[by[i][a]["speaker"]][1] += E(res[b][i])
        print(f"   {a[0]}{a[1]}@{a[2]:g} - {b[0]}{b[1]}@{b[2]:g}: {100*(wer(list(res[a].values()))-wer(list(res[b].values()))):+.1f} pts (95% {bs[75]:+.1f} to {bs[2925]:+.1f}); better in {sum(1 for v in sp.values() if v[0]<v[1])}/{len(sp)} speakers")
    for lr in (1e-4, 1e-3):
        diff(("own",10,lr),("base",0,0.0)); diff(("diff",10,lr),("base",0,0.0)); diff(("own",10,lr),("diff",10,lr))
        if ("own",30,lr) in need: diff(("own",30,lr),("base",0,0.0)); diff(("own",30,lr),("own",10,lr))
k10 = [c for c in conds if c[1] != 30]
for nm, f in (("all clips", lambda r: True), ("sentences", lambda r: r["kind"]=="sentence"), ("words", lambda r: r["kind"]=="word")):
    table(f, nm + " (8 speakers, k=10 conditions)", k10)
    table(f, nm + " (7 speakers with 30 sentences)", conds)
