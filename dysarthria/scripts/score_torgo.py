"""TORGO in-context run, scored with the aphasia benchmark's clean normalizer (Whisper English normalizer).
Usage: <bench python> score_torgo.py <tg_fix.jsonl> <tg_whisper.jsonl> <manifest>"""
import json, random, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "aphasia-benchmark" / "scripts"))
from score import norm as bnorm, counts, wer
N = lambda s: bnorm(s, "L3")
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")] + [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
man = {}
for l in open(sys.argv[3], encoding="utf-8"):
    m = json.loads(l); man[m["id"]] = m
conds = [("whisper",0),("zero",0),("same",6),("same",12),("diff",6),("diff",12),("samev",6),("samev",12)]
by = defaultdict(dict)
for r in rows:
    if not r.get("err"): by[r["id"]][(r["cond"], r["k"])] = r
full = {i: d for i, d in by.items() if len(d) == len(conds)}
print("clips", len(by), "with every condition", len(full))
def grams(t, n=4):
    w = t.split(); return {" ".join(w[j:j+n]) for j in range(len(w)-n+1)}
def copied(r, n=4):
    if not r.get("ex_ids"): return False
    ex = set()
    for e in r["ex_ids"]: ex |= grams(N(man[e]["text"]), n)
    return bool((grams(N(r["hyp"]), n) & ex) - grams(N(r["ref"]), n))
def guard_hit(r, n=3):  # no reference used
    ex = set()
    for e in r.get("ex_ids") or []: ex |= grams(N(man[e]["text"]), n)
    return bool(grams(N(r["hyp"]), n) & ex)
def cs_for(c, ids, guard=False):
    out = {}
    for i in ids:
        r = full[i][c]; hyp = r["hyp"]
        if guard and c[0] not in ("zero","whisper") and guard_hit(r): hyp = full[i][("zero",0)]["hyp"]
        out[i] = counts(N(r["ref"]), N(hyp))
    return out
E = lambda x: x["S"]+x["D"]+x["I"]
def table(ids, label, guard=False):
    print(f"\n== {label}: clips {len(ids)}" + ("  [copy guard on]" if guard else ""))
    res = {}
    spk = sorted({full[i][conds[0]]["speaker"] for i in ids})
    for c in conds:
        cs = cs_for(c, ids, guard); res[c] = cs
        v = list(cs.values()); n = sum(x["N"] for x in v) or 1
        ps = " ".join(f"{s}:{100*wer([cs[i] for i in ids if full[i][c]['speaker']==s]):.0f}" for s in spk)
        ru = sum(1 for i in ids if len(N(full[i][c]['hyp']).split()) > 2*len(N(full[i][c]['ref']).split())+3)
        cp = sum(1 for i in ids if copied(full[i][c]))
        ex = sum(1 for i in ids if N(full[i][c]['hyp']) == N(full[i][c]['ref']))
        print(f"{c[0]:7}{c[1]:3} WER {100*wer(v):6.1f}%  ins {100*sum(x['I'] for x in v)/n:5.1f}  exact {100*ex/len(ids):4.1f}%  runaway {ru:3d} copied {cp:3d}  [{ps}]")
    def diff(a, b):
        idl = sorted(ids); rng = random.Random(3); bs = []
        for _ in range(3000):
            s = [idl[rng.randrange(len(idl))] for _ in idl]
            n = sum(res[a][i]["N"] for i in s) or 1
            bs.append(100*(sum(E(res[a][i]) for i in s)-sum(E(res[b][i]) for i in s))/n)
        bs.sort()
        sp = defaultdict(lambda: [0,0])
        for i in idl:
            sp[full[i][a]["speaker"]][0] += E(res[a][i]); sp[full[i][a]["speaker"]][1] += E(res[b][i])
        print(f"   {a[0]}{a[1]} - {b[0]}{b[1]}: {100*(wer(list(res[a].values()))-wer(list(res[b].values()))):+.1f} pts (95% clip bootstrap {bs[75]:+.1f} to {bs[2925]:+.1f}); better in {sum(1 for v in sp.values() if v[0]<v[1])}/{len(sp)} speakers")
    for a, b in ((("same",6),("zero",0)),(("same",12),("zero",0)),(("same",6),("diff",6)),(("same",12),("diff",12)),(("samev",6),("same",6)),(("same",12),("whisper",0)),(("zero",0),("whisper",0))):
        diff(a, b)
allids = list(full)
kind = lambda i: full[i][("zero",0)].get("kind") or man[i]["kind"]
S = [i for i in allids if kind(i) == "sentence"]; W = [i for i in allids if kind(i) == "word"]
table(allids, "all clips"); table(S, "sentences"); table(W, "single words and two-word items")
table(allids, "all clips", guard=True); table(S, "sentences", guard=True); table(W, "words", guard=True)
mild = {"F03","F04","M03"}  # labelled by this run's own no-example error, see per-speaker columns
no_run = [i for i in allids if all(len(N(full[i][c]['hyp']).split()) <= 2*len(N(full[i][c]['ref']).split())+3 for c in conds)]
table(no_run, "clips where no condition ran away (after-the-fact selection)")
for c in conds:
    v = [full[i][c] for i in allids if c[0] != "whisper"]
    if v: print(f"{c[0]:7}{c[1]:3} mean example audio {sum(r['ex_audio_s'] for r in v)/len(v):5.1f} s  secs per request {sum(r['secs'] for r in v)/len(v):.2f}")
