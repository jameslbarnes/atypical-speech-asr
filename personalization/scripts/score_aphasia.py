"""Aphasia in-context run scored with the benchmark's own normalizer. Numbers only, no transcripts.
Usage: <bench venv python> score_aphasia.py <aph_L0.jsonl> <aph_L3.jsonl> <manifest_L0.jsonl>
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "aphasia-benchmark" / "scripts"))
from score import norm, counts, wer  # benchmark normalizer and counts

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
l3 = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
for r in l3: r["cond"] = "sameL3"
rows += l3
man = {}
for l in open(sys.argv[3], encoding="utf-8"):
    m = json.loads(l); man[m["id"]] = m
conds = [("zero",0),("same",6),("same",12),("diff",6),("diff",12),("sameL3",6)]
by = defaultdict(dict)
for r in rows: by[r["id"]][(r["cond"], r["k"])] = r
full = {i:d for i,d in by.items() if len(d)==len(conds)}
print("clips", len(full))
nwords = {i: len(norm(d[conds[0]]["refs"]["L3"], "L3").split()) for i,d in full.items()}

def grams(t, n=4):
    w = t.split(); return {" ".join(w[j:j+n]) for j in range(len(w)-n+1)}

def table(level, ids, label):
    print(f"\n== {level} {label}: clips {len(ids)}")
    out = {}
    for c in conds:
        cs = {}
        for i in ids:
            r = full[i][c]
            cs[i] = counts(norm(r["refs"][level], level), norm(r["hyp"], level))
        v = list(cs.values()); N = sum(x["N"] for x in v)
        out[c] = cs
        print(f"{c[0]:7}{c[1]:3}  WER {100*wer(v):5.1f}%  sub {100*sum(x['S'] for x in v)/N:4.1f} del {100*sum(x['D'] for x in v)/N:4.1f} ins {100*sum(x['I'] for x in v)/N:5.1f}  words {N}")
    def diff(a,b):
        ids2 = sorted(ids); rng = random.Random(2); bs=[]
        for _ in range(3000):
            s=[ids2[rng.randrange(len(ids2))] for _ in ids2]
            n=sum(out[a][i]["N"] for i in s) or 1
            e=lambda cc: sum(out[cc][i]["S"]+out[cc][i]["D"]+out[cc][i]["I"] for i in s)
            bs.append(100*(e(a)-e(b))/n)
        bs.sort()
        sp=defaultdict(lambda:[0,0])
        for i in ids2:
            s_=full[i][a]["speaker"]
            for k,cc in enumerate((a,b)): sp[s_][k]+=out[cc][i]["S"]+out[cc][i]["D"]+out[cc][i]["I"]
        print(f"   {a[0]}{a[1]} - {b[0]}{b[1]}: {100*(wer(list(out[a].values()))-wer(list(out[b].values()))):+.1f} pts (95% {bs[75]:+.1f} to {bs[2925]:+.1f}); better in {sum(1 for v in sp.values() if v[0]<v[1])}/{len(sp)} speakers")
    diff(("same",6),("zero",0)); diff(("same",12),("zero",0)); diff(("same",6),("diff",6)); diff(("sameL3",6),("same",6))
    return out

allids = list(full)
for lv in ("L0","L3","L4"):
    table(lv, allids, "all clips")

# runaway outputs and copying from the examples
print("\n== runaway (hyp > 2x ref + 3 words, L0) and copied-from-example (shares a 4-word run with an example transcript that is not in the reference)")
run_any=set()
for c in conds:
    ru=cp=0
    for i,d in full.items():
        r=d[c]; h=norm(r["hyp"],"L0"); rf=norm(r["refs"]["L0"],"L0")
        isrun = len(h.split())>2*len(rf.split())+3
        ru+=isrun
        if isrun: run_any.add(i)
        ex=set()
        for e in r.get("ex_ids") or []:
            if e in man: ex|=grams(norm(man[e]["refs"]["L0"],"L0"))|grams(norm(man[e]["refs"]["L3"],"L0"))
        iscp = bool((grams(h)&ex)-grams(rf))
        cp+=iscp
        if iscp: run_any.add(i)
    print(f"{c[0]:7}{c[1]:3}  runaway {ru}  copied {cp}")
clean=[i for i in allids if i not in run_any]
print("clips with no runaway or copy in any condition:", len(clean))
for lv in ("L0","L3","L4"):
    table(lv, clean, "clips with no runaway/copy in any condition")
for lo,hi in ((0,2),(3,7),(8,999)):
    ids=[i for i in allids if lo<=nwords[i]<=hi]
    table("L3", ids, f"clean length {lo}-{hi} words")
    table("L3", [i for i in ids if i not in run_any], f"clean length {lo}-{hi} words, no runaway/copy")
# example audio and timing
for c in conds:
    v=[full[i][c] for i in allids]
    print(f"{c[0]:7}{c[1]:3} mean example audio {sum(r['ex_audio_s'] for r in v)/len(v):5.1f} s  mean secs per clip {sum(r['secs'] for r in v)/len(v):.2f}")
