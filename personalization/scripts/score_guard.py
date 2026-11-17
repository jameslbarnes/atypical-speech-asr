"""Deployable guard: if the output shares a 4-word run with any example transcript, use the no-example output instead.
(No reference is used to decide.) Numbers only."""
import json, sys, random
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "aphasia-benchmark" / "scripts"))
from score import norm, counts, wer
rows=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8")]
man={}
for l in open(sys.argv[2],encoding="utf-8"):
    m=json.loads(l); man[m["id"]]=m
by=defaultdict(dict)
for r in rows: by[r["id"]][(r["cond"],r["k"])]=r
def grams(t,n):
    w=t.split(); return {" ".join(w[j:j+n]) for j in range(len(w)-n+1)}
for n in (4,3):
  for dmin in (0.0,1.5):
    print(f"\nguard: {n}-word run shared with an example; examples used only for clips >= {dmin} s")
    for lv in ("L3","L0"):
        for c in (("zero",0),("same",6),("same",12),("diff",6),("diff",12)):
            cs={}; fb=0
            for i,d in by.items():
                if len(d)<5: continue
                r=d[c]; z=d[("zero",0)]; hyp=r["hyp"]
                if c[0]!="zero":
                    ex=set()
                    for e in r["ex_ids"]: ex|=grams(norm(man[e]["refs"]["L0"],"L0"),n)
                    if (grams(norm(hyp,"L0"),n)&ex) or man[i]["dur"]<dmin: hyp=z["hyp"]; fb+=1
                cs[i]=(counts(norm(r["refs"][lv],lv),norm(hyp,lv)), r["speaker"])
            v=[x[0] for x in cs.values()]
            sp=defaultdict(list)
            for x,s in cs.values(): sp[s].append(x)
            print(f"  {lv} {c[0]:5}{c[1]:3} WER {100*wer(v):5.1f}%  fell back on {fb:4d} clips  [" + " ".join(f"{s}:{100*wer(x):.1f}" for s,x in sorted(sp.items())) + "]")
