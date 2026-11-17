import json, sys, glob, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "aphasia-benchmark" / "scripts"))
from score import norm as bnorm, counts, wer
N = lambda s: bnorm(s, "L3")
base_live = {json.loads(l)["id"]: json.loads(l) for l in open("results/cmp/cmp_r2t2_stream.jsonl", encoding="utf-8")}
tot = {}
print(f"{'speaker':8}{'kind':9}{'clips':>6}  untuned-live  tuned-live  untuned-whole  tuned-whole")
for f in sorted(glob.glob("results/live_audio/stream_*.jsonl")):
    spk = os.path.basename(f)[7:-6]
    tl = {json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8")}
    wh = [json.loads(l) for l in open(f"results/live_audio/whole_{spk}.jsonl", encoding="utf-8")]
    wb = {r["id"]: r for r in wh if r["arm"] == "base"}; wt = {r["id"]: r for r in wh if r["arm"] == "loop"}
    for kind in ("sentence", "word"):
        ids = [i for i in tl if i in wb and i in wt and i in base_live and tl[i]["kind"] == kind and not tl[i].get("err")]
        if not ids: continue
        row = []
        for name, d in (("ul", base_live), ("tl", tl), ("uw", wb), ("tw", wt)):
            cs = [counts(N(d[i]["ref"]), N(d[i]["hyp"])) for i in ids]
            row.append(100 * wer(cs)); tot.setdefault((kind, name), []).extend(cs)
        print(f"{spk:8}{kind:9}{len(ids):6d}  {row[0]:11.1f}  {row[1]:10.1f}  {row[2]:13.1f}  {row[3]:11.1f}")
for kind in ("sentence", "word"):
    print(f"{'ALL':8}{kind:9}{len(tot[(kind,'ul')]):6d}  " + "  ".join(f"{100*wer(tot[(kind,n)]):11.1f}" for n in ("ul", "tl", "uw", "tw")))
