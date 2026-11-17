"""How the single-transcript LLM rewrites changed their source ASR output.

Usage: python rewrite_report.py <derived_dir>
For each variant: share of utterances changed, and of those how many got better, worse or stayed
equal against the target reference; how many already-correct utterances were broken.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_rewrite import VARIANTS  # noqa: E402
from run_asr import usable  # noqa: E402
from score import counts, norm, wer  # noqa: E402

TARGET = {"said": "L3", "meant": "L4", "said-conf": "L3", "said-ultra": "L3", "meant-reason": "L4", "said-alts": "L3"}


def main(derived):
    derived = Path(derived)
    rows = {r["id"]: r for r in usable([json.loads(l) for l in open(derived / "utterances.jsonl", encoding="utf-8")])}
    for variant, (source, _, _) in VARIANTS.items():
        path = derived / "hyps_rewrite" / f"rewrite-{variant}.jsonl"
        if not path.exists():
            continue
        out = {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8")}
        src_path = derived / f"{source}.jsonl" if "/" in source else derived / "hyps" / f"{source}.jsonl"
        src = {json.loads(l)["id"]: json.loads(l) for l in open(src_path, encoding="utf-8")}
        lv = TARGET[variant]
        ids = [u for u in rows if u in out]
        before, after = [], []
        changed = better = worse = broke = 0
        for u in ids:
            ref = norm(rows[u]["refs"][lv], lv)
            a, b = norm(src[u]["hyp"], lv), norm(out[u]["hyp"], lv)
            ca, cb = counts(ref, a), counts(ref, b)
            before.append(ca)
            after.append(cb)
            if a != b:
                changed += 1
                ea, eb = ca["S"] + ca["D"] + ca["I"], cb["S"] + cb["D"] + cb["I"]
                better += eb < ea
                worse += eb > ea
                broke += ea == 0 and eb > 0
        print(f"{variant:10s} source {source:20s} n={len(ids)}  {lv} WER {100*wer(before):.1f} -> {100*wer(after):.1f}  "
              f"changed {changed} ({100*changed/len(ids):.0f}%): better {better}, worse {worse}, "
              f"same {changed-better-worse}; broke {broke} correct ones")


if __name__ == "__main__":
    main(sys.argv[1])
