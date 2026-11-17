"""Merge references and every model's hypothesis into one record per utterance.

This is the input for an LLM correction layer experiment: N hypotheses from different
systems, per-token confidences where the runtime exposes them, and the layered
references (L0 verbatim ... L4 intended) to score against.

Usage: python export_llm_inputs.py <derived_dir>
  writes <derived_dir>/llm_inputs.jsonl (contains transcripts; keep local)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_asr import usable  # noqa: E402


def main(derived):
    derived = Path(derived)
    utts = [json.loads(l) for l in open(derived / "utterances.jsonl", encoding="utf-8")]
    rows = {r["id"]: r for r in usable(utts)}
    hyps = {p.stem: {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")}
            for p in sorted((derived / "hyps").glob("*.jsonl"))}
    complete = [m for m in hyps if set(rows) <= set(hyps[m])]
    with open(derived / "llm_inputs.jsonl", "w", encoding="utf-8") as f:
        for uid, r in rows.items():
            rec = {k: r[k] for k in ("id", "speaker", "qab", "impairment", "start", "end", "chat", "refs", "errors")}
            rec["hyps"] = {}
            for m in complete:
                h = hyps[m][uid]
                conf = [t["p"] for t in h.get("tokens", []) if t.get("t", "").strip()] or [w["p"] for w in h.get("words", [])]
                rec["hyps"][m] = {"text": h["hyp"], "min_p": min(conf) if conf else None,
                                  "tokens": h.get("tokens") or h.get("words") or None}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} utterances x {len(complete)} models -> {derived / 'llm_inputs.jsonl'}")


if __name__ == "__main__":
    main(sys.argv[1])
