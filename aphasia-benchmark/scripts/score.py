"""Score ASR hypotheses against the layered APROCSA references.

Usage: python score.py <derived_dir> <results_dir> [utterances_file] [hyps_subdir] [extra_subdir]
  reads  <derived_dir>/<utterances_file> (default utterances.jsonl) and <derived_dir>/<hyps_subdir>/*.jsonl (default hyps)
  extra_subdir (e.g. hyps_llm): systems built on top of the ASR outputs; scored in every table but left out of
  the oracle and consensus figures, which describe the ASR models alone
  writes <results_dir>/summary.md, wer_by_level.csv, wer_by_speaker.csv
         <derived_dir>/scored_utterances.jsonl   (per-utterance detail; contains transcripts, keep local)

All models are scored on the same utterances (the intersection of what every model finished).
L0 keeps fillers (um/uh unified); L1-L4 use the Whisper English normalizer, which drops them.
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np
from whisper_normalizer.english import EnglishTextNormalizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_asr import usable  # noqa: E402  (same utterance filter as the runs)

LEVELS = ["L0", "L1", "L2", "L3", "L4"]
LEVEL_NAMES = {
    "L0": "verbatim (fillers, fragments, repetitions)",
    "L1": "no fillers",
    "L2": "no fillers/fragments",
    "L3": "clean (no retraced/repeated words)",
    "L4": "intended (paraphasias -> targets)",
}
SPEAKER_ORDER = ["1713", "1554", "1738", "1944", "1833", "1731"]  # mildest aphasia first

COLLOQUIAL = [
    (r"\bgonna\b", "going to"), (r"\bwanna\b", "want to"), (r"\bgotta\b", "got to"),
    (r"\bkinda\b", "kind of"), (r"\bsorta\b", "sort of"), (r"\b(?:'cause|cuz|cos)\b", "because"),
    (r"\bo\.?k\.?\b", "okay"), (r"\b(?:yep|yup)\b", "yes"), (r"\balright\b", "all right"),
    (r"\bhm+\b", "hmm"),  # CHAT writes "hm"; the Whisper normalizer only knows "hmm" as a filler
]
FILLERS = r"\b(?:u+h+m*|u+m+|e+r+m*|a+h+|e+h+|h+m+|m+)\b"
BACKCHANNEL = r"\b(?:mm[- ]?hmm|mhm|mmhmm|m-hm)\b"

_clean = EnglishTextNormalizer()
_verb = EnglishTextNormalizer()
_verb.ignore_patterns = r"(?!x)x"  # keep fillers


def pre(s):
    s = s.lower()
    s = re.sub(BACKCHANNEL, " mhm ", s)
    s = re.sub(r"\buh[- ]huh\b", " uhhuh ", s)
    for pat, rep in COLLOQUIAL:
        s = re.sub(pat, rep, s)
    return s


def norm(s, level):
    s = pre(s)
    if level == "L0":
        s = _verb(s)
        s = re.sub(FILLERS, "uh", s)
    else:
        s = _clean(s)
    return " ".join(t for t in s.split() if any(ch.isalnum() for ch in t))  # drop leftover "." from ellipses


def counts(ref, hyp):
    if not ref:
        return {"S": 0, "D": 0, "I": len(hyp.split()), "N": 0}
    if not hyp:
        return {"S": 0, "D": len(ref.split()), "I": 0, "N": len(ref.split())}
    o = jiwer.process_words(ref, hyp)
    return {"S": o.substitutions, "D": o.deletions, "I": o.insertions, "N": o.hits + o.substitutions + o.deletions}


def wer(cs):
    n = sum(c["N"] for c in cs)
    return (sum(c["S"] + c["D"] + c["I"] for c in cs) / n) if n else float("nan")


def boot_ci(cs, iters=1000, seed=0):
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        sample = [cs[rng.randrange(len(cs))] for _ in cs]
        vals.append(wer(sample))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def auroc(scores, labels):
    """P(score of a positive > score of a negative); positives = utterances with errors."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    ranks = np.argsort(np.argsort(np.array(pos + neg)))
    r_pos = ranks[: len(pos)].sum() + len(pos)  # 1-based ranks
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main(derived, results, utts_file="utterances.jsonl", hyps_dir="hyps", extra_dir=None):
    derived, results = Path(derived), Path(results)
    results.mkdir(parents=True, exist_ok=True)
    utts = {json.loads(l)["id"]: json.loads(l) for l in open(derived / utts_file, encoding="utf-8")}
    hyps = {}
    for p in sorted((derived / hyps_dir).glob("*.jsonl")):
        hyps[p.stem] = {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")}
    extra_names = set()
    if extra_dir:
        for p in sorted((derived / extra_dir).glob("*.jsonl")):
            hyps[p.stem] = {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")}
            extra_names.add(p.stem)
    target = {r["id"] for r in usable(list(utts.values()))}
    complete = [m for m in hyps if target <= set(hyps[m])]
    partial = {m: len(target & set(hyps[m])) for m in hyps if m not in complete}
    if complete:
        models, common = complete, sorted(target)
    else:  # interim look while runs are still going
        models = [m for m in hyps if hyps[m]]
        common = sorted(set.intersection(*(set(hyps[m]) for m in models)))
    asr_models = [m for m in models if m not in extra_names]
    print(f"models: {len(models)}  common utterances: {len(common)}  partial: {partial}")

    refs = {(uid, lv): norm(utts[uid]["refs"][lv], lv) for uid in common for lv in LEVELS}
    per = defaultdict(dict)  # per[model][(uid, level)] = counts
    detail = []
    for m in models:
        for uid in common:
            h = hyps[m][uid]
            row = {"id": uid, "model": m, "hyp": h["hyp"]}
            for lv in LEVELS:
                hn = norm(h["hyp"], lv)
                c = counts(refs[(uid, lv)], hn)
                per[m][(uid, lv)] = c
                row[lv] = c
            detail.append(row)

    with open(derived / "scored_utterances.jsonl", "w", encoding="utf-8") as f:
        for row in detail:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- corpus WER by level
    table = {}
    for m in models:
        table[m] = {lv: wer([per[m][(u, lv)] for u in common]) for lv in LEVELS}
    order = sorted(models, key=lambda m: table[m]["L3"])

    audio_min = sum(utts[u]["dur"] for u in common) / 60
    n_words = {lv: sum(len(refs[(u, lv)].split()) for u in common) for lv in LEVELS}

    lines = ["# ASR on aphasic speech: APROCSA open subset (6 speakers)", ""]
    lines.append(f"{len(common)} participant utterances, {audio_min:.1f} min of audio, scored identically for every model. "
                 f"Reference words: " + ", ".join(f"{lv} {n_words[lv]}" for lv in LEVELS) + ".")
    lines.append("")
    lines.append("## WER by reference level (sorted by L3 clean)")
    lines.append("")
    lines.append("| model | " + " | ".join(f"{lv}" for lv in LEVELS) + " | L3 95% CI | RTF (CPU) |")
    lines.append("|---|" + "---|" * (len(LEVELS) + 2))
    for m in order:
        cs = [per[m][(u, "L3")] for u in common]
        lo, hi = boot_ci(cs)
        if m in extra_names:
            lat = sorted(hyps[m][u]["sec"] for u in common)
            speed = f"API {1000*lat[len(lat)//2]:.0f} ms median"
        else:
            speed = f"{sum(hyps[m][u]['sec'] for u in common) / sum(hyps[m][u]['audio_sec'] for u in common):.2f}"
        lines.append(f"| {m} | " + " | ".join(f"{100*table[m][lv]:.1f}" for lv in LEVELS)
                     + f" | {100*lo:.1f}-{100*hi:.1f} | {speed} |")
    lines.append("")
    lines.append("Levels: " + "; ".join(f"{lv} = {LEVEL_NAMES[lv]}" for lv in LEVELS) + ". "
                 "CI = utterance bootstrap (ignores speaker clustering). RTF was measured on a shared CPU "
                 "at idle priority next to a live workload, so it is only a rough relative guide.")
    if partial:
        lines.append("")
        lines.append("Not yet complete (excluded): " + ", ".join(f"{m} {n}/{len(target)}" for m, n in partial.items()) + ".")

    # ---- error type breakdown at L3
    lines += ["", "## Error types", "",
              "Substitutions/deletions/insertions are shares of L3 reference words. Insertions against L3 include "
              "repetitions a verbatim model faithfully writes down, so invented text is measured against L0 instead: "
              "'extra words vs L0' is the insertion rate against the verbatim reference, and 'invented runs' counts "
              "utterances with 5+ words that are not in even the verbatim reference (a hallucination proxy).", "",
              "| model | subs (L3) | dels (L3) | ins (L3) | extra words vs L0 | invented runs (5+ words) | empty outputs |",
              "|---|---|---|---|---|---|---|"]
    for m in order:
        cs = [per[m][(u, "L3")] for u in common]
        c0 = [per[m][(u, "L0")] for u in common]
        n, n0 = sum(c["N"] for c in cs), sum(c["N"] for c in c0)
        empty = sum(1 for u in common if not hyps[m][u]["hyp"].strip() and refs[(u, "L2")])
        invented = sum(1 for c in c0 if c["I"] >= 5)
        lines.append(f"| {m} | {100*sum(c['S'] for c in cs)/n:.1f} | {100*sum(c['D'] for c in cs)/n:.1f} | "
                     f"{100*sum(c['I'] for c in cs)/n:.1f} | {100*sum(c['I'] for c in c0)/n0:.1f} | {invented} | {empty} |")

    # ---- by speaker
    lines += ["", "## L3 WER by speaker (ordered from mildest to most severe aphasia)", ""]
    hdr = list(SPEAKER_ORDER)
    lines.append("| model | " + " | ".join(hdr) + " |")
    lines.append("|---|" + "---|" * len(SPEAKER_ORDER))
    spk_rows = []
    for m in order:
        vals = []
        for s in SPEAKER_ORDER:
            v = wer([per[m][(u, "L3")] for u in common if utts[u]["speaker"] == s])
            vals.append(v)
            spk_rows.append((m, s, v))
        lines.append(f"| {m} | " + " | ".join(f"{100*v:.1f}" for v in vals) + " |")

    # ---- by utterance length (short answers are the closest proxy here for single-word naming responses)
    buckets = [("1-2 words", 1, 2), ("3-6 words", 3, 6), ("7+ words", 7, 10**6)]
    lines += ["", "## L3 WER by utterance length (reference words)", ""]
    bucket_ids = {name: [u for u in common if lo <= len(refs[(u, "L3")].split()) <= hi] for name, lo, hi in buckets}
    lines.append("| model | " + " | ".join(f"{name} (n={len(bucket_ids[name])})" for name, _, _ in buckets) + " |")
    lines.append("|---|" + "---|" * len(buckets))
    for m in order:
        lines.append(f"| {m} | " + " | ".join(f"{100*wer([per[m][(u, 'L3')] for u in bucket_ids[name]]):.1f}"
                                            for name, _, _ in buckets) + " |")

    # ---- complementarity: oracle best-of-N per utterance (upper bound for hypothesis selection)
    lines += ["", "## Headroom for an LLM layer", ""]
    best_asr = min(asr_models, key=lambda m: table[m]["L3"])
    oracle = []
    for u in common:
        oracle.append(min((per[m][(u, "L3")] for m in asr_models), key=lambda c: c["S"] + c["D"] + c["I"]))
    lines.append(f"- Best single ASR model at L3: {best_asr} {100*table[best_asr]['L3']:.1f}%. "
                 f"Oracle that picks the best of the {len(asr_models)} ASR hypotheses per utterance: {100*wer(oracle):.1f}%. "
                 "The gap is the most a hypothesis-selection layer could recover (a generative layer could in principle go further, or invent).")
    # label-free consensus: per utterance, the hypothesis closest on average to all the others (medoid / MBR)
    for lv in ("L3", "L0"):
        picked = []
        for u in common:
            hs = {m: norm(hyps[m][u]["hyp"], lv) for m in asr_models}

            def dist(a, b):
                c = counts(a, b)
                return (c["S"] + c["D"] + c["I"]) / max(c["N"], 1)
            best_m = min(asr_models, key=lambda m: sum(dist(hs[o], hs[m]) for o in asr_models if o != m))
            picked.append(per[best_m][(u, lv)])
        lines.append(f"- Label-free consensus pick (the hypothesis that agrees most with the others) at {lv}: {100*wer(picked):.1f}% "
                     f"vs best single ASR model at {lv}: {100*min(table[m][lv] for m in asr_models):.1f}%.")
    para = [u for u in common if utts[u]["n_paraphasia"] > 0]
    nopara = [u for u in common if utts[u]["n_paraphasia"] == 0]
    lines.append(f"- Utterances with coded paraphasias: {len(para)}. L3 WER with vs without paraphasias ({best_asr}): "
                 f"{100*wer([per[best_asr][(u,'L3')] for u in para]):.1f}% vs {100*wer([per[best_asr][(u,'L3')] for u in nopara]):.1f}%.")
    lines.append("- L3 minus L4 (positive = model output is closer to the *intended* words than to what was said):")
    for m in order:
        lines.append(f"  - {m}: {100*(table[m]['L3']-table[m]['L4']):+.2f} pts")

    # ---- do confidences flag errors? (utterance level)
    conf_lines = []
    for m in order:
        scores, labels = [], []
        for u in common:
            h = hyps[m][u]
            ps = [t["p"] for t in h.get("tokens", []) if t.get("t", "").strip()] or [w["p"] for w in h.get("words", [])]
            if not ps:
                continue
            c = per[m][(u, "L3")]
            scores.append(-min(ps))
            labels.append((c["S"] + c["D"] + c["I"]) > 0)
        if len(scores) > 50:
            conf_lines.append(f"  - {m}: AUROC {auroc(scores, labels):.2f} (n={len(scores)})")
    if conf_lines:
        lines.append("- Does the lowest token confidence flag utterances with errors? (AUROC, 0.5 = chance)")
        lines += conf_lines

    (results / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(results / "wer_by_level.csv", "w", encoding="utf-8") as f:
        f.write("model," + ",".join(LEVELS) + "\n")
        for m in order:
            f.write(m + "," + ",".join(f"{table[m][lv]:.4f}" for lv in LEVELS) + "\n")
    with open(results / "wer_by_speaker.csv", "w", encoding="utf-8") as f:
        f.write("model,speaker,wer_L3\n")
        for m, s, v in spk_rows:
            f.write(f"{m},{s},{v:.4f}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:6])
