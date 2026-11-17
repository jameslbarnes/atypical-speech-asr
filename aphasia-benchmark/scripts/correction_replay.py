"""Does a memory of confirmed corrections help an LLM read a speaker's later utterances?

Replays each speaker's utterances in transcript order. Two conditions, same model, same 10 ASR
hypotheses, same 3-utterance context:
  base - the LLM writes what the speaker meant (intended words), no memory
  mem  - same, plus the speaker's confirmed corrections so far

Simulated therapist: after every utterance that contains a coded paraphasia, the therapist confirms
what was meant. The memory entry is what two recognizers heard for that utterance and the intended
(L4) reference. Only EARLIER utterances' references ever enter a prompt; the current one never does.

Usage:
  python correction_replay.py run   <derived_dir> [--model qwen-3.8-27b] [--limit N]
  python correction_replay.py score <derived_dir>
Writes <derived_dir>/hyps_replay/{base,mem}.jsonl (resumable). Prints aggregates only: the data's
terms forbid redistributing transcript text.
"""
import argparse
import collections
import difflib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_correct import api_key, budget_ok, call  # noqa: E402

TIDY, VERBATIM = "whisper-large-v3", "parakeet-ultra-0.6b"
N_CONTEXT, MAX_MEMORY, MIN_INTERVAL = 3, 25, 1.0

SYSTEM = (
    "Several speech-recognition systems transcribed the same short audio clip of a person with aphasia "
    "(a language disorder after stroke) talking with a clinician. Some systems write down every repeated "
    "word and false start; others tidy them away. They also make different recognition errors, and the "
    "speaker sometimes produces a wrong, distorted or made-up word in place of the one they intend.\n\n"
    "Write the single most likely transcript of what the speaker MEANT to say in this clip. Leave out fillers "
    "(um, uh), cut-off fragments, and words the speaker repeated or abandoned. Where the speaker produced a "
    "wrong or distorted word, write the intended word only if the evidence supports it: agreement or "
    "disagreement between systems, sound-alike plausibility, the context, and any confirmed corrections for "
    "this speaker. Otherwise keep the speaker's words. Never add content or complete their sentences. If no "
    "system heard any words, output nothing. Output only the transcript, on one line."
)


def pairs(l3, l4):
    """(produced, target) word groups where the intended reference differs from the clean one."""
    a, b = l3.lower().split(), l4.lower().split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "equal" and j2 > j1:
            out.append((" ".join(a[i1:i2]), " ".join(b[j1:j2])))
    return out


def load(derived):
    recs = [json.loads(l) for l in open(derived / "llm_inputs.jsonl", encoding="utf-8")]
    by_speaker = collections.OrderedDict()
    for r in recs:  # transcript order within each speaker
        by_speaker.setdefault(r["speaker"], []).append(r)
    return by_speaker


def user_prompt(rs, i, memory):
    r = rs[i]
    import random
    models = sorted(r["hyps"])
    random.Random(r["id"]).shuffle(models)
    ctx = [p["hyps"][TIDY]["text"] for p in rs[max(0, i - N_CONTEXT): i]]
    parts = []
    if memory:
        parts.append("Confirmed corrections for this speaker (a therapist checked what the speaker meant):\n"
                     + "\n".join(f'- systems heard: "{m["tidy"]}" / "{m["verbatim"]}" -> speaker meant: "{m["meant"]}"'
                                 for m in memory[-MAX_MEMORY:]))
    parts.append("Earlier in the conversation (automatic transcript, may contain errors):\n"
                 + ("\n".join(f"- {c}" for c in ctx) if ctx else "- (start of recording)"))
    parts.append("Transcripts of the clip:\n"
                 + "\n".join(f"{k}. {r['hyps'][m]['text'] or '(nothing)'}" for k, m in enumerate(models, 1)))
    return "\n\n".join(parts)


def run(args):
    derived = Path(args.derived)
    by_speaker = load(derived)
    out_dir = derived / "hyps_replay"
    out_dir.mkdir(exist_ok=True)
    done = {}
    for cond in ("base", "mem"):
        p = out_dir / f"{cond}.jsonl"
        done[cond] = {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")} if p.exists() else {}
    key = api_key(args.env)
    last, n_calls, n_seen = 0.0, 0, 0
    files = {c: open(out_dir / f"{c}.jsonl", "a", encoding="utf-8") for c in ("base", "mem")}

    def ask(user):
        nonlocal last, n_calls
        time.sleep(max(0.0, MIN_INTERVAL - (time.time() - last)))
        last = time.time()
        data, headers, wall = call(key, {"model": args.model, "temperature": 0, "reasoning_effort": "none",
                                         "max_completion_tokens": 200,
                                         "messages": [{"role": "system", "content": SYSTEM},
                                                      {"role": "user", "content": user}]})
        n_calls += 1
        if not budget_ok(headers):
            print("  key budget below 60%; pausing 15s", flush=True)
            time.sleep(15)
        raw = (data["choices"][0]["message"].get("content") or "").strip()
        usage = data.get("usage", {})
        return {"hyp": raw.splitlines()[0].strip().strip('"') if raw else "", "sec": round(wall, 3),
                "prompt_tokens": usage.get("prompt_tokens")}

    for spk, rs in by_speaker.items():
        memory = []
        for i, r in enumerate(rs):
            if args.limit and n_seen >= args.limit:
                break
            n_seen += 1
            if r["id"] not in done["base"]:
                rec = {"id": r["id"], "n_mem": 0, **ask(user_prompt(rs, i, []))}
                files["base"].write(json.dumps(rec, ensure_ascii=False) + "\n"); files["base"].flush()
                done["base"][r["id"]] = rec
            if r["id"] not in done["mem"]:
                if memory:
                    rec = {"id": r["id"], "n_mem": len(memory), **ask(user_prompt(rs, i, memory))}
                else:  # identical prompt to base: reuse
                    rec = {**done["base"][r["id"]], "n_mem": 0, "reused": True}
                files["mem"].write(json.dumps(rec, ensure_ascii=False) + "\n"); files["mem"].flush()
            if r["errors"]:  # the therapist confirms what was meant, after the system has answered
                memory.append({"tidy": r["hyps"][TIDY]["text"], "verbatim": r["hyps"][VERBATIM]["text"],
                               "meant": r["refs"]["L4"]})
            if n_seen % 100 == 0:
                print(f"  {n_seen} utterances, {n_calls} calls", flush=True)
    print(f"done: {n_seen} utterances, {n_calls} calls", flush=True)


def score(args):
    from score import counts, norm, wer
    derived = Path(args.derived)
    by_speaker = load(derived)
    hyp = {c: {json.loads(l)["id"]: json.loads(l) for l in open(derived / "hyps_replay" / f"{c}.jsonl", encoding="utf-8")}
           for c in ("base", "mem")}
    rows = []
    for spk, rs in by_speaker.items():
        seen_pairs, n_corr = set(), 0
        for r in rs:
            if r["id"] not in hyp["base"] or r["id"] not in hyp["mem"]:
                continue
            ref = norm(r["refs"]["L4"], "L4")
            ps = pairs(norm(r["refs"]["L3"], "L3"), ref) if r["errors"] else []
            row = {"spk": spk, "err": bool(r["errors"]), "n_corr": n_corr, "targets": []}
            for c in ("base", "mem"):
                h = norm(hyp[c][r["id"]]["hyp"], "L4")
                row[c] = counts(ref, h)
                row[c + "_tok"] = set(h.split())
            for prod, tgt in ps:
                row["targets"].append({"repeat": (prod, tgt) in seen_pairs, "words": tgt.split()})
            rows.append(row)
            if r["errors"]:
                seen_pairs.update(ps)
                n_corr += 1

    def w(sel, c):
        cs = [x[c] for x in rows if sel(x)]
        return f"{100 * wer(cs):5.1f}" if cs else "  n/a"

    def line(name, sel):
        n = sum(1 for x in rows if sel(x))
        print(f"{name:<46} n={n:<5} base {w(sel, 'base')}  mem {w(sel, 'mem')}")

    print("Intended-word (L4) WER %, qwen meant-prompt, base vs with correction memory")
    line("all utterances", lambda x: True)
    line("utterances with a coded speaker error", lambda x: x["err"])
    line("  ... with >=1 earlier correction", lambda x: x["err"] and x["n_corr"] >= 1)
    line("utterances without a coded error", lambda x: not x["err"])
    line("  ... with >=1 earlier correction", lambda x: not x["err"] and x["n_corr"] >= 1)
    for lo, hi in ((0, 0), (1, 5), (6, 15), (16, 999)):
        line(f"all, {lo}-{hi} earlier corrections", lambda x, lo=lo, hi=hi: lo <= x["n_corr"] <= hi)
    for spk in by_speaker:
        line(f"speaker {spk}", lambda x, spk=spk: x["spk"] == spk)

    print("\nIntended target words recovered (the word the speaker meant appears in the output)")
    for name, sel in (("all targets", lambda t, x: True),
                      ("targets with >=1 earlier correction", lambda t, x: x["n_corr"] >= 1),
                      ("  same error seen before (repeat)", lambda t, x: t["repeat"]),
                      ("  new error, >=1 earlier correction", lambda t, x: not t["repeat"] and x["n_corr"] >= 1)):
        tot = hit = {"base": 0, "mem": 0}
        tot, hit = 0, {"base": 0, "mem": 0}
        for x in rows:
            for t in x["targets"]:
                if sel(t, x):
                    tot += 1
                    for c in ("base", "mem"):
                        hit[c] += all(wd in x[c + "_tok"] for wd in t["words"])
        print(f"{name:<46} n={tot:<5} base {hit['base']:>3}  mem {hit['mem']:>3}")

    e = lambda c: c["S"] + c["D"] + c["I"]
    for name, sel in (("with coded error", lambda x: x["err"] and x["n_corr"] >= 1),
                      ("without coded error", lambda x: not x["err"] and x["n_corr"] >= 1)):
        sub = [x for x in rows if sel(x)]
        better = sum(e(x["mem"]) < e(x["base"]) for x in sub)
        worse = sum(e(x["mem"]) > e(x["base"]) for x in sub)
        broke = sum(e(x["base"]) == 0 and e(x["mem"]) > 0 for x in sub)
        print(f"\nutterances {name}, memory in use: n={len(sub)}  mem better {better}, worse {worse}, "
              f"broke a correct one {broke}")
    for c in ("base", "mem"):
        secs = [v["sec"] for v in hyp[c].values() if not v.get("reused")]
        toks = [v["prompt_tokens"] for v in hyp[c].values() if not v.get("reused") and v.get("prompt_tokens")]
        if secs:
            print(f"{c}: {len(secs)} calls, median {statistics.median(secs):.3f} s, "
                  f"median prompt {int(statistics.median(toks)) if toks else 0} tokens")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("derived")
    ap.add_argument("--model", default="qwen-3.8-27b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--env", default=".env")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else score(a)
