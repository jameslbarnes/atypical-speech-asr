"""Aphasia-aware LLM rewrite of a single ASR transcript (Cerebras).

The LLM sees one ASR system's transcript of the utterance (optionally with the words that system was
unsure about marked), the same system's transcript of the speaker's previous 3 utterances, and a
generic description of how aphasic speech and ASR on it typically go wrong. It never sees the
reference transcripts or other systems' output. It decides where the transcript is probably wrong
and rewrites it.

Usage: python llm_rewrite.py <derived_dir> <variant> [--limit N] [--model qwen-3.8-27b] [--env PATH]
  variants:  said       whisper-large-v3 -> clean transcript of what the speaker actually said (score at L3)
             meant      whisper-large-v3 -> what the speaker was trying to say (score at L4)
             said-conf  parakeet-ultra-0.6b with low-confidence words marked -> what was said (score at L3)
             said-ultra parakeet-ultra-0.6b without marks (control for said-conf)
             meant-reason  as meant, with reasoning_effort low
             said-alts  Parakeet Ultra (ONNX) with the decoder's alternative tokens for unsure words
  reads  <derived_dir>/utterances.jsonl and <derived_dir>/hyps/<source>.jsonl
  writes <derived_dir>/hyps_rewrite/rewrite-<variant>.jsonl (resumable)
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_correct import MIN_INTERVAL, api_key, budget_ok, call  # noqa: E402
from run_asr import usable  # noqa: E402

CONF_THRESHOLD = 0.9  # marks ~9% of Parakeet Ultra words here; 60% of those are wrong (base rate 14%)

KNOWLEDGE = """How aphasic speech commonly goes wrong:
- Word-finding trouble: pauses, fillers, empty phrases ("the thing", "that one"), talking around a word.
- Phonemic paraphasias: sounds swapped, added or dropped ("tup" for "cup", "pasketti" for "spaghetti"), sometimes several attempts in a row homing in on the word ("t- ta- table").
- Semantic paraphasias: a related but wrong word ("fork" for "spoon", "brother" for "son").
- Neologisms: made-up words.
- Agrammatism: short utterances missing small words and endings ("girl kick ball"); or the opposite, fluent but confused sentences.
- Repetitions, false starts, abandoned phrases, and perseveration on earlier words.
- Some speakers also have effortful or slurred articulation.

How speech recognition commonly fails on this speech:
- It turns a distorted or made-up word into the nearest real word, often one that makes no sense in context.
- It drops or inserts short function words, and joins or splits words.
- It smooths an ungrammatical utterance into fluent text, sometimes adding words that were never said.
- It mishears names, places and numbers."""

INTRO = ("You are checking an automatic speech recognition (ASR) transcript of a person with aphasia (a language "
         "disorder after stroke) talking with a clinician. The ASR system was trained on typical speech and often gets "
         "aphasic speech wrong. You cannot hear the audio: you only see the ASR transcript of this utterance and of a few "
         "earlier ones.")

TASK = {
    "said": ("Decide where this transcript is probably wrong and rewrite it as the most likely clean transcript of what "
             "the speaker actually said. Leave out fillers (um, uh), cut-off word fragments and abandoned or repeated "
             "words. Keep the speaker's own words, including wrong words, made-up words and ungrammatical phrasing, and "
             "do not complete their sentences. Change a word only when it is implausible as something this speaker said "
             "here and a similar-sounding alternative fits clearly better. If the transcript looks right, return it "
             "unchanged. Output only the transcript, on one line."),
    "meant": ("Decide where this transcript is probably wrong and rewrite it as what the speaker was trying to say. Fix "
              "likely recognition errors; where the speaker probably produced a wrong or distorted word and the intended "
              "word is clear from context, write the intended word. Leave out fillers (um, uh), cut-off word fragments and "
              "abandoned or repeated words. Do not add ideas the speaker did not express, do not complete unfinished "
              "sentences, and keep their sentence structure. If the transcript looks right, return it unchanged. Output "
              "only the transcript, on one line."),
}
VARIANTS = {  # variant: (source ASR model or derived-relative path, task, annotation: None | "conf" | "alts")
    "said": ("whisper-large-v3", "said", None),
    "meant": ("whisper-large-v3", "meant", None),
    "said-conf": ("parakeet-ultra-0.6b", "said", "conf"),
    "said-ultra": ("parakeet-ultra-0.6b", "said", None),  # control: same source without confidence marks
    "meant-reason": ("whisper-large-v3", "meant", None),  # same as meant, with the model allowed to think first
    "said-alts": ("hyps_alts/parakeet-ultra-onnx", "said", "alts"),  # Parakeet Ultra with its alternative tokens
}
REASONING = {"meant-reason": "low"}  # everything else runs with reasoning off, as in the live stack


def marked_text(h):
    """Rebuild words from SentencePiece tokens (leading space starts a word) and mark unsure ones with (?)."""
    words = []
    for t in h.get("tokens") or []:
        piece = t["t"]
        if piece.startswith(" ") or not words:
            words.append([piece.strip(), t["p"]])
        else:
            words[-1][0] += piece
            words[-1][1] = min(words[-1][1], t["p"])
    if not words:
        return h["hyp"]
    return " ".join(w + ("(?)" if p < CONF_THRESHOLD else "") for w, p in words if w)


def alts_text(h, min_p=0.05):
    """Words, with the alternatives the decoder considered for an unsure word-initial token: year{here 0.28}."""
    words = []  # [text, [token dicts]]
    for t in h.get("tokens") or []:
        if t["t"].startswith(" ") or not words:
            words.append([t["t"].strip(), [t]])
        else:
            words[-1][0] += t["t"]
            words[-1][1].append(t)
    def p_same(t):  # probability of the chosen piece, counting case-only variants as the same choice
        c = t["t"].strip().lower()
        return t["p"] + sum(p for piece, p in t["alts"] if piece != "<blank>" and piece.strip().lower() == c)

    out = []
    for text, toks in words:
        real = [t for t in toks if any(ch.isalnum() for ch in t["t"])]  # ignore unsure punctuation
        weakest = min(real, key=p_same) if real else None
        if weakest is None or p_same(weakest) >= CONF_THRESHOLD:
            out.append(text)
        elif weakest is toks[0]:
            chosen = toks[0]["t"].strip().lower()
            alts = []
            for piece, p in weakest["alts"]:
                if p < min_p:
                    continue
                if piece == "<blank>":
                    alts.append(f"(no word) {p:.2f}")
                elif piece.strip().strip(".,?!") and piece.strip().lower() != chosen:
                    alts.append(f"{piece.strip()}{'...' if len(toks) > 1 else ''} {p:.2f}")
            out.append(text + ("{" + ", ".join(alts) + "}" if alts else "(?)"))
        else:
            out.append(text + "(?)")
    return " ".join(out) if out else h["hyp"]


def system_prompt(task, mode):
    note = {
        "conf": "\n\nWords the ASR system itself was unsure about are marked (?). Unmarked words can still be wrong.",
        "alts": ("\n\nWhere the ASR system was unsure, the other words it considered are shown in curly brackets with "
                 "their probabilities, e.g. bowl{ball 0.31, bell 0.06}; '...' means only the start of that alternative "
                 "is known, and '(no word)' means it considered that nothing was said there. (?) marks an unsure word "
                 "with no clear alternative. Unmarked words can still be wrong."),
    }.get(mode, "")
    return f"{INTRO}\n\n{KNOWLEDGE}{note}\n\n{TASK[task]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("derived")
    ap.add_argument("variant", choices=sorted(VARIANTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="qwen-3.8-27b")
    ap.add_argument("--env", default=".env")
    args = ap.parse_args()

    derived = Path(args.derived)
    source, task, mode = VARIANTS[args.variant]
    rows = usable([json.loads(l) for l in open(derived / "utterances.jsonl", encoding="utf-8")])
    src_path = derived / f"{source}.jsonl" if "/" in source else derived / "hyps" / f"{source}.jsonl"
    hyps = {json.loads(l)["id"]: json.loads(l) for l in open(src_path, encoding="utf-8")}
    if args.limit:
        rows = rows[: args.limit]
    out_path = derived / "hyps_rewrite" / f"rewrite-{args.variant}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")} if out_path.exists() else set()

    history = {}  # speaker -> earlier transcripts from the same ASR system, in transcript order
    system = system_prompt(task, mode)
    key = api_key(args.env)
    last, marked_words, total_words = 0.0, 0, 0
    print(f"{args.variant}: {len([r for r in rows if r['id'] not in done])} to do, source {source}", flush=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for n, r in enumerate(rows, 1):
            h = hyps[r["id"]]
            ctx = history.setdefault(r["speaker"], [])
            text = {"conf": marked_text, "alts": alts_text}.get(mode, lambda x: x["hyp"])(h)
            if r["id"] not in done:
                if not h["hyp"].strip():
                    out = {"id": r["id"], "hyp": "", "raw": "", "sec": 0, "skipped": "empty ASR output"}
                else:
                    total_words += len(text.split())
                    marked_words += text.count("(?)") + text.count("{")
                    user = ("Earlier in the conversation (ASR transcript, may contain errors):\n"
                            + ("\n".join(f"- {c}" for c in ctx[-3:]) if ctx else "- (start of recording)")
                            + f"\n\nASR transcript of this utterance:\n{text}")
                    effort = REASONING.get(args.variant, "none")
                    payload = {"model": args.model, "temperature": 0, "reasoning_effort": effort,
                               "max_completion_tokens": 200 if effort == "none" else 2000,
                               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
                    time.sleep(max(0.0, MIN_INTERVAL - (time.time() - last)))
                    last = time.time()
                    data, headers, wall = call(key, payload)
                    raw = (data["choices"][0]["message"].get("content") or "").strip()
                    usage = data.get("usage", {})
                    out = {"id": r["id"], "hyp": raw.splitlines()[0].strip().strip('"') if raw else "", "raw": raw,
                           "sec": round(wall, 3), "asr_in": text,
                           "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
                    if not budget_ok(headers):
                        print("  key budget below 60%; pausing 15s", flush=True)
                        time.sleep(15)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
            ctx.append(h["hyp"])
            if n % 100 == 0 or n == len(rows):
                rate = f", {100*marked_words/total_words:.0f}% of words marked" if mode and total_words else ""
                print(f"  {n}/{len(rows)}{rate}", flush=True)


if __name__ == "__main__":
    main()
