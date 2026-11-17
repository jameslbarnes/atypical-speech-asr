"""LLM correction layer over several ASR hypotheses, via Cerebras (OpenAI-compatible API).

Usage: python llm_correct.py <derived_dir> <variant> [--model qwen-3.8-27b] [--limit N] [--env PATH]
  variant: select  - the LLM picks one of the hypotheses (cannot invent words)
           fuse    - the LLM writes the clean transcript, using the hypotheses as evidence
  reads  <derived_dir>/llm_inputs.jsonl
  writes <derived_dir>/hyps_llm/llm-<variant>.jsonl (resumable)

Only ASR hypotheses go to the API, never the reference transcripts. Cerebras does not retain
inputs/outputs on the regular (non-batch) API. Our key was shared with a live service, so requests
are sequential, paced to ~1/s, and back off whenever the key's per-minute budget drops below 60%.
"""

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.cerebras.ai/v1/chat/completions"
CONTEXT_MODEL = "whisper-large-v3"  # earlier utterances are shown as this model heard them
N_CONTEXT = 3
MIN_INTERVAL = 1.0

INTRO = ("Several speech-recognition systems transcribed the same short audio clip of a person with aphasia "
         "(a language disorder after stroke) talking with a clinician. Some systems write down every repeated "
         "word and false start; others tidy them away. They also make different recognition errors.")
RULES = ("A clean transcript leaves out fillers (um, uh), cut-off word fragments, and words or phrases the speaker "
         "repeated or abandoned before restarting, but keeps the speaker's actual words, even when they are the "
         "wrong word, a made-up word, or ungrammatical. Never correct the speaker or complete their sentences.")
SYSTEM = {
    "fuse": f"{INTRO}\n\nWrite the single most likely clean transcript of what the speaker said in this clip. {RULES} "
            "Resolve disagreements between systems using agreement between them, sound-alike plausibility and the "
            "context. If no system heard any words, output nothing. Output only the transcript, on one line.",
    "select": f"{INTRO}\n\nPick the transcript most likely to match a clean transcript of the clip. {RULES} "
              "Reply with only its number.",
}


def api_key(env_path):
    for line in Path(env_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("CEREBRAS_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit(f"CEREBRAS_API_KEY not found in {env_path}")


def call(key, payload, retries=5):
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        req = urllib.request.Request(API, data=body, headers={"Authorization": f"Bearer {key}",
                                                              "Content-Type": "application/json",
                                                              "User-Agent": "aphasia-asr-bench/0.1"})
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
                return data, dict(r.headers), time.time() - t
        except urllib.error.HTTPError as e:
            wait = 20 if e.code == 429 else 3 * (attempt + 1)
            print(f"  HTTP {e.code} {e.read()[:200]!r}; retry in {wait}s", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  {type(e).__name__}; retry", flush=True)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("request failed after retries")


def budget_ok(headers):
    """True while the key has at least 60% of its per-minute requests and tokens left."""
    try:
        r = int(headers["x-ratelimit-remaining-requests-minute"]) / int(headers["x-ratelimit-limit-requests-minute"])
        k = int(headers["x-ratelimit-remaining-tokens-minute"]) / int(headers["x-ratelimit-limit-tokens-minute"])
        return min(r, k) >= 0.6
    except (KeyError, ValueError, ZeroDivisionError):
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("derived")
    ap.add_argument("variant", choices=sorted(SYSTEM))
    ap.add_argument("--model", default="qwen-3.8-27b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--env", default=".env")
    args = ap.parse_args()

    derived = Path(args.derived)
    recs = [json.loads(l) for l in open(derived / "llm_inputs.jsonl", encoding="utf-8")]
    if args.limit:
        recs = recs[: args.limit]
    out_path = derived / "hyps_llm" / f"llm-{args.variant}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")} if out_path.exists() else set()

    by_speaker = {}
    for r in recs:  # llm_inputs is in transcript order within each speaker
        by_speaker.setdefault(r["speaker"], []).append(r)
    position = {r["id"]: (r["speaker"], i) for spk, rs in by_speaker.items() for i, r in enumerate(rs)}

    key = api_key(args.env)
    last = 0.0
    todo = [r for r in recs if r["id"] not in done]
    print(f"{args.variant}: {len(todo)} to do ({len(done)} done) with {args.model}", flush=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for n, r in enumerate(todo, 1):
            models = sorted(r["hyps"])
            random.Random(r["id"]).shuffle(models)  # no fixed position for any system
            spk, i = position[r["id"]]
            ctx = [p["hyps"][CONTEXT_MODEL]["text"] for p in by_speaker[spk][max(0, i - N_CONTEXT): i]]
            lines = [f"{k}. {r['hyps'][m]['text'] or '(nothing)'}" for k, m in enumerate(models, 1)]
            user = ("Earlier in the conversation (automatic transcript, may contain errors):\n"
                    + ("\n".join(f"- {c}" for c in ctx) if ctx else "- (start of recording)")
                    + "\n\nTranscripts of the clip:\n" + "\n".join(lines))
            payload = {"model": args.model, "temperature": 0, "reasoning_effort": "none",
                       "max_completion_tokens": 8 if args.variant == "select" else 200,
                       "messages": [{"role": "system", "content": SYSTEM[args.variant]},
                                    {"role": "user", "content": user}]}
            time.sleep(max(0.0, MIN_INTERVAL - (time.time() - last)))
            last = time.time()
            data, headers, wall = call(key, payload)
            raw = (data["choices"][0]["message"].get("content") or "").strip()
            if args.variant == "select":
                m = re.search(r"\d+", raw)
                pick = int(m.group()) if m and 1 <= int(m.group()) <= len(models) else None
                hyp = r["hyps"][models[pick - 1]]["text"] if pick else r["hyps"][CONTEXT_MODEL]["text"]
                extra = {"picked": models[pick - 1] if pick else None}
            else:
                hyp = raw.splitlines()[0].strip().strip('"') if raw else ""
                extra = {}
            usage = data.get("usage", {})
            f.write(json.dumps({"id": r["id"], "hyp": hyp, "raw": raw, "sec": round(wall, 3),
                                "server_sec": round((data.get("time_info") or {}).get("total_time", 0), 4),
                                "prompt_tokens": usage.get("prompt_tokens"),
                                "completion_tokens": usage.get("completion_tokens"), **extra},
                               ensure_ascii=False) + "\n")
            f.flush()
            if not budget_ok(headers):
                print("  key budget below 60%; pausing 15s", flush=True)
                time.sleep(15)
            if n % 100 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
