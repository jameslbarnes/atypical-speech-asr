"""Per-speaker few-utterance tuning of Whisper large-v3 with LoRA (TORGO).
For each speaker: base (no tuning), own k (k of the speaker's sentences), diff 10 (10 sentences of another speaker).
All conditions are scored on the same held-out clips: the speaker's test clips whose text is in no training set.
Settings fixed before seeing results: LoRA r=16 on q_proj,v_proj (encoder and decoder), batch 2, about 80 steps for k=10
and 120 for k=30, two learning rates reported side by side.
Usage: whisper_fewshot.py --manifest m.jsonl --out o.jsonl --speakers F01,F03 [--lrs 1e-4,1e-3] [--ks 10,30]
"""
import argparse, json, random, re
import librosa, numpy as np, soundfile as sf, torch
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--speakers", required=True)
ap.add_argument("--lrs", default="1e-4,1e-3"); ap.add_argument("--ks", default="10,30")
ap.add_argument("--limit-eval", type=int, default=0)
args = ap.parse_args()
LRS = [float(x) for x in args.lrs.split(",")]; KS = [int(x) for x in args.ks.split(",")]
MODEL = "openai/whisper-large-v3"

def norm(t):
    return " ".join(re.sub(r"[.,?!;:\"'()\[\]]", " ", (t or "").lower()).split())

rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
by = {}
for r in rows:
    by.setdefault(r["speaker"], []).append(r)
speakers = sorted(by)
cache = {}
def audio(r):
    if r["id"] not in cache:
        x, sr = sf.read(r["wav"], dtype="float32")
        if x.ndim > 1: x = x.mean(axis=1)
        if sr != 16000: x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        cache[r["id"]] = x
    return cache[r["id"]]

proc = WhisperProcessor.from_pretrained(MODEL, language="en", task="transcribe")
base = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32).cuda()
base.generation_config.language = "en"; base.generation_config.task = "transcribe"

def feats(rs):
    return proc.feature_extractor([audio(r) for r in rs], sampling_rate=16000, return_tensors="pt").input_features.cuda()

def distinct(pool, rng):
    pool = list(pool); rng.shuffle(pool); seen, out = set(), []
    for r in pool:
        if norm(r["text"]) not in seen:
            seen.add(norm(r["text"])); out.append(r)
    return out

@torch.no_grad()
def evaluate(model, ev):
    model.eval(); hyps = []
    for i in range(0, len(ev), 16):
        b = ev[i:i + 16]
        with torch.autocast("cuda", dtype=torch.float16):
            ids = model.generate(input_features=feats(b), language="en", task="transcribe", max_new_tokens=120, num_beams=1)
        hyps += proc.batch_decode(ids, skip_special_tokens=True)
    return hyps

def train(model, tr, lr, steps, seed):
    rng = random.Random(seed); model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0); scaler = torch.cuda.amp.GradScaler()
    order, step, last = [], 0, 0.0
    while step < steps:
        if len(order) < 2:
            order = list(tr); rng.shuffle(order)
        b, order = order[:2], order[2:]
        tok = proc.tokenizer([r["text"] for r in b], padding=True, return_tensors="pt")
        lab = tok.input_ids.masked_fill(tok.attention_mask == 0, -100)
        if (lab[:, 0] == base.config.decoder_start_token_id).all():
            lab = lab[:, 1:]
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 5)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_features=feats(b), labels=lab.cuda()).loss
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update()
        step += 1; last = float(loss)
    return last

out = open(args.out, "a", encoding="utf-8")
def write(ev, hyps, s, cond, k, lr, extra):
    for r, h in zip(ev, hyps):
        out.write(json.dumps({"id": r["id"], "speaker": s, "cond": cond, "k": k, "lr": lr, "ref": r["text"], "hyp": h,
                              "kind": r["kind"], "dur": r["dur"], **extra}) + "\n")
    out.flush()

for s in args.speakers.split(","):
    rng = random.Random(11 + speakers.index(s))
    ok = lambda r: r.get("ex_ok") and 1.0 <= r["dur"] <= 15
    own = distinct([r for r in by[s] if ok(r)], rng)
    kmax = max(k for k in KS if len(own) >= k + 15) if any(len(own) >= k + 15 for k in KS) else min(KS)
    ks = [k for k in KS if k <= kmax]
    o = speakers[(speakers.index(s) + 1) % len(speakers)]
    own_texts = {norm(r["text"]) for r in own[:kmax]}
    diff = [r for r in distinct([r for r in by[o] if ok(r)], rng) if norm(r["text"]) not in own_texts][:10]
    excl = own_texts | {norm(r["text"]) for r in diff}
    ev = [r for r in by[s] if r.get("test_ok") and 0.3 <= r["dur"] <= 30 and norm(r["text"]) not in excl]
    if args.limit_eval: ev = ev[:args.limit_eval]
    print(s, "own distinct", len(own), "ks", ks, "diff from", o, len(diff), "eval clips", len(ev),
          "sentences", sum(r["kind"] == "sentence" for r in ev), flush=True)
    write(ev, evaluate(base, ev), s, "base", 0, 0.0, {})
    jobs = [("own", k, own[:k]) for k in ks] + [("diff", 10, diff)]
    for cond, k, tr in jobs:
        for lr in LRS:
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none")
            model = get_peft_model(base, cfg)
            steps = 80 if k <= 10 else 120
            loss = train(model, tr, lr, steps, seed=1)
            hyps = evaluate(model, ev)
            write(ev, hyps, s, cond, k, lr, {"train_ids": [r["id"] for r in tr], "train_speaker": tr[0]["speaker"], "final_loss": round(loss, 3)})
            print(s, cond, k, lr, "final loss", round(loss, 3), flush=True)
            base = model.unload(); del model; torch.cuda.empty_cache()
print("done", flush=True)
