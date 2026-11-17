"""Session-loop simulation: confirmed utterances arrive one at a time and the recognizer is retuned on all so far.
For each speaker and seed: a random arrival order of up to 30 of the speaker's utterances (distinct texts); after
n = 1,3,5,10,20,30 the model is tuned from the untuned weights on the first n and scored on held-out clips whose text
is in none of the 30. Arms:
  loop    labels = the confirmed transcript
  noise20 20% of labels replaced by the untuned recognizer's own output (wrong confirmations)
  self    all labels = the untuned recognizer's own output (no human confirmation at all)
  alt     labels from another reference level (e.g. intended words), at the largest n only
Modes: lora (rank 16 on q_proj,v_proj; lr 1e-4; batch 2; steps = clamp(8n, 20, 120)) or
       full (the published EvolveCaptions recipe: full fine-tune, lr 1e-5, batch 8, 100 steps).
"""
import argparse, json, os, random, re, time
import librosa, soundfile as sf, torch
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True); ap.add_argument("--speakers", required=True)
ap.add_argument("--model", default="openai/whisper-large-v3"); ap.add_argument("--mode", default="lora")
ap.add_argument("--seeds", default="1,2,3"); ap.add_argument("--ns", default="1,3,5,10,20,30")
ap.add_argument("--label", default="", help="reference level to train on; default is the row text")
ap.add_argument("--alt", default=""); ap.add_argument("--noise-seeds", default="1,2")
ap.add_argument("--eval-cap", type=int, default=0); ap.add_argument("--max-new", type=int, default=120)
ap.add_argument("--eval-bs", type=int, default=8); ap.add_argument("--tag", default="")
args = ap.parse_args()
SEEDS = [int(x) for x in args.seeds.split(",")]; NS = [int(x) for x in args.ns.split(",")]
NOISE = [int(x) for x in args.noise_seeds.split(",")] if args.noise_seeds not in ("", "none") else []
PUNCT = re.compile(r"[.,?!;:\"'()\[\]]")


def norm(t):
    return " ".join(PUNCT.sub(" ", (t or "").lower()).split())


def lab(r, field=None):
    f = field or args.label
    return (r.get("refs") or {}).get(f, "") if f else r["text"]


rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
base_dir = os.path.dirname(os.path.abspath(args.manifest))
by = {}
for r in rows:
    if not os.path.isabs(r["wav"]):
        r["wav"] = os.path.join(base_dir, r["wav"])
    by.setdefault(r["speaker"], []).append(r)
speakers = sorted(by); cache = {}


def audio(r):
    if r["id"] not in cache:
        x, sr = sf.read(r["wav"], dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16000:
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        cache[r["id"]] = x[: 30 * 16000]
    return cache[r["id"]]


proc = WhisperProcessor.from_pretrained(args.model, language="en", task="transcribe")
base = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32).cuda()
base_state = {k: v.detach().clone() for k, v in base.state_dict().items()} if args.mode == "full" else None


fcache = {}


def feats(rs):
    for r in rs:
        if r["id"] not in fcache:
            fcache[r["id"]] = proc.feature_extractor([audio(r)], sampling_rate=16000, return_tensors="pt").input_features[0].half()
    return torch.stack([fcache[r["id"]] for r in rs]).float().cuda()


@torch.no_grad()
def evaluate(model, ev):
    model.eval(); hyps = []
    for i in range(0, len(ev), args.eval_bs):
        with torch.autocast("cuda", dtype=torch.float16):
            ids = model.generate(input_features=feats(ev[i:i + args.eval_bs]), language="en", task="transcribe",
                                 max_new_tokens=args.max_new, num_beams=1)
        hyps += proc.batch_decode(ids, skip_special_tokens=True)
    return hyps


def train(model, pairs, seed):
    n = len(pairs)
    lr, bs, steps = (1e-4, 2, min(120, max(20, 8 * n))) if args.mode == "lora" else (1e-5, 8, 100)
    rng = random.Random(seed); model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0); scaler = torch.amp.GradScaler("cuda")
    order, last = [], 0.0
    for step in range(steps):
        if len(order) < min(bs, n):
            order = list(pairs); rng.shuffle(order)
        b, order = order[:bs], order[bs:]
        tok = proc.tokenizer([t for _, t in b], padding=True, return_tensors="pt", truncation=True, max_length=200)
        y = tok.input_ids.masked_fill(tok.attention_mask == 0, -100)
        if (y[:, 0] == base.config.decoder_start_token_id).all():
            y = y[:, 1:]
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 5)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_features=feats([r for r, _ in b]), labels=y.cuda()).loss
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); last = float(loss)
    return last, steps


done = set()
if os.path.exists(args.out):
    old = []
    for l in open(args.out, encoding="utf-8"):
        try:
            old.append(json.loads(l))
        except Exception:
            pass
    cnt = {}
    for d in old:
        k = (d["speaker"], d["seed"], d["arm"], d["n"]); cnt[k] = cnt.get(k, 0) + 1
    # a run is complete only if it has as many rows as the untuned pass for the same speaker and seed
    done = {k for k, c in cnt.items() if c == cnt.get((k[0], k[1], "base", 0), -1)}
    with open(args.out, "w", encoding="utf-8") as f:
        for d in old:
            if (d["speaker"], d["seed"], d["arm"], d["n"]) in done:
                f.write(json.dumps(d, ensure_ascii=False) + chr(10))
    print("resume: complete runs", len(done), "dropped incomplete", len(cnt) - len(done), flush=True)
out = open(args.out, "a", encoding="utf-8")


def write(ev, hyps, s, seed, arm, n, extra):
    for r, h in zip(ev, hyps):
        rec = {"id": r["id"], "speaker": s, "seed": seed, "arm": arm, "n": n, "ref": lab(r), "hyp": h, "tag": args.tag, **extra}
        for k in ("refs", "meta", "kind", "dur"):
            if k in r:
                rec[k] = r[k]
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.flush()


def run(pairs, ev, s, seed, arm, n):
    global base
    pairs = [(r, t) for r, t in pairs if norm(t)]
    if not pairs or (s, seed, arm, n) in done:
        return
    for attempt in (1, 2, 3):
        model = None
        try:
            t0 = time.time()
            if args.mode == "lora":
                model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                        target_modules=["q_proj", "v_proj"], bias="none"))
            else:
                base.load_state_dict(base_state); model = base
                for p in model.parameters():
                    p.requires_grad_(True)
            loss, steps = train(model, pairs, seed)
            secs = time.time() - t0
            hyps = evaluate(model, ev)
            write(ev, hyps, s, seed, arm, n, {"train_secs": round(secs, 1), "steps": steps, "final_loss": round(loss, 3),
                                              "train_audio_s": round(sum(r["dur"] for r, _ in pairs), 1),
                                              "train_ids": [r["id"] for r, _ in pairs]})
            print(s, seed, arm, n, "loss", round(loss, 3), "train", round(secs, 1), "s", flush=True)
            if args.mode == "lora":
                base = model.unload()
            torch.cuda.empty_cache()
            return
        except torch.OutOfMemoryError:
            print("OOM, retrying", s, seed, arm, n, flush=True)
            if args.mode == "lora" and model is not None:
                base = model.unload()
            torch.cuda.empty_cache(); time.sleep(20)


for s in args.speakers.split(","):
    def usable(r):
        w = len(norm(lab(r)).split())
        return r.get("ex_ok", w >= 3) and 1.0 <= r["dur"] <= 15 and w >= 1

    for seed in SEEDS:
        rng = random.Random(seed * 1000 + speakers.index(s))
        pool = [r for r in by[s] if usable(r)]; rng.shuffle(pool)
        seen, arrive = set(), []
        for r in pool:
            if norm(lab(r)) not in seen:
                seen.add(norm(lab(r))); arrive.append(r)
        nmax = max([n for n in NS if len(arrive) >= n + 15] or [min(NS)])
        arrive = arrive[:nmax]; ns = [n for n in NS if n <= nmax]
        excl = {norm(lab(r)) for r in arrive}; ids = {r["id"] for r in arrive}
        ev = [r for r in by[s] if r.get("test_ok", True) and 0.3 <= r["dur"] <= 30 and norm(lab(r))
              and norm(lab(r)) not in excl and r["id"] not in ids]
        if args.eval_cap and len(ev) > args.eval_cap:
            keep = [r for r in ev if (r.get("meta") or {}).get("n_paraphasia", 0) > 0]
            rest = [r for r in ev if (r.get("meta") or {}).get("n_paraphasia", 0) == 0]
            random.Random(99 + speakers.index(s)).shuffle(rest)
            ev = keep + rest[: max(0, args.eval_cap - len(keep))]
        print(s, "seed", seed, "arrivals", len(arrive), "ns", ns, "eval", len(ev), flush=True)
        if args.mode == "full":
            base.load_state_dict(base_state)
        if (s, seed, "base", 0) not in done:
            write(ev, evaluate(base, ev), s, seed, "base", 0, {})
        own_hyp = evaluate(base, arrive)
        for n in ns:
            run([(r, lab(r)) for r in arrive[:n]], ev, s, seed, "loop", n)
        if seed in NOISE:
            for n in [x for x in (10, 30) if x in ns]:
                bad = set(random.Random(seed).sample(range(n), max(1, round(0.2 * n))))
                run([(r, own_hyp[i] if i in bad else lab(r)) for i, r in enumerate(arrive[:n])], ev, s, seed, "noise20", n)
                run([(r, own_hyp[i]) for i, r in enumerate(arrive[:n])], ev, s, seed, "self", n)
        if args.alt:
            run([(r, lab(r, args.alt)) for r in arrive[:nmax]], ev, s, seed, "alt", nmax)
print("done", flush=True)
