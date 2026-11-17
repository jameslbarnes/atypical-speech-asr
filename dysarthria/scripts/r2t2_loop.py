"""Confucius4-R2T2 version (whole-clip decoding through the transformers backend).
Session-loop simulation: confirmed utterances arrive one at a time and the recognizer is retuned on all so far.
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
from peft import LoraConfig, inject_adapter_in_model
from qwen_asr import Qwen3ASRModel

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True); ap.add_argument("--speakers", required=True)
ap.add_argument("--model", default="netease-youdao/Confucius4-R2T2"); ap.add_argument("--mode", default="lora")
ap.add_argument("--save-dir", default=""); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--where", default="audio", choices=("audio", "llm", "both")); ap.add_argument("--rank", type=int, default=16); ap.add_argument("--step-mult", type=float, default=1.0)
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


wrapper = Qwen3ASRModel.from_pretrained(args.model, dtype=torch.float32, device_map="cuda:0",
                                        max_inference_batch_size=args.eval_bs, max_new_tokens=args.max_new)
base = wrapper.model; proc = wrapper.processor
for p in base.parameters():
    p.requires_grad_(False)
REGEX = {"audio": r"thinker\.audio_tower\.layers\.\d+\.self_attn\.(q_proj|v_proj)",
         "llm": r"thinker\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)",
         "both": r"thinker\.(audio_tower|model)\.layers\.\d+\.self_attn\.(q_proj|v_proj)"}[args.where]
inject_adapter_in_model(LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, target_modules=REGEX, bias="none"), base)
LORA = [m for m in base.modules() if hasattr(m, "lora_A") and hasattr(m, "reset_lora_parameters")]
print("adapter layers", len(LORA), "trainable", sum(p.numel() for p in base.parameters() if p.requires_grad), flush=True)


def save_merged(dst, ev):
    """Write the tuned model as a plain checkpoint (adapter folded into the weights) for the streaming engine."""
    import glob, shutil
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    os.makedirs(dst, exist_ok=True)
    for m in LORA:
        m.merge()
    sd = {k.replace(".base_layer.", "."): v.detach().to(torch.bfloat16).cpu().contiguous()
          for k, v in base.state_dict().items() if "lora_" not in k}
    for m in LORA:
        m.unmerge()
    save_file(sd, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
    src = snapshot_download(args.model)
    for f in glob.glob(os.path.join(src, "*")):
        if not f.endswith(".safetensors") and not f.endswith("model.safetensors.index.json") and os.path.isfile(f):
            shutil.copy(f, dst)
    json.dump([r["id"] for r in ev], open(os.path.join(dst, "eval_ids.json"), "w"))
    print("saved", dst, len(sd), "tensors", flush=True)


def reset_adapter():
    """Back to the untuned model: the adapter's second matrix is zero, so it adds nothing."""
    for m in LORA:
        m.reset_lora_parameters("default", True)


PREFIX = proc.apply_chat_template([[{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": None}]}]],
                                  add_generation_prompt=True, tokenize=False)[0]
EOS = proc.tokenizer.eos_token or ""
PROMPT = PREFIX + "language English<asr_text>"


@torch.no_grad()
def evaluate(model, ev):
    base.eval(); hyps = []
    for i in range(0, len(ev), args.eval_bs):
        b = ev[i:i + args.eval_bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res = wrapper.transcribe(audio=[(audio(r), 16000) for r in b], language=["English"] * len(b))
        hyps += [(x.text or "").strip() for x in res]
    return hyps


def train(model, pairs, seed):
    n = len(pairs)
    lr, bs, steps = args.lr, 2, int(args.step_mult * min(120, max(20, 8 * n)))
    rng = random.Random(seed); base.eval()
    params = [p for p in base.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    order, last = [], 0.0
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 5)
        opt.zero_grad(set_to_none=True); tot = 0.0
        # One recording per forward pass, two per update. Batching recordings of different lengths gave a
        # teacher-forced loss near 10 on text the model itself produced (0.07 to 0.24 one at a time), so padding is avoided.
        for _ in range(2):
            if not order:
                order = list(pairs); rng.shuffle(order)
            r, t = order.pop()
            aud = [audio(r)]
            fi = proc(text=[PROMPT + t + EOS], audio=aud, return_tensors="pt", padding=True, truncation=False)
            pl = int(proc(text=[PROMPT], audio=aud, return_tensors="pt", padding=True, truncation=False)["attention_mask"].sum())
            labels = fi["input_ids"].clone(); labels[0, :pl] = -100
            fi = {k: v.to("cuda") for k, v in fi.items()}
            base.thinker.rope_deltas = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = base.thinker(**fi, labels=labels.to("cuda")).loss / 2
            loss.float().backward(); tot += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); last = tot
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
            reset_adapter(); model = base
            loss, steps = train(model, pairs, seed)
            secs = time.time() - t0
            hyps = evaluate(model, ev)
            if args.save_dir and arm == "loop" and n == max(NS):
                save_merged(os.path.join(args.save_dir, f"{s}_{seed}"), ev)
            write(ev, hyps, s, seed, arm, n, {"train_secs": round(secs, 1), "steps": steps, "final_loss": round(loss, 3),
                                              "train_audio_s": round(sum(r["dur"] for r, _ in pairs), 1),
                                              "train_ids": [r["id"] for r, _ in pairs], "lr": args.lr, "where": args.where})
            print(s, seed, arm, n, "loss", round(loss, 3), "train", round(secs, 1), "s", flush=True)
            reset_adapter(); torch.cuda.empty_cache()
            return
        except torch.OutOfMemoryError:
            print("OOM, retrying", s, seed, arm, n, flush=True)
            reset_adapter(); torch.cuda.empty_cache(); time.sleep(20)


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
        reset_adapter()
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
