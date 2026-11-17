"""Parse APROCSA CHAT transcripts into participant utterances with layered references.

Reference levels (all built from the same CHAT line):
  L0 verbatim   - everything produced: fillers, fragments, retraced material, paraphasias as said
  L1 -fillers   - L0 without &-fillers
  L2 -fragments - L1 without &+fragments
  L3 clean      - L2 without retraced/repeated material ([/] [//] [///] ...)
  L4 intended   - L3 with paraphasias replaced by their [: target]

Usage: python prepare.py <data_dir>
  reads  <data_dir>/cha/aprocsa<ID>a.cha and <data_dir>/wav/<ID>.wav
  writes <data_dir>/derived/utterances.jsonl
"""

import json
import re
import sys
import wave
from pathlib import Path

SPEAKER_IDS = ["1738", "1944", "1713", "1554", "1833", "1731"]


def load_speakers(data):
    """Participant scores are part of the dataset's participant table, which may not be redistributed, so they
    are read from <data_dir>/speakers.json, which you write from that table:
    {"1738": {"qab": <QAB overall, 0-10>, "impairment": <consensus overall communication impairment, 0-4>}, ...}"""
    meta = json.loads((data / "speakers.json").read_text(encoding="utf-8"))
    return {s: meta[s] for s in SPEAKER_IDS}

BULLET = re.compile(r"\x15(\d+)_(\d+)\x15")
TOKEN = re.compile(r"\[[^\]]*\]|<|>|[^\s<>\[\]]+")
PAUSE = re.compile(r"^\(\d*\.*\d*\)$|^\(\.+\)$")
RETRACE = re.compile(r"^/{1,3}[-?]?$")
LEGACY_FILLERS = {"uh", "um", "er", "erm", "ah", "eh", "hm", "hmm", "mm", "uhm"}


def clean_word(tok):
    """Strip CHAT word-internal notation; return (surface, is_phonological)."""
    phon = "@u" in tok
    w = re.sub(r"@[a-z:]+$", "", tok)          # @l @o @n @u @s:spa ...
    w = re.sub(r"\(([^)]*)\)", r"\1", w)        # (be)cause -> because
    w = w.replace("_", " ").replace("+", " ")    # compounds / multiword names
    w = re.sub(r"[:\^↑↓ˈˌ≈≠]", "", w) if not phon else w
    return w.strip(), phon


def parse_line(text):
    """Return (items, flags) for a CHAT main-tier utterance (bullet already removed)."""
    items, scope_stack = [], []
    last_span = None
    flags = {"unintelligible": False, "overlap": False}
    text = text.replace("+<", " +< ")  # lazy-overlap linker; must not be read as a scope opener
    for tok in re.findall(r"\+<|" + TOKEN.pattern, text):
        if tok == "<":
            scope_stack.append(len(items))
            continue
        if tok == ">":
            start = scope_stack.pop() if scope_stack else max(len(items) - 1, 0)
            last_span = (start, len(items))
            continue
        if tok.startswith("["):
            ann = tok[1:-1].strip()
            if ann in ("<", ">") or re.fullmatch(r"[<>]\d*", ann):
                flags["overlap"] = True
                continue
            if last_span is None:
                continue
            span = range(*last_span)
            if RETRACE.match(ann):
                for i in span:
                    items[i]["retraced"] = True
            elif ann.startswith(":"):
                target = ann.lstrip(":").strip()
                target = clean_word(target)[0] if " " not in target else " ".join(clean_word(t)[0] for t in target.split())
                for n, i in enumerate(span):
                    items[i]["target"] = target if n == 0 else ""
            elif ann.startswith("*"):
                for i in span:
                    items[i]["error"] = ann[1:].strip() or "*"
            continue  # [+ gram], [!], [?], [=! ...], [% ...], [^ ...] etc. are ignored
        if tok.startswith("+"):
            if tok in ("+<",):
                flags["overlap"] = True
            continue  # terminators/linkers: +... +/. +//. +" +^ +, ++
        if tok in (".", "?", "!", "‡", "„", ",", ";") or PAUSE.match(tok):
            continue
        if tok.startswith("&="):
            continue  # gestures, laughs, events
        if tok.startswith("&-"):
            items.append({"form": tok[2:], "kind": "filler"})
        elif tok.startswith("&+"):
            items.append({"form": tok[2:], "kind": "frag"})
        elif tok.startswith("&"):
            body = tok[1:]
            items.append({"form": body, "kind": "filler" if body.lower() in LEGACY_FILLERS else "frag"})
        elif re.match(r"^(xxx|yyy|www)", tok):
            flags["unintelligible"] = True
            continue
        elif tok.startswith("0"):
            continue  # omitted word, not spoken
        else:
            form, phon = clean_word(tok)
            if not form:
                continue
            items.append({"form": form, "kind": "phon" if phon else "word"})
        last_span = (len(items) - 1, len(items))
    return items, flags


def levels(items):
    def join(seq):
        return " ".join(s for s in seq if s).strip()
    keep_l1 = [it for it in items if it["kind"] != "filler"]
    keep_l2 = [it for it in keep_l1 if it["kind"] != "frag"]
    keep_l3 = [it for it in keep_l2 if not it.get("retraced")]
    return {
        "L0": join(it["form"] for it in items),
        "L1": join(it["form"] for it in keep_l1),
        "L2": join(it["form"] for it in keep_l2),
        "L3": join(it["form"] for it in keep_l3),
        "L4": join(it.get("target", it["form"]) for it in keep_l3),
    }


def read_main_tiers(path):
    """Yield (speaker, text) for each main tier, joining continuation lines."""
    cur = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("*"):
            if cur:
                yield cur
            spk, _, rest = raw[1:].partition(":")
            cur = [spk, rest.strip()]
        elif raw.startswith("\t") and cur is not None:
            cur[1] += " " + raw.strip()
        else:
            if cur:
                yield cur
            cur = None
    if cur:
        yield cur


def impossible(start, end, text):
    """A bullet faster than ~16 words/s means the forced alignment collapsed there."""
    words = len([w for w in text.split() if re.match(r"^[A-Za-z]", w)])
    return words >= 3 and (end - start) < 0.06 * words + 0.1


def broken_regions(tiers):
    """Time spans with 4+ impossible bullets within 3 minutes (any speaker): timing there is unreliable."""
    bad = sorted(s for s, e, t in tiers if impossible(s, e, t))
    regions = []
    for i, t in enumerate(bad):
        cluster = [b for b in bad[i:] if b - t <= 180]
        if len(cluster) >= 4:
            regions.append((t - 30, cluster[-1] + 60))
    merged = []
    for a, b in sorted(regions):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    for i, (a, b) in enumerate(merged):  # extend through stragglers that keep appearing after a cluster
        for t in bad:
            if b < t + 60 and t <= b + 180:
                b = t + 60
        merged[i] = (a, b)
    return merged


def wav_seconds(path):
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def main(data_dir):
    data = Path(data_dir)
    out = data / "derived"
    out.mkdir(exist_ok=True)
    rows = []
    for spk_id, meta in load_speakers(data).items():
        cha = data / "cha" / f"aprocsa{spk_id}a.cha"
        wav = data / "wav" / f"{spk_id}.wav"
        dur = wav_seconds(wav) if wav.exists() else None
        tiers = list(read_main_tiers(cha))
        timed = [(int(m.group(1)) / 1000, int(m.group(2)) / 1000, BULLET.sub(" ", t))
                 for _, t in tiers if (m := BULLET.search(t))]
        regions = broken_regions(timed)
        if regions:
            print(f"  {spk_id}: unreliable timing " + ", ".join(f"{a/60:.1f}-{b/60:.1f} min" for a, b in regions))
        for n, (who, text) in enumerate(tiers):
            if who != "PAR":
                continue
            m = BULLET.search(text)
            if not m:
                continue
            start, end = int(m.group(1)) / 1000, int(m.group(2)) / 1000
            body = BULLET.sub(" ", text)
            items, flags = parse_line(body)
            refs = levels(items)
            flags["timing_ok"] = not impossible(start, end, body) and not any(a <= start <= b for a, b in regions)
            rows.append({
                "id": f"{spk_id}_{n:04d}",
                "speaker": spk_id,
                **meta,
                "start": start,
                "end": end,
                "dur": round(end - start, 3),
                "wav": str(wav),
                "wav_dur": dur,
                "chat": body.strip(),
                "refs": refs,
                "n_paraphasia": sum(1 for it in items if it.get("error", "").startswith(("p", "s", "n", "m"))),
                "errors": [it["error"] for it in items if "error" in it],
                **flags,
            })
    with open(out / "utterances.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    usable = [r for r in rows if not r["unintelligible"] and not r["overlap"] and r["timing_ok"] and r["refs"]["L0"]]
    print(f"utterances: {len(rows)}  usable: {len(usable)}  (unintelligible {sum(r['unintelligible'] for r in rows)}, "
          f"overlap {sum(r['overlap'] for r in rows)}, bad timing {sum(not r['timing_ok'] for r in rows)})")
    for spk in SPEAKERS:
        u = [r for r in usable if r["speaker"] == spk]
        print(f"  {spk}: {len(u):4d} utts  {sum(r['dur'] for r in u)/60:5.1f} min  "
              f"L0 words {sum(len(r['refs']['L0'].split()) for r in u):5d}  "
              f"L3 words {sum(len(r['refs']['L3'].split()) for r in u):5d}  max {max((r['dur'] for r in u), default=0):.1f}s")
    print(f"  total: {sum(r['dur'] for r in usable)/60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[1])
