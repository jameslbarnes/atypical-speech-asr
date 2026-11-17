# Aphasia benchmark: open speech recognizers on aphasic speech

Ten open speech recognizers, run on the same recordings of people with aphasia and scored against five kinds of reference transcript. Then four ways a language model can sit on top of them. Measured Oct 22 to Nov 9, 2026.

## Data

The six openly shared participants of the APROCSA corpus: AphasiaBank-protocol videos and CHAT transcripts of six people with chronic aphasia after a stroke ([langneurosci.org/aprocsa-dataset](https://langneurosci.org/aprocsa-dataset/); corpus page at [TalkBank](https://talkbank.org/aphasia/access/English/Protocol/APROCSA.html)).

- The corpus may be used for research and education only, and may not be redistributed. Keep audio, transcripts, and anything containing transcripts (recognizer outputs, scored utterances) out of git and off shared storage. Only aggregate numbers are in `results/`.
- Cite Ezzes et al. (2022) and Casilio et al. (2026) with any use; full references are in the [top-level README](../README.md#data).
- `scripts/fetch_aprocsa.sh` downloads the videos and transcripts into `$DATA` (default `data/aprocsa`) and converts the audio to 16 kHz mono WAV. It needs `ffmpeg` (or set `FFMPEG`).

## Pipeline

```
python -I scripts/prepare.py data/aprocsa                         # CHAT -> utterances.jsonl with L0-L4 references
TRANSCRIBE_LIBRARY=/path/to/transcribe.dll bash scripts/run_lane.sh 4 <model> [<model> ...]   # transcribe (CPU, resumable)
python -I -X utf8 scripts/score.py data/aprocsa/derived results
```

Reference levels, all derived from the same CHAT line:

| level | contents |
|---|---|
| L0 | verbatim: fillers, fragments, repetitions, paraphasias as produced |
| L1 | L0 minus fillers |
| L2 | L1 minus word fragments |
| L3 | clean: L2 minus retraced or repeated material |
| L4 | intended: L3 with each paraphasia replaced by its coded `[: target]` |

Utterances that are unintelligible (`xxx`), overlap the interviewer, or are shorter than 0.3 s or longer than 30 s are excluded. Every model is scored on the same utterances.

Transcript timing: the CHAT time marks come from forced alignment, which collapsed in places (whole sentences stamped as 0.01–0.15 s). `prepare.py` drops any utterance with physically impossible timing, and any stretch with 4 or more such marks within 3 minutes. That removes one speaker (1944) from 28.9 minutes to the end of the transcript, where clips would otherwise contain the neighbouring sentences. `align_offsets.py` (recognizer word times against the `%wor` tier) confirmed the remaining timing is within about 0.1 s.

## Models and runtimes

CPU only. Most models run through [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) v0.4.0 with GGUF files from `huggingface.co/handy-computer` (Q8_0 for models of 2B parameters or more, F16 otherwise), with its GPU backend removed. `run_asr.py` reads models from `$GGUF_DIR` (default `models/gguf`) and also has a faster-whisper entry for a runtime cross-check, which was not run.

Notes on individual models:
- Cohere Transcribe's Hugging Face repo is gated; its Apache-2.0 GGUF mirror was used instead.
- Nemotron streaming gets a 0.5 s silent tail per clip; otherwise its last chunk is never flushed and short utterances come back empty. It still returns nothing for about 10% of utterances, mostly under 1 s.
- Whisper pads every clip to 30 s, so on short utterances it is by far the slowest on CPU.
- Voxtral Mini 3B was not run: about 1.2–2.7× real time on the shared CPU.

## Results (Oct 22, 2026: 10 models, 1,080 utterances, 70 minutes)

Full tables: [`results/summary.md`](results/summary.md). Word error rate (%), lower is better:

| model | L0 verbatim | L2 | L3 clean | L4 intended | CPU real-time factor |
|---|---|---|---|---|---|
| whisper-large-v3 | 30.6 | 21.1 | **20.6** | 20.2 | 1.18 |
| cohere-transcribe-2b | 29.6 | 20.3 | 21.3 | 20.7 | 0.24 |
| whisper-large-v3-turbo | 29.9 | 20.7 | 21.6 | 21.2 | 0.97 |
| granite-speech-4.1-2b | 30.5 | 21.5 | 22.1 | 21.7 | 0.54 |
| canary-1b-v2 | 30.4 | 21.5 | 22.7 | 22.4 | 0.14 |
| nemotron-streaming-0.6b | 37.4 | 27.9 | 25.1 | 25.0 | 0.16 |
| canary-qwen-2.5b | 20.8 | **17.6** | 29.1 | 28.7 | 0.36 |
| parakeet-tdt-0.6b-v2 | 20.2 | 19.0 | 29.4 | 29.1 | 0.08 |
| parakeet-tdt-0.6b-v3 | 20.2 | 19.8 | 31.5 | 31.3 | 0.08 |
| parakeet-ultra-0.6b (Moondream) | **19.5** | 18.3 | 30.0 | 29.7 | 0.06 |

- Two styles: Parakeet and Canary-Qwen write down repetitions and false starts; Whisper, Cohere, Granite and Canary-1B tidy them away. Which model "wins" depends entirely on the reference level.
- The top five at L3 are within each other's confidence intervals.
- Severity: about 10–16% for the mildest speaker, about 30–37% for the most severe.
- One- and two-word utterances: 42–49% for every model, against 17–28% for seven words or more.
- Utterances with coded paraphasias (117): 32.6% against 18.6% without (Whisper large-v3, L3).
- Runs of invented text (5 or more words not in even the verbatim reference) were rare: 0–5 utterances per model.
- Headroom for a language-model layer at L3: picking the best of the 10 transcripts per utterance would give 11.5%; picking, without labels, the transcript that agrees most with the others gives 19.0% (against 20.6% for the best single model). The lowest token confidence flags utterances that contain any L3 error at an AUROC of 0.71–0.76 (the Parakeet and Nemotron models).
- Paraphasias are too rare in this mostly mild sample for L3 against L4 to say much.

Caveats: six speakers, mostly mild to moderate; the intervals are utterance bootstraps that ignore speaker clustering; the GGUF ports were not re-validated here; utterance-level word error on short clips is not directly comparable to published AphasiaBank numbers.

## A language model over ten recognizers (`llm_correct.py`)

Model: Qwen 3.8 27B on Cerebras (`qwen-3.8-27b`, OpenAI-compatible API; key read from `CEREBRAS_API_KEY` in `--env`, default `.env`). Each request carries one utterance's 10 recognizer transcripts (shuffled, unlabelled) plus the speaker's previous 3 utterances as heard by Whisper large-v3. It never sees the reference transcripts. `fuse` asks for the clean transcript (drop fillers, fragments and abandoned or repeated words; keep the speaker's actual words, even wrong ones); `select` asks which transcript is best. No reasoning, temperature 0. Requests are sequential at about one a second. Cerebras's regular API doesn't retain inputs or outputs.

| system | L0 | L3 clean | L4 intended | 1–2 word utterances | invented runs | language-model latency |
|---|---|---|---|---|---|---|
| llm-fuse | 29.4 | **17.1** (15.6–18.6) | 16.5 | 36.1 | 0 | 156 ms median |
| llm-select | 29.2 | 18.2 (16.7–19.7) | 17.7 | 35.6 | 0 | 153 ms median |
| best single recognizer (whisper-large-v3) | 30.6 | 20.6 (18.9–22.5) | 20.2 | 42.7 | 1 | – |

- Latency is the language-model request alone, once the ten transcripts exist; recognition time is not included.
- Fuse beats the best single model for every one of the six speakers; the biggest gains are for the most severe speaker (30.8 → 23.9) and speaker 1738 (28.8 → 22.9).
- Per utterance against Whisper large-v3: better on 164, the same on 846, worse on 70. Fuse mostly acts as a smart merger: it ties the best available transcript 78% of the time and beats all of them only 9 times.
- It does not "correct" the speaker more than the recognizers do (L3 minus L4: +0.66 points, against +0.2 to +0.6).
- Both runs together used about 710,000 tokens.

## Rewriting a single transcript (`llm_rewrite.py`)

The language model sees one system's transcript, the same system's transcript of the speaker's previous 3 utterances, and a general description of aphasic speech errors and of how recognizers fail on them. No references, no other systems. `said` asks for what was said (scored at L3), `meant` for what was meant (L4). `run_alternatives.py` re-decodes Parakeet Ultra (ONNX fp32; transcripts identical to the GGUF run on 1,067 of 1,080) and records the top 4 tokens at every step for the `said-alts` variant.

| variant | source | reference | before | after | changes better / worse | broke a correct one | latency |
|---|---|---|---|---|---|---|---|
| said | whisper-large-v3 | L3 | 20.6 | 20.1 | 28 / 23 | 11 | 0.17 s |
| meant | whisper-large-v3 | L4 | 20.2 | **19.3** | 44 / 29 | 13 | 0.17 s |
| meant, reasoning low | whisper-large-v3 | L4 | 20.2 | 19.7 | 81 / 92 | 43 | 0.42 s (p95 0.97) |
| said | parakeet-ultra | L3 | 30.0 | 20.8 | 259 / 25 | 10 | |
| said + low-confidence marks | parakeet-ultra | L3 | 30.0 | 20.8 | 263 / 27 | 11 | |
| said + alternative tokens | parakeet-ultra (ONNX) | L3 | 30.0 | 20.7 | 271 / 30 | 10 | |

- On a good transcript the gain is small. With reasoning the model changed far more transcripts, 81 for the better and 92 for the worse (44 and 29 without), so overall it did worse: its new readings are plausible, fluent and often wrong.
- On Parakeet's verbatim output the big drop is mostly the removal of repetitions and false starts.
- Neither confidence marks nor one model's alternative tokens helped (alternatives against none: 59 better, 56 worse). The alternatives are mostly sub-word pieces, and absent exactly where the model is confidently wrong.
- In one case we looked at by hand, the right reading was recoverable only from the disagreement between several recognizers, and only a strong reasoning model, not this one, used it. That is the case for the ten-recognizer layer above.

## A memory of confirmed corrections (`correction_replay.py`, Nov 9)

The 1,080 utterances replayed in order. The language model writes what the speaker meant from the 10 transcripts and 3 utterances of context, with and without a memory of up to 25 earlier confirmed corrections for the same speaker (what two recognizers heard, and what was meant). 2,059 calls. Full table: [`results/correction_replay.md`](results/correction_replay.md).

- Intended-word error: 15.4% without the memory, 15.9% with it. Intended target words recovered: 54 of 107 without, 57 with.
- With the memory in use, on utterances without a coded error: 23 better, 28 worse, and 14 correct ones broken.
- Reading: a word-level memory of corrections did not help this model on these speakers; the differences are within noise and lean slightly negative. No significance test was run. Not tested: sound-level input, a stronger model, explicit rule extraction, dysarthric speech.

## Would a word list help? (`paraphasia_consistency.py`, Nov 8)

Counts only, never transcript text. Among the speakers' coded paraphasias, a list of (produced, intended) pairs made from the first half of each conversation would have caught 2 of 89 paraphasias in the second half. A list learned during the conversation, adding each pair as a correction confirms it, would have caught 42 of all 177 (a paraphasia counts as caught when the same pair was confirmed earlier), mostly in two speakers. So most aphasic word errors don't repeat: learn from corrections during a session, and use set-up time for names and practice targets rather than a list of past mistakes.

## Files

| script | what it does |
|---|---|
| `fetch_aprocsa.sh` | download the six recordings and transcripts; convert to 16 kHz mono WAV |
| `prepare.py` | CHAT transcripts to utterances with L0–L4 references, timing checks |
| `align_offsets.py` | check transcript timing against recognizer word times |
| `run_asr.py`, `run_lane.sh` | transcribe every utterance with each model (CPU, resumable) |
| `run_alternatives.py` | Parakeet Ultra (ONNX) with the top tokens at every step |
| `run_phones.py` | a sound-level (phoneme) recognizer; started, not finished |
| `score.py` | word error by level, speaker, length and error type; headroom; confidence AUROC |
| `export_llm_inputs.py` | per utterance, all transcripts and token confidences for the language-model layers |
| `llm_correct.py` | the ten-recognizer layer (`fuse`, `select`) |
| `llm_rewrite.py`, `rewrite_report.py` | single-transcript rewrite and its report |
| `correction_replay.py` | the correction-memory replay |
| `paraphasia_consistency.py` | the word-list check (counts only) |
