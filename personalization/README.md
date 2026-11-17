# Personalization: adapting a recognizer to one speaker

Can a recognizer get better for one person from a handful of their own utterances? We tried two ways on the six APROCSA aphasia speakers (see the [top-level README](../README.md#data) for the data and its terms), Nov 14–15, 2026. Every run happened on a single rented GPU, and the clips were deleted with the machine. Only aggregate scores are in `results/`.

## 1. Retuning Whisper as confirmations arrive (`whisper_loop.py`)

Picture a session: the person speaks, a clinician or partner confirms what was said, and the recognizer learns from each confirmed utterance. We simulate that with no extra recording. The confirmed utterance itself is the training example. Utterances arrive one at a time in a random order. After 1, 3, 5, 10, 20 and 30 of them, Whisper large-v3 is retuned from its original weights on all of them so far (LoRA rank 16 on attention, learning rate 1e-4, 8 steps per utterance, clamped to 20–120), then scored on held-out clips that share no text with the 30 (up to 150 per speaker).

Scored against the clean transcript (L3), six speakers, one arrival order ([`results/session_loop_aphasia.txt`](results/session_loop_aphasia.txt)):

| confirmed utterances | word error | change (95% interval) |
|---|---|---|
| 0 | 20.7% | |
| 1, 3, 5 | 21.0%, 21.3%, 21.0% | none |
| 10 | 19.7% | −1.0 (−2.2 to +0.1) |
| 20 | 19.3% | −1.4 (−2.8 to −0.1) |
| 30 | 18.6% | −2.1 (−3.6 to −0.7) |
| 30, trained on the intended words | 18.1% | −2.6 (−4.0 to −1.3) |

- The gain is small and appears only from about ten utterances. Against the verbatim transcript there is none.
- **Wrong confirmations** (20% of labels replaced by the recognizer's own untuned output) did little harm: 18.9% at 30.
- **No human at all** (trained on its own output): 19.8% at 30, about 40% of the gain with confirmations.
- **Hidden errors** ([`results/session_loop_aphasia_error_words.txt`](results/session_loop_aphasia_error_words.txt)): on the held-out clips with a coded word error, the intended word appears in the untuned recognizer's transcript for 73 of 192 intended words the speakers didn't say (38%), and the produced word for 41 of 193 (21%). Tuning on what was said leaves that at 37% and 20%; tuning on what was meant moves it to 41% and 20%. Whisper hides errors before any personalization, and 30 utterances of tuning barely change that either way. `score_kept.py` checks whether each word appears anywhere in the transcript rather than aligning words one by one, so read this as a first estimate.
- **A published recipe for comparison.** EvolveCaptions (ASSETS 2025) retunes Whisper base for deaf and hard-of-hearing speakers from their corrections. Its training settings (full fine-tune, learning rate 1e-5, batch 8, 100 steps) applied to our loop: 32.2% untuned to 26.5% at 30 ([`results/session_loop_aphasia_published_recipe.txt`](results/session_loop_aphasia_published_recipe.txt), two arrival orders). A larger drop, from a much worse start; Whisper large-v3 with the loop above ends far lower.

Limits: one arrival order for the full comparison, a random order rather than a real session, one training run per order, six speakers, and intervals that resample clips rather than speakers.

## 2. Examples in the prompt, no training (`icl_asr.py`)

Roll et al. 2025 ([arXiv 2505.14887](https://arxiv.org/abs/2505.14887)) adapt Phi-4 Multimodal to a speaker by putting a few of the speaker's clips and transcripts in the prompt before the clip to transcribe. We first reproduced their result on the corpus they used (typical accented speech), then ran the same method on the aphasia speakers: 1,079 clips, examples transcribed verbatim, scored with the benchmark's normalizer ([`results/in_context_examples_aphasia.txt`](results/in_context_examples_aphasia.txt)).

| condition | clean transcript (L3) | verbatim (L0) | clips where the output copied an example |
|---|---|---|---|
| no examples | 35.1% | 28.1% | 0 |
| 6 of the speaker's own | 38.3% | 35.8% | 57 |
| 12 of the speaker's own | 43.6% | 41.5% | 60 |
| 6 of another speaker's | 62.9% | 57.6% | 159 |
| 12 of another speaker's | 70.1% | 63.6% | 149 |

- **It failed the bar we set in advance.** The speaker's own examples beat another speaker's by a wide margin, but not giving no examples at all.
- **Why:** the model sometimes answers with one of the example sentences instead of transcribing the new clip ("copied" means the output shares a four-word run with an example transcript that is not in the reference). It never happens without examples, and it is worst on one- and two-word clips.
- **A guard that needs no reference** (if the output shares a four-word run with an example, use the no-example output): 34.1% with six own examples against 35.1% with none, lower for five of six speakers, but worse against the verbatim transcript ([`results/in_context_examples_aphasia_guard.txt`](results/in_context_examples_aphasia_guard.txt)).
- **The model starts far behind Whisper** on this data: 35.1% with no examples, against 20.6% for Whisper large-v3 on nearly the same clips.

Our reading: the prompting recipe we tested did not lower overall error without the guard. A model trained to use examples would likely do better, and training one takes recordings from many speakers.

## Files

| script | what it does |
|---|---|
| `make_aphasia_manifest.py` | cut the scoreable APROCSA utterances into clips and write manifests (into `$APROCSA_DIR/derived/icl`, never a repo) |
| `whisper_loop.py` | the session loop: retune after each confirmed utterance, with the wrong-label, no-human and intended-word arms |
| `score_loop.py`, `score_kept.py` | score the loop; count errors written as said or as intended |
| `icl_asr.py`, `run_aphasia.sh` | in-context examples with Phi-4 Multimodal |
| `score_aphasia.py`, `score_guard.py`, `score_icl.py` | score the in-context runs, with and without the copy guard |
| `whisper_baseline.py` | Whisper large-v3 on a manifest's test clips, as a reference point |

The scorers import the benchmark's normalizer from `../aphasia-benchmark/scripts`.
