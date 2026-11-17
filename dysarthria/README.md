# Dysarthria: learning one speaker, live recognition and word lists

Dysarthria makes speech slurred, slow or effortful because the muscles that produce it are weak or poorly controlled, for example after cerebral palsy, ALS, Parkinson's disease or a stroke. Unlike the word errors of aphasia, a dysarthric speaker's errors are fairly consistent from one attempt to the next, so a recognizer that learns the speaker should gain more.

All results here use the TORGO database ([University of Toronto](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)): 8 speakers with dysarthria (cerebral palsy or ALS) and 7 control speakers, reading single words and sentences from prompts. Read speech with known prompts is easier than conversation, and eight speakers make a small sample. The runs took place Nov 14–16, 2026, each on a single rented GPU whose copy of the recordings was deleted with the machine. Only aggregate scores are in `results/`. Cite TORGO with any use:

- Rudzicz, F., Namasivayam, A. K., & Wolff, T. (2012). The TORGO database of acoustic and articulatory speech from speakers with dysarthria. *Language Resources and Evaluation, 46*(4), 523–541.
- Rudzicz, F., Hirst, G., & van Lieshout, P. (2012). Vocal tract representation in the recognition of cerebral palsied speech. *Journal of Speech, Language, and Hearing Research, 55*(4), 1190–1207.

Word error rates below are scored with the aphasia benchmark's clean normalizer (the Whisper English normalizer), on held-out clips that share no text with any training sentence. Intervals are 95% bootstraps over clips, or in the session loop over clip runs (each clip scored once per arrival order), not over speakers, so they may understate the uncertainty. The loop's learning rate was chosen on the earlier ten-sentence test with the same speakers, so treat these as exploratory results rather than a clean held-out test.

## 1. Ten of a speaker's sentences (Whisper large-v3 with LoRA)

Rank 16 on the attention query and value projections, encoder and decoder; batch 2; 80 steps for ten sentences and 120 for thirty; one seed; two learning rates fixed in advance and both reported in [`results/few_sentence_tuning_whisper.txt`](results/few_sentence_tuning_whisper.txt).

| version (learning rate 1e-4) | sentences, 8 speakers (407 clips) | sentences, 7 speakers (400 clips) |
|---|---|---|
| Whisper, untuned | 27.1% | 26.5% |
| 10 of the speaker's own sentences | 21.6% | 21.1% |
| 30 of the speaker's own sentences | n/a | 18.9% |
| 10 of another speaker's sentences | 25.1% | 24.6% |

- Ten own sentences: −5.5 points against untuned (−7.7 to −3.4), better for 6 of 8 speakers, and −3.6 against ten of another speaker's sentences. Part of the gain (−2.0) comes from getting used to the recordings and reading material rather than the speaker.
- The gain is in the speakers the recognizer understood worst. Untuned, ten sentences, thirty sentences: M01 58%, 39%, 33%; M02 46%, 39%, 30%; M04 67%, 57%, 48%. The two speakers with the lowest error stay at 1–2%.
- The higher learning rate (1e-3) overfits with ten sentences and gains little.

## 2. The session loop: retuning as confirmations arrive

The same loop as for aphasia ([`personalization/`](../personalization/)): confirmed sentences arrive one at a time in a random order, and after 1, 3, 5, 10, 20 and 30 of them the recognizer is retuned from its original weights on all of them so far. Seven speakers had enough sentences; three arrival orders ([`results/session_loop_three_orders.txt`](results/session_loop_three_orders.txt)).

| confirmed sentences | word error | change |
|---|---|---|
| 0 | 27.6% | |
| 1 | 25.6% | −2.0 (−3.2 to −0.9) |
| 3 | 23.7% | −3.9 (−5.2 to −2.8) |
| 5 | 22.2% | −5.4 (−7.4 to −3.6) |
| 10 | 20.6% | −7.0 (−9.2 to −5.1) |
| 20 | 19.0% | −8.5 (−10.8 to −6.6) |
| 30 | 18.5% | −9.1 (−11.3 to −7.2) |

- The three arrival orders agree within two points at every size.
- The three speakers with the highest starting error, at 30 sentences: M01 57% to 31%, M02 41% to 29%, M04 81% to 47%.
- Two more arms ran on two of the three orders ([`results/session_loop_all_arms_two_orders.txt`](results/session_loop_all_arms_two_orders.txt); on those orders the clean loop scores 27.8% untuned, 20.5% at 10 and 18.5% at 30). **Noisy labels**: replacing 20% of the labels with the recognizer's own untuned output (some of which may be right) reduced the gain only modestly, to 21.7% at 10 and 19.3% at 30. **No human at all**, training on its own output, gives 24.0% at both 10 and 30: about half the ten-sentence gain, and it doesn't grow.
- Retuning took about 10 seconds for one sentence, about a minute for ten, and about 100 seconds for twenty or thirty, on a GPU shared by three runs.
- The published EvolveCaptions recipe (Whisper base, full fine-tune) on the same loop: 64.2% untuned, 40.0% at 10, 35.7% at 30 ([`results/session_loop_published_recipe.txt`](results/session_loop_published_recipe.txt)).

On aphasia, the same loop gained 2.1 points after 30 utterances.

## 3. Live recognition: Confucius4-R2T2 against Nemotron

Untuned, on the same 2,360 clips ([`results/untuned_comparison.txt`](results/untuned_comparison.txt)):

| | Nemotron 3.5 streaming (320 ms lookahead, language auto-detected) | Confucius4-R2T2, live (160 ms chunks) | Confucius4-R2T2, whole clips | Whisper large-v3, whole clips |
|---|---|---|---|---|
| control speakers, sentences | 6.5% | 2.5% | 1.7% | 1.9% |
| control speakers, single words | 46.8% | 23.3% | 23.3% | 29.0% |
| dysarthric speakers, sentences | 50.8% | 34.6% | 27.7% | 29.1% |
| dysarthric speakers, single words | 72.2% | 55.0% | 55.4% | 92.1% |

- Nemotron returned nothing for 113 of the 714 dysarthric sentences; Confucius returned text for every one. Fixing Nemotron's language to English, or raising its lookahead to 1.12 s, helped by 1 to 4 points.
- Whisper's single-word score is high because it sometimes writes long invented passages for two speakers; it gets more single words exactly right than its error rate suggests.
- Tuning Nemotron needed two workarounds in the transformers library and gained little on two speakers, so we didn't pursue it.

**Tuning Confucius, and why the live check matters.** Tuned with the session loop on whole sentences, with an adapter on both its audio encoder and its language model, Confucius improved on whole clips: 24.3% untuned to 20.2% at 30 sentences ([`results/confucius_session_loop.txt`](results/confucius_session_loop.txt); single words got slightly worse, 50.5% to 53.9%). Run through its live streaming engine, the same tuned models got worse: 43.7% untuned, 55.3% tuned, worse for every speaker ([`results/confucius_live_tuned_both_parts.txt`](results/confucius_live_tuned_both_parts.txt)). With the adapter on the audio encoder only (the four hardest speakers, one arrival order): live 54.6% to 48.5%, whole clips 42.9% to 32.7% ([`results/confucius_live_tuned_hearing_only.txt`](results/confucius_live_tuned_hearing_only.txt)). In this experiment, adapting the audio encoder alone improved pooled live accuracy, though not for every speaker (M05 went from 31.1% to 31.6%), and single words were mixed. Our working explanation, not yet tested, is that training on whole sentences disturbs the language model's live behaviour. Either way, score a tuned streaming model through its live engine.

## 4. Word lists, and the echo

Confucius takes a text context. Live, no training, 328 sentences from 5 dysarthric speakers ([`results/confucius_vocabulary_hints.txt`](results/confucius_vocabulary_hints.txt)):

| context given | raw word error | clips where the list was written out as speech | with the guard |
|---|---|---|---|
| none | 43.7% | 0 | 43.7% |
| plain list of 40 words, including the ones said | 116.1% | 90 | 39.3% |
| "Hotwords:" and the same 40 words | 75.4% | 39 | 37.5% |
| short list of 10, mostly the words said | 62.9% | 123 | 36.7% |
| plain list of 40 wrong words | 110.1% | 68 | 44.9% |

- The guard keeps every clip: when the transcript repeats part of the list, it falls back to the same model's transcript with no list. Without the guard, every list made results worse (raw error 62.9% to 116.1%).
- Every wording makes the model write the list into its transcript on some clips. With the guard, the right words beat wrong words in the same format by 5.6 points (39.3% against 44.9%), so the list carries real information.
- Even after the guard, some transcripts gained a listed word the speaker never said. For speech practice, that is a hidden error in a new form.

**Word lists and tuning together** (live Confucius, 240 held-out sentences from the four hardest speakers; tuning is the audio-encoder adapter on 30 sentences; the guard falls back to the same model's no-list transcript when the output repeats three words of the list, and keeps every clip; [`results/confucius_hints_and_tuning.txt`](results/confucius_hints_and_tuning.txt)):

| | no list | right words, guarded | wrong words, guarded |
|---|---|---|---|
| untuned | 54.6% | 46.8% | 54.1% |
| tuned | 49.4% | 38.4% | 50.2% |

- They add up, slightly more than their sum: tuning alone −5.2 points, the list alone −7.8, both −16.2. Per speaker with both: M01 67% to 35%, M04 83% to 60%, M02 55% to 42%, M05 31% to 26%.
- With the same guard, wrong lists leave both models where they were, so right lists beat wrong lists under identical fallback. The guard itself matters: the tuned model with the right list scores 82.4% raw and 38.4% guarded.
- The right list contains the words actually spoken, which is the best case. A real list of likely words will be noisier. Four speakers, one arrival order.

## 5. Examples in the prompt (Phi-4 Multimodal)

The in-context method from [`personalization/`](../personalization/) didn't help here either ([`results/in_context_examples_phi4.txt`](results/in_context_examples_phi4.txt)). On sentences, Whisper large-v3 with no examples scored 32.2% and Phi-4 with no examples 60.4%, mostly because Phi-4 invents long passages for the two speakers with the highest error; any examples reduce that, and another speaker's reduce it as much. On the 982 clips where no version ran away, six of the speaker's own examples scored 31.2% against 31.6% with none (−0.5, −1.9 to +1.0).

## Files

| script | what it does |
|---|---|
| `make_torgo_manifest.py` | build a manifest from the TORGO speakers (head microphone when present) |
| `whisper_fewshot.py`, `score_ft.py` | ten or thirty sentences of tuning, own and another speaker's |
| `../personalization/scripts/whisper_loop.py`, `score_loop.py` | the session loop (shared with aphasia) |
| `nemotron_loop.py`, `r2t2_loop.py` | the session loop for Nemotron and for Confucius4-R2T2 |
| `eval_base.py`, `eval_r2t2.py`, `eval_r2t2_torch.py`, `score_cmp.py` | the untuned comparison; Confucius live and whole-clip (set `R2T2_DIR` to the maker's code) |
| `push_client.py` | play recordings into a live recognizer at real-time pace |
| `score_live.py`, `score_live_audio.py` | tuned Confucius through the live engine |
| `score_ctx2.py`, `score_comb.py` | word lists, with and without tuning |
| `run_torgo.sh`, `score_torgo.py` | in-context examples with Phi-4 Multimodal (`../personalization/scripts/icl_asr.py`) |

The scorers expect the per-clip outputs (not included) under `results/` and `logs/` in the working directory, as they were run, and import the benchmark's normalizer from `../aphasia-benchmark/scripts`.
