# Correction-memory replay (2026-11-09)

`scripts/correction_replay.py`, qwen-3.8-27b, reasoning none, 1,080 utterances, 6 speakers, 2,059 calls.
Base: LLM writes what the speaker meant from 10 ASR hypotheses + 3 utterances of context.
Mem: same, plus up to 25 earlier therapist-confirmed corrections (what two recognizers heard -> intended reference).

| Intended-word (L4) WER % | n | base | mem |
|---|---|---|---|
| all utterances | 1080 | 15.4 | 15.9 |
| with a coded speaker error | 117 | 22.0 | 22.6 |
| without a coded error | 963 | 14.2 | 14.8 |
| 0 earlier corrections | 79 | 16.9 | 16.9 |
| 1-5 | 366 | 14.2 | 14.9 |
| 6-15 | 331 | 15.4 | 15.8 |
| 16+ | 304 | 16.7 | 17.3 |

Per speaker (base / mem): 1713 8.7 / 9.6; 1554 13.0 / 12.8; 1944 13.8 / 14.0; 1833 18.7 / 19.2; 1738 19.1 / 20.1; 1731 20.1 / 20.6.

Intended target words recovered: all 107: base 54, mem 57. Same error seen before (10): 4 vs 6. New error with memory in use (91): 48 vs 49.
Memory in use, utterances with a coded error (111): better 5, worse 8, broke a correct one 3. Without (890): better 23, worse 28, broke 14.
Latency median 0.173 s (348 prompt tokens) vs 0.187 s (711).

Reading: utterance-level correction memory at word level did not help this model on these speakers; differences are within noise and lean slightly negative. No significance test run. Not tested: sound-level input, a stronger model, rule extraction, dysarthric speech.
