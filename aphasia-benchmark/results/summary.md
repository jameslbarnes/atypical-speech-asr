# ASR on aphasic speech: APROCSA open subset (6 speakers)

1080 participant utterances, 70.0 min of audio, scored identically for every model. Reference words: L0 7581, L1 6822, L2 6541, L3 5774, L4 5776.

## WER by reference level (sorted by L3 clean)

| model | L0 | L1 | L2 | L3 | L4 | L3 95% CI | RTF (CPU) |
|---|---|---|---|---|---|---|---|
| llm-fuse | 29.4 | 22.2 | 19.3 | 17.1 | 16.5 | 15.6-18.6 | API 156 ms median |
| llm-select | 29.2 | 22.1 | 19.1 | 18.2 | 17.7 | 16.7-19.7 | API 153 ms median |
| whisper-large-v3 | 30.6 | 23.8 | 21.1 | 20.6 | 20.2 | 18.9-22.5 | 1.18 |
| cohere-transcribe-2b | 29.6 | 23.0 | 20.3 | 21.3 | 20.7 | 19.8-23.0 | 0.24 |
| whisper-large-v3-turbo | 29.9 | 23.5 | 20.7 | 21.6 | 21.2 | 19.9-23.3 | 0.97 |
| granite-speech-4.1-2b | 30.5 | 24.0 | 21.5 | 22.1 | 21.7 | 20.5-23.9 | 0.54 |
| canary-1b-v2 | 30.4 | 24.2 | 21.5 | 22.7 | 22.4 | 21.0-24.6 | 0.14 |
| nemotron-streaming-0.6b | 37.4 | 30.7 | 27.9 | 25.1 | 25.0 | 23.3-27.0 | 0.16 |
| canary-qwen-2.5b | 20.8 | 18.9 | 17.6 | 29.1 | 28.7 | 27.2-31.3 | 0.36 |
| parakeet-tdt-0.6b-v2 | 20.2 | 19.0 | 19.0 | 29.4 | 29.1 | 27.4-31.5 | 0.08 |
| parakeet-ultra-0.6b | 19.5 | 18.4 | 18.3 | 30.0 | 29.7 | 28.1-32.3 | 0.06 |
| parakeet-tdt-0.6b-v3 | 20.2 | 19.4 | 19.8 | 31.5 | 31.3 | 29.4-33.8 | 0.08 |

Levels: L0 = verbatim (fillers, fragments, repetitions); L1 = no fillers; L2 = no fillers/fragments; L3 = clean (no retraced/repeated words); L4 = intended (paraphasias -> targets). CI = utterance bootstrap (ignores speaker clustering). RTF was measured on a shared CPU at idle priority next to a live workload, so it is only a rough relative guide.

## Error types

Substitutions/deletions/insertions are shares of L3 reference words. Insertions against L3 include repetitions a verbatim model faithfully writes down, so invented text is measured against L0 instead: 'extra words vs L0' is the insertion rate against the verbatim reference, and 'invented runs' counts utterances with 5+ words that are not in even the verbatim reference (a hallucination proxy).

| model | subs (L3) | dels (L3) | ins (L3) | extra words vs L0 | invented runs (5+ words) | empty outputs |
|---|---|---|---|---|---|---|
| llm-fuse | 6.4 | 3.6 | 7.2 | 1.2 | 0 | 0 |
| llm-select | 6.3 | 4.2 | 7.6 | 1.2 | 0 | 1 |
| whisper-large-v3 | 8.0 | 4.2 | 8.5 | 1.5 | 1 | 0 |
| cohere-transcribe-2b | 8.5 | 3.8 | 9.1 | 1.3 | 0 | 0 |
| whisper-large-v3-turbo | 8.0 | 4.1 | 9.5 | 1.8 | 2 | 0 |
| granite-speech-4.1-2b | 8.2 | 4.3 | 9.6 | 1.7 | 0 | 4 |
| canary-1b-v2 | 9.0 | 4.1 | 9.7 | 1.7 | 1 | 2 |
| nemotron-streaming-0.6b | 8.2 | 11.4 | 5.4 | 1.0 | 0 | 104 |
| canary-qwen-2.5b | 8.4 | 3.5 | 17.2 | 2.2 | 0 | 0 |
| parakeet-tdt-0.6b-v2 | 7.8 | 4.0 | 17.6 | 3.1 | 3 | 5 |
| parakeet-ultra-0.6b | 8.3 | 3.8 | 17.9 | 2.5 | 0 | 4 |
| parakeet-tdt-0.6b-v3 | 8.1 | 4.4 | 19.0 | 3.5 | 5 | 6 |

## L3 WER by speaker (ordered from mildest to most severe aphasia, by the dataset's Quick Aphasia Battery scores)

| model | 1713 | 1554 | 1738 | 1944 | 1833 | 1731 |
|---|---|---|---|---|---|---|
| llm-fuse | 8.9 | 13.4 | 22.9 | 15.8 | 19.3 | 23.9 |
| llm-select | 10.3 | 15.2 | 23.3 | 16.8 | 20.0 | 24.8 |
| whisper-large-v3 | 9.8 | 16.7 | 28.8 | 18.6 | 21.7 | 30.8 |
| cohere-transcribe-2b | 10.8 | 17.1 | 26.8 | 22.0 | 21.1 | 31.3 |
| whisper-large-v3-turbo | 12.1 | 16.5 | 28.6 | 20.5 | 22.9 | 29.9 |
| granite-speech-4.1-2b | 12.8 | 17.7 | 28.6 | 21.1 | 22.2 | 32.0 |
| canary-1b-v2 | 11.8 | 18.1 | 28.5 | 23.7 | 22.8 | 32.3 |
| nemotron-streaming-0.6b | 12.3 | 17.1 | 36.8 | 23.6 | 24.9 | 37.3 |
| canary-qwen-2.5b | 14.7 | 19.3 | 31.7 | 40.5 | 30.7 | 30.2 |
| parakeet-tdt-0.6b-v2 | 14.6 | 22.8 | 32.1 | 40.1 | 30.5 | 31.2 |
| parakeet-ultra-0.6b | 14.5 | 22.2 | 33.3 | 41.7 | 31.6 | 30.3 |
| parakeet-tdt-0.6b-v3 | 16.2 | 30.5 | 34.4 | 42.0 | 31.7 | 31.9 |

## L3 WER by utterance length (reference words)

| model | 1-2 words (n=294) | 3-6 words (n=445) | 7+ words (n=327) |
|---|---|---|---|
| llm-fuse | 36.1 | 18.1 | 14.2 |
| llm-select | 35.6 | 19.8 | 14.9 |
| whisper-large-v3 | 42.7 | 22.6 | 16.7 |
| cohere-transcribe-2b | 44.1 | 23.2 | 17.2 |
| whisper-large-v3-turbo | 42.5 | 23.4 | 17.9 |
| granite-speech-4.1-2b | 45.0 | 24.0 | 18.1 |
| canary-1b-v2 | 47.4 | 25.2 | 18.1 |
| nemotron-streaming-0.6b | 49.3 | 26.3 | 21.3 |
| canary-qwen-2.5b | 48.1 | 31.7 | 25.1 |
| parakeet-tdt-0.6b-v2 | 47.2 | 31.1 | 26.2 |
| parakeet-ultra-0.6b | 47.6 | 31.3 | 27.0 |
| parakeet-tdt-0.6b-v3 | 48.1 | 33.2 | 28.4 |

## Headroom for an LLM layer

- Best single ASR model at L3: whisper-large-v3 20.6%. Oracle that picks the best of the 10 ASR hypotheses per utterance: 11.5%. The gap is the most a hypothesis-selection layer could recover (a generative layer could in principle go further, or invent).
- Label-free consensus pick (the hypothesis that agrees most with the others) at L3: 19.0% vs best single ASR model at L3: 20.6%.
- Label-free consensus pick (the hypothesis that agrees most with the others) at L0: 28.7% vs best single ASR model at L0: 19.5%.
- Utterances with coded paraphasias: 117. L3 WER with vs without paraphasias (whisper-large-v3): 32.6% vs 18.6%.
- L3 minus L4 (positive = model output is closer to the *intended* words than to what was said):
  - llm-fuse: +0.66 pts
  - llm-select: +0.47 pts
  - whisper-large-v3: +0.44 pts
  - cohere-transcribe-2b: +0.60 pts
  - whisper-large-v3-turbo: +0.39 pts
  - granite-speech-4.1-2b: +0.39 pts
  - canary-1b-v2: +0.35 pts
  - nemotron-streaming-0.6b: +0.13 pts
  - canary-qwen-2.5b: +0.37 pts
  - parakeet-tdt-0.6b-v2: +0.37 pts
  - parakeet-ultra-0.6b: +0.29 pts
  - parakeet-tdt-0.6b-v3: +0.20 pts
- Does the lowest token confidence flag utterances with errors? (AUROC, 0.5 = chance)
  - nemotron-streaming-0.6b: AUROC 0.71 (n=965)
  - parakeet-tdt-0.6b-v2: AUROC 0.76 (n=1075)
  - parakeet-ultra-0.6b: AUROC 0.73 (n=1076)
  - parakeet-tdt-0.6b-v3: AUROC 0.76 (n=1074)
