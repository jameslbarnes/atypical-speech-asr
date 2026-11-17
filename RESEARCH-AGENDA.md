# Research agenda: where recognition for people with speech disorders can move fastest

November 14, 2026. A ranking of twelve open research paths in speech recognition for people with speech disorders (aphasia, dysarthria, apraxia, stuttering, children's speech-sound disorders), and what we are doing first. The ranking is a judgment exercise, not a measurement; the reasoning is below so you can disagree with it.

## The twelve paths

| # | Path | What it is |
|---|---|---|
| P1 | Reading hidden errors from model internals | Detect from inside the model when its expectation overrides the sound; find where unusual articulation gets lost; steer it toward literal transcription without retraining |
| P2 | Said-and-meant recognition | One recognizer that returns what was said, with its errors, and what was meant, aligned |
| P3 | Calibrated confidence and safe failure | Confidence that means what it says on disordered speech; reliable detection of invented text; "I didn't catch that" instead of a wrong answer |
| P4 | Human-listener modelling | Predict what an unfamiliar human listener would hear; find when machine success matches human understanding |
| P5 | Adaptive difficulty | A validated dial for how easy the machine listener is to satisfy, plus per-person scheduling of practice items and cues |
| P6 | Turn-taking and live recognition | Not cutting people off during word-finding pauses and blocks; live, on-device recognition of disordered speech |
| P7 | Instant and lifelong personalization | Adapting to one speaker from a few examples; following a speaker whose condition changes over months |
| P8 | Context-aware and multimodal recognition | Using the scene, the conversation and the face |
| P9 | Data for the under-served | Recordings for children, rare conditions and other languages; synthetic disordered speech; training across clinics |
| P10 | Explanatory feedback | Telling the person what to change, from estimates of tongue position to corrected playback in their own voice |
| P11 | Communicative-success feedback and its clinical evidence | Showing a person what was understood beside what they meant, and the trials that would show whether it helps |
| P12 | Clinically meaningful evaluation and fairness | Standard tests that measure communication and clinical change, reported by subgroup |

## How we ranked them

Three values, each judged separately so one could not colour another:
- **Chance:** how likely is a field-changing result by the end of 2028?
- **Impact:** if it happens, how much does life change for people with speech disorders, for how many, and how directly?
- **Speed when AI agents do most of the research work:** thinking, coding and analysis get much faster; recruiting patients, ethics approval, therapy follow-up and new recordings don't.

Four AI research agents first gathered the evidence on each path. Nine AI judges (Claude Fable 5.2), three for each value, then compared all 66 pairs of paths on the usual 1-to-9 scale, with a written reason for every comparison. The weights come from the analytic hierarchy process (principal eigenvector of the combined comparisons). Each column sums to 100; an average path scores 8.3. "Overall" weights the three values equally.

| Rank | Path | Chance | Impact | Speed | Overall |
|---|---|---|---|---|---|
| 1 | Reading hidden errors from model internals | **20.7** | 2.5 | 16.7 | **13.3** |
| 2 | Calibrated confidence and safe failure | 10.2 | 4.5 | **17.3** | 10.7 |
| 3 | Said-and-meant recognition | 11.1 | 6.6 | 11.2 | 9.6 |
| 4 | Turn-taking and live recognition | 9.2 | 9.9 | 8.1 | 9.1 |
| 5 | Explanatory feedback | 8.3 | 12.2 | 3.9 | 8.1 |
| 6 | Context-aware and multimodal recognition | 5.8 | 10.1 | 8.1 | 8.0 |
| 7 | Clinically meaningful evaluation and fairness | 6.5 | 2.7 | 14.8 | 8.0 |
| 8 | Communicative-success feedback and its trials | 2.2 | **19.8** | 1.8 | 8.0 |
| 9 | Instant and lifelong personalization | 3.8 | 14.2 | 5.0 | 7.7 |
| 10 | Human-listener modelling | 11.6 | 3.9 | 5.8 | 7.1 |
| 11 | Adaptive difficulty | 7.0 | 5.3 | 4.5 | 5.6 |
| 12 | Data for the under-served | 3.6 | 8.2 | 2.7 | 4.8 |

**How firm it is.** First place held in 77% of 120 different weightings of the three values (each given at least 10%); the feedback-and-trials path was first in 22%. Places two to four are close: multiplying the values instead of adding them gives a near tie between hidden errors, said-and-meant, calibrated confidence and turn-taking. Ranks five to ten can't be separated. The judges agreed closely, but they are one model reading one body of evidence, mostly from paper abstracts, and the grouping into twelve paths is ours.

## What we take from it

**A recognizer that tells the truth about what was said is the work to do now.** The top three paths share one goal: report what was actually said, and know when you don't know. They also share one scarce input, recordings in which a person has coded every error.

**It is also what the highest-impact path is waiting for.** Showing someone what was understood beside what they meant could change how people practise speech, and it is by far the most impactful path, but it needs clinical trials that are unlikely to report before 2029. A practice picture that shows "cat" when the person said "tat" teaches the wrong thing. The number any clinician will ask for is how often the system hides an error, and the top paths produce that number and the fix. In our aphasia data, the intended word appeared in Whisper large-v3's transcript for 38% of the speakers' word errors, a first estimate (see [`personalization/`](personalization/)).

**As research labour gets cheap, work that needs only code and existing data gets done fast.** Over the next two years the scarce things are the ones agents can't make: coded recordings, data agreements, clinical partners, ethics approval and people's time. So we do the fast work now and start the slow work at the same time.

## What we're doing first

**1. Hidden errors (P1).** When the recognizer writes the target word although the speaker produced an error, can we tell from inside the model, live, and make it write what was said? In 700 sentences synthesized with text-to-speech, where the context pointed to the wrong word, Whisper wrote the expected word 21.8% of the time, and patching its internal activity restored the spoken word in 88.9% of those cases ([Glazer et al. 2025](https://arxiv.org/abs/2508.15882)). We found no study that tries this on disordered speech. The plan:
- measure how often Whisper writes the target on real coded errors, and reproduce the published result on typical speech to check our tools;
- train a detector on synthetic mispronounced words, and test it only on real coded errors from held-out speakers;
- compare it with what is visible from outside the model: token confidence, disagreement among our ten recognizers, and a recognizer that writes sounds rather than words;
- try the internal edit, and measure how often it restores the error as spoken, and what it costs in speed.

The bar, set before any run: on real errors from held-out speakers, the detector must clearly beat the best outside signal measured on the same errors. (The 0.71–0.76 we measured for flagging utterances that contain any error is a different task, so it is background, not the bar.) If it doesn't, we'll publish that.

**2. Personalization that works live (P7, P6).** In aphasia, a few of a speaker's own utterances barely helped (see [`personalization/`](personalization/)). In dysarthria, thirty confirmed sentences cut Whisper's word error by a third, and adapting only a streaming recognizer's audio encoder helped live (see [`dysarthria/`](dysarthria/)). Next: the same in conversation rather than read prompts, a growth curve for the live recognizer, and word lists that can't end up in the transcript as speech.

**3. Correction in real time (P2, P6).** Several recognizers and a fast language model, inside the delay of a live conversation, returning both what was said and what was meant, with the word-for-word transcript always kept beside the corrected one.

**4. More speakers.** Six speakers is a pilot. We want larger collections of aphasic and dysarthric speech, and children's speech, under their owners' terms.

If you work on any of this, or have speech that today's recognizers get wrong and want to help change that, please get in touch: open an issue, or write to us through [withetherea.com](https://withetherea.com).
