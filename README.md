# Speech recognition for people with speech disorders

After a stroke, someone may know exactly what they want to say and still lose the word on the way out. Speech recognizers miss far more of what they say, and sometimes write the word they expected instead of the one that was spoken.

[Etherea](https://withetherea.com) turns spoken words into live video, so it has to understand everyone who speaks to it. We are starting with the people who need the most help communicating. This repository holds our code, our methods and our results so far. It is open source, and it contains no recordings or transcripts: the datasets we use don't allow redistribution, so you get them from their owners (see [Data](#data)).

Etherea is made by General Semantics, a creative research lab in New York. Most of these experiments were run by AI coding agents working under our direction. Every number below comes from the scripts in this repository.

## What we've found so far

These results come from six people with chronic aphasia after a stroke: 1,080 utterances, 70 minutes of conversation, from the open APROCSA recordings. Six speakers make a pilot, so read the intervals as well as the headline numbers.

**1. The best open recognizer gets about one word in five wrong.** We ran ten open speech recognizers on the same utterances. Whisper large-v3 did best against a clean transcript, at 20.6% word error (95% interval 18.9–22.5). The top five sit within each other's intervals, at roughly 20–23%. Error rose with severity, from about 10–16% for the mildest speaker to 30–37% for the most severe.

**2. Which recognizer wins depends on what you count.** Some recognizers write down repetitions and false starts (Parakeet, Canary-Qwen); others tidy them away (Whisper, Cohere Transcribe, Granite Speech, Canary-1B). Against a word-for-word transcript, Parakeet Ultra is best at 19.5%. Against a clean one, the same model scores 30.0%. A benchmark that doesn't say which kind of transcript it uses can't be compared with another.

**3. Short answers are the hardest.** One- and two-word utterances came out 42–49% wrong for every recognizer, against 17–28% for utterances of seven words or more. For Whisper large-v3, utterances that contained a word error by the speaker (a paraphasia) scored 32.6%, against 18.6% without one.

**4. The recognizer often hides the speaker's own error.** When a speaker produced the wrong word, the word they meant appeared in Whisper large-v3's transcript for 73 of 192 such errors (38%), and the word they actually produced for 41 of 193 (21%). The transcript looks right when it isn't. This is a first estimate: it checks whether each word appears anywhere in the transcript rather than aligning words one by one, on held-out clips from the personalization runs. For anything that helps people practise speech, this is the number that matters most, and we know of no earlier measurement of it on aphasic speech.

**5. Ten recognizers are better than one.** A fast language model (Qwen 3.8 27B on Cerebras) read all ten transcripts of each utterance, plus the speaker's previous three, and wrote one clean transcript. Error fell from 20.6% to 17.1% (15.6–18.6), for all six speakers, and the language-model step took a median of 156 ms once the ten transcripts existed. The largest gain was for the most severe speaker, from 30.8% to 23.9%. Picking the best of the ten transcripts every time would reach 11.5%, so there is more to recover.

**6. Confidence scores only partly flag mistakes.** In the four recognizers that report a confidence for each word piece (the Parakeet and Nemotron models), the lowest confidence in an utterance picks out utterances containing any recognition error at an AUROC of 0.71–0.76, where 0.5 is chance.

**7. Learning the speaker helped a little.** We retuned Whisper large-v3 separately for each speaker on their own confirmed utterances, adding one at a time, as a clinician's confirmations would arrive in a session. Pooled over the six speakers, error went from 20.7% to 18.6% after 30 utterances each (−2.1 points, interval −3.6 to −0.7, resampling clips rather than speakers) and didn't move before about ten. It didn't change how often the recognizer hides an error. In aphasia the wrong words tend to change from one attempt to the next, which likely leaves a per-speaker model little to learn. We expect more in dysarthria, where a speaker's errors are more consistent.

**8. What didn't help, on aphasia:**
- Showing an audio language model a few of the speaker's own recordings with their transcripts, with no training (Phi-4 Multimodal, after Roll et al. 2025). It sometimes answered with one of the example sentences: 35.1% with no examples, 38.3% with six.
- Giving the language model in (5) a memory of up to 25 earlier confirmed corrections for the same speaker: 15.4% without it, 15.9% with it, scored against what the speaker meant.
- Asking a language model to rewrite a single good transcript. It changed little (20.2% to 19.3%, against what the speaker meant). Letting it reason made it change far more transcripts, 81 for the better and 92 for the worse (against 44 and 29), so overall it did worse.
- A word list built before a session. Most word errors don't repeat: a list made from the first half of each conversation would have caught 2 of 89 later errors. A list learned from corrections during the session would have caught 42 of 177, mostly in two speakers.

Details, tables and the scripts behind each number are in [`aphasia-benchmark/`](aphasia-benchmark/) (findings 1–6, the correction memory, the rewrite and the word list) and [`personalization/`](personalization/) (finding 7 and the in-context examples).

## What's next

Our [research agenda](RESEARCH-AGENDA.md) ranks twelve open problems in recognition for people with speech disorders. The short version:

- **A recognizer that tells the truth about what was said.** We want to catch the moment a recognizer writes the word it expected instead of the word it heard, by reading the model's internal activity. In 700 sentences synthesized with text-to-speech, where the context pointed to the wrong word, Whisper wrote the expected word 21.8% of the time, and patching its internal activity restored the spoken word in 88.9% of those cases ([Glazer et al. 2025](https://arxiv.org/abs/2508.15882)). We found no study that tries this on disordered speech. Our bar, set in advance: on real errors from held-out speakers, the detector has to clearly beat the best signal available from outside the model, measured on the same errors.
- **"I didn't catch that."** Confidence that means what it says on disordered speech, so a system asks instead of guessing. We found no published measurement of calibration on disordered speech.
- **Correction in real time.** Several recognizers and a fast language model, inside the latency of a live conversation, returning both what was said and what was meant.
- **Dysarthria and more speakers.** Six speakers is a pilot. Next come dysarthric speech, children's speech and larger multi-speaker datasets.

## Data

We use datasets under their owners' terms and publish aggregate numbers only: no audio, transcripts, recognizer outputs or example utterances.

**APROCSA** (aphasia; 6 openly shared participants, from [langneurosci.org/aprocsa-dataset](https://langneurosci.org/aprocsa-dataset/); corpus page at [TalkBank](https://talkbank.org/aphasia/access/English/Protocol/APROCSA.html), [doi:10.21415/KT40-EA41](https://doi.org/10.21415/KT40-EA41)). The corpus's usage restrictions: "These materials may be used only for research and education purposes. Further dissemination of these materials is not permitted." Any use must cite:

- Ezzes, Z., Schneck, S. M., Casilio, M., Mefford, A., de Riesthal, M., & Wilson, S. M. (2022). An open dataset of connected speech in aphasia with consensus ratings of auditory-perceptual features. *Data, 7*(11), 148.
- Casilio, M., Dutta, M., Bryan, K., Crouch, K., Dewall, Z., Feld, I., Kile, J., McMahan, K., Samora, J., Shibata, K., Steel, S., Kasdan, A. V., Walljasper, L., de Riesthal, M., & Wilson, S. M. (2026). A clinical training program for auditory-perceptual rating of connected speech in aphasia. *American Journal of Speech-Language Pathology, 35*(1), 52–68.

`aphasia-benchmark/scripts/fetch_aprocsa.sh` downloads the six recordings and transcripts from the authors' site.

## What's here

| Folder | What it holds |
|---|---|
| [`aphasia-benchmark/`](aphasia-benchmark/) | Ten open recognizers on aphasic speech, scored against five kinds of reference transcript; the language-model layers that read several transcripts, the correction memory, the rewrite and the word-list check. |
| [`personalization/`](personalization/) | Adapting a recognizer to one speaker: retuning Whisper on confirmed utterances as they arrive, and in-context examples with Phi-4 Multimodal. |
| [`RESEARCH-AGENDA.md`](RESEARCH-AGENDA.md) | The open problems, how we ranked them, and what we'd measure first. |

Each folder's README says how to run its scripts. The scripts are research code, kept as they were run (on a shared CPU, or on single rented GPUs whose disks were deleted afterwards), with local paths replaced by settings.

## Licence

Code and text in this repository: MIT (see [LICENSE](LICENSE)). The datasets are not included and keep their own terms.

## Contact

Open an issue here, or write to us through [withetherea.com](https://withetherea.com). We'd especially like to hear from clinicians, researchers, and people with speech disorders who want recognition to work for them.
