#!/bin/bash
# As run on a rented single-tenant GPU: scripts and manifests in /root, recordings deleted with the machine.
# TORGO dysarthric speakers: no examples, own examples (6, 12), another speaker's (6, 12), own with a stricter final instruction.
cd /root
export PYTHONHASHSEED=0
COMMON="--source manifest:/root/torgo_manifest.jsonl --trials 100000 --shots 6,12 --conds zero,same,diff,samev --min-test-dur 0.3 --min-ex-dur 1.0 --max-ex-dur 15 --frame one_speaker --max-new-tokens 120"
python icl_asr.py $COMMON --only-speakers F03,M05 --out tg_a.jsonl > tg_a.log 2>&1 &
python icl_asr.py $COMMON --only-speakers F04,M03,F01 --out tg_b.jsonl > tg_b.log 2>&1 &
python icl_asr.py $COMMON --only-speakers M01,M02,M04 --out tg_c.jsonl > tg_c.log 2>&1 &
wait
cat tg_a.jsonl tg_b.jsonl tg_c.jsonl > tg_all.jsonl
python whisper_baseline.py /root/torgo_manifest.jsonl tg_whisper.jsonl > tg_whisper.log 2>&1
echo ALLDONE
