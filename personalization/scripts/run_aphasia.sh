#!/bin/bash
# As run on a rented single-tenant GPU: scripts and manifests in /root, clips deleted with the machine.
# Step 2: six aphasia speakers, examples transcribed verbatim (L0). Three processes share the GPU.
cd /root
COMMON="--source manifest:/root/icl/manifest_L0.jsonl --trials 400 --shots 6,12 --conds zero,same,diff --min-test-dur 0.3 --min-ex-dur 1.5 --max-ex-dur 15 --frame one_speaker --max-new-tokens 160"
python icl_asr.py $COMMON --only-speakers 1731 --out aph_a.jsonl > aph_a.log 2>&1 &
python icl_asr.py $COMMON --only-speakers 1944,1554 --out aph_b.jsonl > aph_b.log 2>&1 &
python icl_asr.py $COMMON --only-speakers 1738,1713,1833 --out aph_c.jsonl > aph_c.log 2>&1 &
wait
cat aph_a.jsonl aph_b.jsonl aph_c.jsonl > aph_L0.jsonl
# Variant: the same test clips with clean (L3) example transcripts, six same-speaker examples only
python icl_asr.py --source manifest:/root/icl/manifest_L3.jsonl --trials 400 --shots 6 --conds same --min-test-dur 0.3 --min-ex-dur 1.5 --max-ex-dur 15 --frame one_speaker --max-new-tokens 160 --out aph_L3.jsonl > aph_L3.log 2>&1
echo ALLDONE
