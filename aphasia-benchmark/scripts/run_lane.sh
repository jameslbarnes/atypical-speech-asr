#!/usr/bin/env bash
# Run several models back to back (one "lane"). Usage: run_lane.sh <threads> <model> [<model> ...]
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-data/aprocsa/derived}
# transcribe.cpp v0.4.0 native library (transcribe.dll / libtranscribe.so); set it before running.
: "${TRANSCRIBE_LIBRARY:?set TRANSCRIBE_LIBRARY to the transcribe.cpp library}"
export TRANSCRIBE_LIBRARY
threads=$1; shift
for m in "$@"; do
  echo "$(date +%T) start $m"
  "$HERE/.venv/Scripts/python.exe" -I -X utf8 "$HERE/scripts/run_asr.py" "$m" "$DATA/utterances.jsonl" "$DATA/hyps" --threads "$threads" \
    > "$DATA/logs/$m.log" 2> "$DATA/logs/$m.stderr.log"
  status=$?
  echo "$(date +%T) end $m (exit $status) $(tail -1 "$DATA/logs/$m.log")"
done
echo LANE_DONE
