#!/usr/bin/env bash
# Download the open APROCSA dataset (Ezzes et al. 2022; research/education use only, do not redistribute)
# and convert the videos to 16 kHz mono WAV. Videos download in parallel: the server caps each connection at ~2 MB/s.
# Usage: fetch_aprocsa.sh [id ...]   (default: all six)
set -uo pipefail
DATA=${DATA:-data/aprocsa}
FFMPEG=${FFMPEG:-ffmpeg}
one() {
  id=$1
  [ -s "$DATA/wav/${id}.wav" ] && { echo "skip $id"; return; }
  echo "$(date +%T) downloading $id"
  curl -sS -L -A "Mozilla/5.0" --retry 5 -C - --limit-rate 8M -o "$DATA/raw/${id}.mp4" "https://lnl.app.vumc.org/aprocsa-dataset/${id}.mp4" || { echo "FAILED download $id"; return; }
  "$FFMPEG" -nostdin -hide_banner -loglevel error -y -i "$DATA/raw/${id}.mp4" -vn -ac 1 -ar 16000 -c:a pcm_s16le "$DATA/wav/${id}.tmp.wav" \
    && mv "$DATA/wav/${id}.tmp.wav" "$DATA/wav/${id}.wav" && rm -f "$DATA/raw/${id}.mp4" && echo "$(date +%T) done $id" || echo "FAILED extract $id"
}
IDS=${*:-1554 1713 1731 1738 1833 1944}
mkdir -p "$DATA/raw" "$DATA/cha" "$DATA/wav"
for id in $IDS; do
  curl -sS -L -A "Mozilla/5.0" --retry 3 -o "$DATA/cha/aprocsa${id}a.cha" "https://langneurosci.org/files/aprocsa/dataset/aprocsa${id}a.cha"
done
for id in $IDS; do one "$id" & done
wait
echo ALL_DONE
