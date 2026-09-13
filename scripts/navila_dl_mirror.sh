#!/bin/bash
# Detached, proxy-free, resumable download of NaVILA R2R train.tar.gz via hf-mirror.
# Route rationale (measured 2026-09-13): huggingface.co is unreachable from
# 202.199.13.0/24 (443 timeout, IPv6 resolves into 2a03:2880::/32). hf-mirror.com
# resolves the 302 to cas-bridge.xethub.hf.co, which IS directly reachable.
# Throughput: 6.37-7.81 MB/s via mirror vs 0.27 MB/s via the SSH-tunnel proxy.
# No proxy is used, so this survives the operator's local port-forward closing.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
DEST=/mnt/wxh/go2_short_vln/downloads/navila_probe
FILE=$DEST/R2R_train.tar.gz
URL=https://hf-mirror.com/datasets/a8cheng/NaVILA-Dataset/resolve/main/R2R/train.tar.gz
EXP=19703787520
LOG=$DEST/download_mirror.log
echo "$(date -Is) === mirror downloader started pid $$ (no proxy) ===" >> "$LOG"
a=0
while :; do
  a=$((a+1))
  cur=$(stat -c%s "$FILE" 2>/dev/null || echo 0)
  if [ "$cur" -ge "$EXP" ]; then echo "$(date -Is) DONE $cur/$EXP" >> "$LOG"; break; fi
  echo "$(date -Is) try#$a resume from $cur/$EXP ($((cur*100/EXP))%)" >> "$LOG"
  t0=$(date +%s)
  curl -sSL --max-time 3000 --connect-timeout 30 --speed-time 120 --speed-limit 10240 \
       --retry 0 -C - -o "$FILE" "$URL" >>"$LOG" 2>&1
  rc=$?
  t1=$(date +%s); new=$(stat -c%s "$FILE" 2>/dev/null || echo 0)
  d=$((t1-t0)); [ $d -lt 1 ] && d=1
  echo "$(date -Is) try#$a rc=$rc got $((new-cur))B in ${d}s = $(( (new-cur)/d/1024 )) KB/s; now $new/$EXP" >> "$LOG"
  if [ "$new" -ge "$EXP" ]; then echo "$(date -Is) DONE $new/$EXP" >> "$LOG"; break; fi
  if [ "$new" -le "$cur" ]; then sleep 30; else sleep 5; fi
done
echo "$(date -Is) === hashing ($(stat -c%s "$FILE") bytes) ===" >> "$LOG"
sha256sum "$FILE" >> "$LOG" 2>&1
# NOTE (2026-09-13): despite the .tar.gz name (and HF serving it as
# content-type: application/gzip), R2R/train.tar.gz is an UNCOMPRESSED POSIX tar
# (GNU). A gzip check therefore always fails with "not in gzip format" and is not
# evidence of corruption. Verify with tar instead.
echo "$(date -Is) === tar integrity check ===" >> "$LOG"
if tar -tf "$FILE" > /dev/null 2>>"$LOG"; then
  echo "$(date -Is) tar OK: $(tar -tf "$FILE" 2>/dev/null | wc -l) entries" >> "$LOG"
else
  echo "$(date -Is) tar FAILED -- archive incomplete or corrupt" >> "$LOG"
fi
echo "$(date -Is) === mirror downloader exit ===" >> "$LOG"
