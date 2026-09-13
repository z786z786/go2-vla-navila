#!/bin/bash
# P4-T1: extract the NaVILA R2R training frames.
#
# The archive is a plain POSIX tar despite the .tar.gz name (HF also serves it
# as content-type: application/gzip). Extract with -xf and never -z; a gzip
# check fails with "not in gzip format" and that is NOT evidence of corruption.
#
# Idempotent: re-running rewrites identical bytes to identical paths.
set -euo pipefail
ARCHIVE=/mnt/wxh/go2_short_vln/downloads/navila_probe/R2R_train.tar.gz
DEST=/mnt/wxh/go2_short_vln/data/navila_dataset/R2R
EXPECT_SHA=49ffc4f88e15ae4bdd2a8f01b16a8685deee0bd4d05711998d9ae88a99d4d228
EXPECT_BYTES=19703787520
EXPECT_JPG=601125
EXPECT_DIRS=10819

echo "[1/4] verifying input"
[ "$(stat -c%s "$ARCHIVE")" = "$EXPECT_BYTES" ] || { echo "size mismatch"; exit 1; }
[ "$(sha256sum "$ARCHIVE" | cut -d' ' -f1)" = "$EXPECT_SHA" ] || { echo "sha256 mismatch"; exit 1; }

echo "[2/4] quarantining the truncated annotations copy if present"
if [ -f "$DEST/annotations.json" ]; then
  mv -n "$DEST/annotations.json" "$DEST/annotations.INCOMPLETE-184991744B.json.quarantine"
fi

echo "[3/4] extracting (plain tar, no -z)"
mkdir -p "$DEST"
tar -xf "$ARCHIVE" -C "$DEST"

echo "[4/4] verifying output"
J=$(find "$DEST/train" -name '*.jpg' | wc -l)
V=$(find "$DEST/train" -maxdepth 1 -mindepth 1 -type d | wc -l)
echo "  jpg=$J/$EXPECT_JPG  dirs=$V/$EXPECT_DIRS"
[ "$J" = "$EXPECT_JPG" ] && [ "$V" = "$EXPECT_DIRS" ] || { echo "count mismatch"; exit 1; }
tar -tf "$ARCHIVE" | sed 's:/$::' | grep -v '^train$' | sort > /tmp/.p4t1_tar.$$
( cd "$DEST" && find train -mindepth 1 \( -type f -o -type d \) | sort ) > /tmp/.p4t1_disk.$$
A=$(comm -23 /tmp/.p4t1_tar.$$ /tmp/.p4t1_disk.$$ | wc -l)
B=$(comm -13 /tmp/.p4t1_tar.$$ /tmp/.p4t1_disk.$$ | wc -l)
rm -f /tmp/.p4t1_tar.$$ /tmp/.p4t1_disk.$$
echo "  bidirectional diff: in_tar_not_on_disk=$A  on_disk_not_in_tar=$B"
[ "$A" = 0 ] && [ "$B" = 0 ] || { echo "diff not empty"; exit 1; }
echo "OK"
