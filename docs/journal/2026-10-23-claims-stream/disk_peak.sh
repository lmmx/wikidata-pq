#!/usr/bin/env bash
# Sample the disk's used bytes every 0.2 s until $1 exists; print the peak over the start
start=$(df --output=used -B1M / | tail -1); peak=$start
while [ ! -e "$1" ]; do u=$(df --output=used -B1M / | tail -1); [ $u -gt $peak ] && peak=$u; sleep 0.2; done
echo "disk used over start: peak $((peak - start)) MB"
