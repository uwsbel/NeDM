#!/bin/bash
# arena_gator_20260925 task B stage 2 (Bfull): progress of the 2,318 new soil_v4 rows (ids in G3/tools/bf/logs/bf_new_ids.txt) on the cluster login node.
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
cd $G3/soil_v1
n=0; c=0; cl=0; g=0; h=0
while read d; do n=$((n+1)); if [ -e runs/$d/episode_complete.json ]; then c=$((c+1)); case $d in gator__*) g=$((g+1));; *) h=$((h+1));; esac; elif [ -d claims/$d ]; then cl=$((cl+1)); fi; done < $G3/tools/bf/logs/bf_new_ids.txt
echo "$(date +%H:%M) new rows $n complete $c (gator $g hmmwv $h) claimed-running $cl failed-files $(ls failed | wc -l) jobs $(squeue -u harry -h -r | wc -l) running $(squeue -u harry -h -r -t R | wc -l)"
