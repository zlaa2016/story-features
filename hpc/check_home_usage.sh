#!/usr/bin/env bash
# What is using the home quota? Run on a login node. Read-only: it deletes and moves nothing.
echo "== quota"
(myquota 2>/dev/null || quota -s 2>/dev/null || echo "no quota command found") | head -20
echo
echo "== largest items in \$HOME (top 15)"
du -sh "$HOME"/.[!.]* "$HOME"/* 2>/dev/null | sort -h | tail -15
echo
echo "== known cache locations"
for d in .cache .local .conda .singularity .apptainer .huggingface .ipython .jupyter .nv .triton; do
  [[ -e "$HOME/$d" ]] && du -sh "$HOME/$d" 2>/dev/null
done
