#!/bin/zsh
# Run the full driver on every input JSON directly under inputs/ (NOT inputs/extra/).
cd "$(dirname "$0")"
mkdir -p run_logs
SUMMARY=run_logs/_summary.txt
: > "$SUMMARY"
for f in inputs/*.json; do
  base=$(basename "$f" .json)
  log="run_logs/${base}.log"
  start=$(date +%s)
  if python3 species_limits.py "$f" > "$log" 2>&1; then
    st="PASS"
  else
    st="FAIL"
  fi
  end=$(date +%s)
  dur=$((end - start))
  line="${st}  ${dur}s  ${base}"
  echo "$line" | tee -a "$SUMMARY"
done
echo "---- done ----" | tee -a "$SUMMARY"
