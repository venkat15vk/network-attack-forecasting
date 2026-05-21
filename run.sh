#!/usr/bin/env bash
# End-to-end driver for the Patent 2 viability experiment.
# Run from the repo root (the directory containing src/, data/, results/).
set -euo pipefail

# 0) install deps if needed
python3 -m pip install -r requirements.txt

# 1) ensure data is there
if [[ ! -f data/OpenSSH.log ]]; then
  echo "data/OpenSSH.log not found. Download the Loghub OpenSSH full dataset first:"
  echo "  mkdir -p data"
  echo "  curl -L -o data/OpenSSH.tar.gz 'https://zenodo.org/records/8196385/files/OpenSSH.tar.gz?download=1'"
  echo "  tar -xzf data/OpenSSH.tar.gz -C data/"
  exit 1
fi

# 2) run the experiment
python3 src/run_forecast.py \
    --data data/OpenSSH.log \
    --out  results/forecast.csv \
    --lookback 5 \
    --horizon  15

echo
echo "Done. Paste results/forecast.csv (it's ~5 KB) and the terminal output above."
