# Network attack forecasting

Reproducible implementation of a per-host attack-volume forecaster that
combines per-IP behavioral-profile outlier scores with short-window
aggregated count features, evaluated on the public Loghub OpenSSH
server log corpus.

**Paper**: *Forecasting Per-Host Attack Volume from Short-Window
Behavioral Features: A Study on Real SSH Server Logs*. Venkatakrishnan
Gopalakrishnan, 2026. [arXiv preprint — link added after submission]

## Headline result

On the full 28-day Loghub OpenSSH corpus (542,861 events, 1,241 source
IPs), classifying whether each source IP will produce any
attack-related event in the next 15 minutes:

| Method            | AUC-PR              | AUC-ROC             |
| ----------------- | ------------------- | ------------------- |
| **RF[all]**       | **0.952 ± 0.022**   | **0.970 ± 0.008**   |
| RF[agg]           | 0.927 ± 0.035       | 0.948 ± 0.023       |
| persist[>0]       | 0.871 ± 0.039       | 0.919 ± 0.009       |
| RF[outlier]       | 0.742 ± 0.124       | 0.880 ± 0.049       |
| mean_baseline     | 0.409 ± 0.115       | 0.500 ± 0.000       |

Mean ± SD over 5 chronological train/test splits. The combined feature
stack wins on every split and beats every alternative with paired
t-tests p < 0.05.

## Layout

    src/
      parse_openssh.py     # Loghub OpenSSH parser (CSV or raw .log)
      features.py          # per-IP outlier-score + count features
      run_forecast.py      # original event-anchored harness
      run_forecast_fast.py # streaming windowed-aggregate harness (used in paper)
    results/
      forecast_full.csv         # per-split metrics, full dataset run
      forecast_full.log         # terminal output from the run
      forecast_full_summary.md  # human-readable summary
    requirements.txt
    LICENSE                 # MIT
    .gitignore
    README.md

## Reproducing

### 1. Install

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

### 2. Get the data

The Loghub OpenSSH dataset is a 70 MB compressed file hosted at Zenodo
(record 8196385).

    mkdir -p data
    curl -L -o data/SSH.tar.gz \
      "https://zenodo.org/records/8196385/files/SSH.tar.gz?download=1"
    tar -xzf data/SSH.tar.gz -C data/

You should now have `data/SSH.log`, ~70 MB and 655K lines.

### 3. Run

    python3 src/run_forecast_fast.py \
      --data data/SSH.log \
      --out  results/forecast.csv \
      --lookback 5 \
      --horizon  15 \
      --bucket   1

Runs in ~1 minute on a recent laptop. The script prints classification
and regression tables to stdout and writes per-split metrics to the
CSV.

## Dataset notes

- 28-day collection (2015-12-10 to 2016-01-07) of a real SSH server.
- Loghub publishes the file as `SSH.tar.gz` (originally `OpenSSH` in the
  Loghub catalog).
- We parse 542,861 events from 655,146 raw lines; the dropped lines are
  multi-line entries and non-standard headers that the parser skips.
- 754 of 1,241 source IPs have ≥5 events and are usable for behavioral
  profiling.

## Cite

If you use this code please cite the paper above and the Loghub dataset:

    @article{he2020loghub,
      title   = {Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics},
      author  = {He, Shilin and Zhu, Jieming and He, Pinjia and Lyu, Michael R.},
      journal = {arXiv preprint arXiv:2008.06448},
      year    = {2020}
    }

## License

MIT (see `LICENSE`).
