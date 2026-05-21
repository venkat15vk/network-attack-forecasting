"""
Phase 2 viability experiment for Patent 2 (adaptive network attack
prediction) on the REAL OpenSSH log dataset (logpai/loghub 2K sample).

Task: given the past 5 minutes of activity for source IP X, predict
  - REGRESSION target: the number of attacky events on X in the next 15 minutes
  - CLASSIFICATION target: whether there will be ANY attacky event in the next 15 minutes

Feature stacks compared:
  - RF[all]       : outlier + agg + ctx     (Patent 2's full feature set)
  - RF[outlier]   : outlier scores only
  - RF[agg]       : aggregated counts only
  - RF[noagg]     : outlier + ctx (no counts) -- ablation
  - RF[noout]     : agg + ctx (no outlier)    -- ablation
  - persist       : naive baseline: "next 15 min count == past 5 min attacky count"
  - mean_baseline : predict the training-set mean for everything

CRITICAL CAVEATS:
  - 2K-event sample. Only 30 distinct IPs over ~4 hours.
  - Cell counts are small: ~1,600 viable prediction cells total.
  - The attacky-event heuristic (event template IDs) is the same as Paper 1's;
    not ground-truth attack labels.
  - 4-hour window is far smaller than the patent's 3-week behavioral profile
    timeframe. We fit the profile on the first half of each IP's history.

The harness can be re-run on the FULL Loghub OpenSSH dataset (655K events,
28 days, 70 MB on Zenodo) on a machine outside this sandbox; the harness
is dataset-agnostic.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import StandardScaler
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from parse_openssh import parse_openssh_csv             # noqa: E402
from features import (                                   # noqa: E402
    build_features, fb_to_array, FEATURE_NAMES,
    OUTLIER_FEATURE_IDX, AGG_FEATURE_IDX, CTX_FEATURE_IDX,
)


# --------------------------------------------------------------------- prep

def prepare_events(path: Path) -> pd.DataFrame:
    df = parse_openssh_csv(path)
    df = df[df['src_ip'].notna()].reset_index(drop=True)
    df['ts'] = pd.to_datetime(df['timestamp_unix'], unit='s', utc=True)
    df = df.sort_values('ts').reset_index(drop=True)

    # Pre-compute the categorical attributes the feature builder will use.
    df['ip0']      = df['src_ip'].str.split('.').str[0]
    df['hour_str'] = df['ts'].dt.hour.astype(str)
    df['hour_now'] = df['ts'].dt.hour
    return df


# ----------------------------------------------------------------- cells

def build_cells(
    df: pd.DataFrame,
    lookback_min: int = 5,
    horizon_min: int = 15,
    sample_per_ip: int | None = None,
    rng: np.random.RandomState | None = None,
) -> pd.DataFrame:
    """For each event on each IP, build one prediction cell where:
       - features come from [t - lookback_min, t)
       - target_count = #attacky events in [t, t + horizon_min)
       - target_any   = 1 if target_count > 0
       - profile_data = the IP's events in [start_of_history, t - lookback_min)
                        (used for fitting the per-IP outlier profile)

    Only cells with at least one lookback event AND at least one horizon
    event AND a non-empty profile are returned.
    """
    rng = rng or np.random.RandomState(0)
    rows = []
    cells_seen = 0
    cells_kept = 0

    by_ip = df.groupby('src_ip')

    for ip, sub in by_ip:
        sub = sub.sort_values('ts').reset_index(drop=True)
        times = sub['ts'].values
        attacky = sub['is_attacky'].values.astype(int)

        # iterate over events of this IP as anchor points
        for i, t in enumerate(times):
            cells_seen += 1
            t = sub['ts'].iloc[i]
            t_low = t - pd.Timedelta(minutes=lookback_min)
            t_high = t + pd.Timedelta(minutes=horizon_min)

            # lookback events on this IP
            lb_mask = (sub['ts'] >= t_low) & (sub['ts'] < t)
            if lb_mask.sum() == 0:
                continue

            # horizon events
            hz_mask = (sub['ts'] >= t) & (sub['ts'] < t_high)
            if hz_mask.sum() == 0:
                continue

            # profile data: this IP's events strictly before t_low (so no leakage
            # from features into profile, and no leakage from horizon into profile).
            prof_data = sub[sub['ts'] < t_low]
            if len(prof_data) < 3:
                continue  # patent says >= 3 weeks; we relax to >= 3 events min

            fb = build_features(df, ip, t, lookback_min, prof_data)
            if fb is None:
                continue

            target_count = int(sub.loc[hz_mask, 'is_attacky'].sum())

            rows.append({
                'ip': ip,
                't_anchor': t,
                'target_count': target_count,
                'target_any': int(target_count > 0),
                **{f'f_{name}': v for name, v in zip(FEATURE_NAMES, fb_to_array(fb))},
            })
            cells_kept += 1

    print(f"  cells seen={cells_seen}, cells kept (with lookback+horizon+profile)={cells_kept}",
          flush=True)
    if not rows:
        return pd.DataFrame(columns=['ip','t_anchor','target_count','target_any',
                                      *[f'f_{n}' for n in FEATURE_NAMES]])
    out = pd.DataFrame(rows).sort_values('t_anchor').reset_index(drop=True)
    return out


# ------------------------------------------------------------- evaluation

def eval_clf(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {'auc_pr': float('nan'), 'auc_roc': float('nan'),
                'n': len(y_true), 'n_pos': int(y_true.sum())}
    return {
        'auc_pr':  float(average_precision_score(y_true, y_score)),
        'auc_roc': float(roc_auc_score(y_true, y_score)),
        'n':       int(len(y_true)),
        'n_pos':   int(y_true.sum()),
    }


def eval_reg(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'mae':  float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2':   float(r2_score(y_true, y_pred)),
        'n':    int(len(y_true)),
    }


# ----------------------------------------------------------------- splits

def chrono_split(cells: pd.DataFrame, train_frac: float = 0.7):
    cut = int(len(cells) * train_frac)
    return cells.iloc[:cut].reset_index(drop=True), cells.iloc[cut:].reset_index(drop=True)


# --------------------------------------------------------------- runner

def run_one(cells: pd.DataFrame, train_frac: float, seed: int) -> list[dict]:
    train, test = chrono_split(cells, train_frac=train_frac)
    if len(train) < 10 or len(test) < 10 or test['target_any'].sum() < 3:
        return []

    feature_cols = [f'f_{n}' for n in FEATURE_NAMES]
    X_train = train[feature_cols].values
    X_test  = test[feature_cols].values

    y_train_cls = train['target_any'].values.astype(int)
    y_test_cls  = test['target_any'].values.astype(int)
    y_train_reg = train['target_count'].values.astype(float)
    y_test_reg  = test['target_count'].values.astype(float)

    out_idx = OUTLIER_FEATURE_IDX
    agg_idx = AGG_FEATURE_IDX
    ctx_idx = CTX_FEATURE_IDX

    feature_stacks = {
        'RF[all]':     list(range(len(feature_cols))),
        'RF[outlier]': out_idx + ctx_idx,
        'RF[agg]':     agg_idx + ctx_idx,
        'RF[noagg]':   out_idx + ctx_idx,            # alias of outlier
        'RF[noout]':   agg_idx + ctx_idx,            # alias of agg
    }
    # Dedup aliases
    seen, dedup = set(), {}
    for k, idx in feature_stacks.items():
        key = tuple(sorted(idx))
        if key in seen and k != 'RF[all]': continue
        seen.add(key); dedup[k] = idx
    feature_stacks = dedup

    rows = []
    for name, idx in feature_stacks.items():
        Xtr = X_train[:, idx]; Xte = X_test[:, idx]

        # classifier
        clf = RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=seed,
            class_weight='balanced_subsample', min_samples_leaf=2,
        ).fit(Xtr, y_train_cls)
        s = clf.predict_proba(Xte)[:, 1]
        r = eval_clf(y_test_cls, s)
        r.update(method=name, kind='cls', seed=seed, train_frac=train_frac)
        rows.append(r)

        # regressor
        reg = RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=seed,
            min_samples_leaf=2,
        ).fit(Xtr, y_train_reg)
        yhat = reg.predict(Xte)
        r = eval_reg(y_test_reg, yhat)
        r.update(method=name, kind='reg', seed=seed, train_frac=train_frac)
        rows.append(r)

    # baselines
    # 1. persist: predict the past-5min attacky count for the future-15min
    persist_pred = test['f_n_failed_password'].values + \
                   test['f_n_invalid_user'].values + \
                   test['f_n_pam_auth_fail'].values
    # for classification, just threshold persist > 0
    r = eval_clf(y_test_cls, (persist_pred > 0).astype(float))
    r.update(method='persist[>0]', kind='cls', seed=seed, train_frac=train_frac); rows.append(r)
    # for regression, scale persist up by 3 (5min lookback -> 15min horizon)
    r = eval_reg(y_test_reg, persist_pred * 3.0)
    r.update(method='persist[×3]', kind='reg', seed=seed, train_frac=train_frac); rows.append(r)

    # 2. mean baseline
    r = eval_reg(y_test_reg, np.full_like(y_test_reg, fill_value=y_train_reg.mean()))
    r.update(method='mean_baseline', kind='reg', seed=seed, train_frac=train_frac); rows.append(r)
    # for classification: predict training positive rate
    r = eval_clf(y_test_cls, np.full_like(y_test_cls, fill_value=y_train_cls.mean(), dtype=float))
    r.update(method='mean_baseline', kind='cls', seed=seed, train_frac=train_frac); rows.append(r)

    return rows


def main(args):
    print(f"\n=== Phase 2 viability — Patent 2 forecasting on REAL OpenSSH 2K ===\n",
          flush=True)
    t0 = time.time()
    df = prepare_events(args.data)
    print(f"Parsed {len(df):,} events, {df['src_ip'].nunique()} IPs")

    print(f"\nBuilding prediction cells (lookback={args.lookback}min, "
          f"horizon={args.horizon}min) ...", flush=True)
    cells = build_cells(df, lookback_min=args.lookback, horizon_min=args.horizon)
    print(f"  -> {len(cells):,} cells")
    if len(cells) == 0:
        print("No cells generated. Aborting.")
        return
    print(f"  target_any  rate: {cells['target_any'].mean()*100:.1f}%")
    print(f"  target_count: mean={cells['target_count'].mean():.1f}, "
          f"median={cells['target_count'].median():.0f}, "
          f"max={cells['target_count'].max()}, std={cells['target_count'].std():.1f}")

    print("\nMulti-split evaluation ...", flush=True)
    all_rows = []
    for i, tf in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
        print(f"  train_frac={tf}")
        rs = run_one(cells, train_frac=tf, seed=i)
        all_rows.extend(rs)

    res = pd.DataFrame(all_rows)

    # Print summaries
    print("\n" + "="*70)
    print("CLASSIFICATION (any attacky event in next horizon)")
    print("="*70)
    cls_res = res[res['kind']=='cls'].copy()
    if len(cls_res):
        agg = cls_res.groupby('method').agg(
            pr_mean=('auc_pr','mean'), pr_std=('auc_pr','std'),
            roc_mean=('auc_roc','mean'), roc_std=('auc_roc','std'),
            n_runs=('auc_pr','count'),
        ).sort_values('pr_mean', ascending=False)
        for name, row in agg.iterrows():
            print(f"  {name:<18s} AUC-PR={row['pr_mean']:.3f}±{row['pr_std']:.3f}  "
                  f"AUC-ROC={row['roc_mean']:.3f}±{row['roc_std']:.3f}  (n={row['n_runs']})")

        # paired tests vs RF[all]
        if 'RF[all]' in agg.index:
            print("\nPaired t-tests (RF[all] vs each):")
            base = cls_res[cls_res.method=='RF[all]'].sort_values(['train_frac','seed'])['auc_pr'].values
            for m in agg.index:
                if m == 'RF[all]': continue
                other = cls_res[cls_res.method==m].sort_values(['train_frac','seed'])['auc_pr'].values
                if len(other) != len(base) or np.isnan(other).any() or np.isnan(base).any():
                    continue
                t, p = stats.ttest_rel(base, other)
                wins = int((base > other).sum())
                print(f"  vs {m:<18s} wins={wins}/{len(base)}  Δ={base.mean()-other.mean():+.4f}  p={p:.4f}")

    print("\n" + "="*70)
    print("REGRESSION (count of attacky events in next horizon)")
    print("="*70)
    reg_res = res[res['kind']=='reg'].copy()
    if len(reg_res):
        agg = reg_res.groupby('method').agg(
            mae_mean=('mae','mean'), mae_std=('mae','std'),
            rmse_mean=('rmse','mean'), rmse_std=('rmse','std'),
            r2_mean=('r2','mean'), r2_std=('r2','std'),
            n_runs=('mae','count'),
        ).sort_values('mae_mean')
        for name, row in agg.iterrows():
            print(f"  {name:<18s} MAE={row['mae_mean']:.2f}±{row['mae_std']:.2f}  "
                  f"RMSE={row['rmse_mean']:.2f}±{row['rmse_std']:.2f}  "
                  f"R²={row['r2_mean']:+.3f}±{row['r2_std']:.3f}  (n={row['n_runs']})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',     type=Path, default=Path('data/openssh_2k.csv'))
    ap.add_argument('--out',      type=Path, default=Path('results/forecast.csv'))
    ap.add_argument('--lookback', type=int,  default=5,  help='Lookback window (minutes)')
    ap.add_argument('--horizon',  type=int,  default=15, help='Prediction horizon (minutes)')
    args = ap.parse_args()
    main(args)
