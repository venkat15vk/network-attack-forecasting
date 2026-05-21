"""
Patent 2 forecasting harness (fast version).

Streams windowed aggregates over sorted events per IP, total work O(events).
Anchors at every 1-minute bucket by default (configurable with --bucket).

Predict (per IP, at anchor time t):
  target_count = #attacky events in [t, t + horizon_min)
  target_any   = 1[target_count > 0]
From features over the lookback window [t - lookback_min, t):
  - per-IP outlier scores against behavioral profile (events strictly
    before the lookback window)
  - aggregated counts in lookback
"""

from __future__ import annotations
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from parse_openssh import parse_openssh  # noqa: E402


FEATURE_NAMES = [
    'out_user', 'out_port', 'out_ip0', 'out_hour', 'out_template',
    'n_events', 'n_invalid_user', 'n_failed_password',
    'n_pam_auth_fail', 'n_break_in_warn',
    'n_distinct_users', 'n_distinct_ports', 'n_distinct_templates',
    'fraction_attacky', 'hour_now',
]
OUTLIER_IDX = list(range(0, 5))
AGG_IDX     = list(range(5, 14))
CTX_IDX     = [14]


def prepare_events(path: Path) -> pd.DataFrame:
    df = parse_openssh(path)
    df = df[df['src_ip'].notna()].reset_index(drop=True)
    df['ts'] = pd.to_datetime(df['timestamp_unix'], unit='s', utc=True)
    df = df.sort_values('ts').reset_index(drop=True)
    df['ip0']      = df['src_ip'].astype(str).str.split('.').str[0]
    df['hour_str'] = df['ts'].dt.hour.astype(str)
    df['hour_now'] = df['ts'].dt.hour.astype(int)
    df['attempted_user'] = df['attempted_user'].fillna('').astype(str)
    df['port']           = df['port'].fillna('').astype(str)
    return df


def build_cells_fast(
    df: pd.DataFrame,
    lookback_min: int = 5,
    horizon_min:  int = 15,
    bucket_min:   int = 1,
    min_profile:  int = 3,
    min_events_per_ip: int = 5,
) -> pd.DataFrame:
    """O(events) per-IP streaming windowed aggregate."""
    lookback_s = lookback_min * 60
    horizon_s  = horizon_min  * 60
    bucket_s   = bucket_min   * 60

    rows = []
    cells_seen = 0
    cells_kept = 0
    ips_processed = 0
    ips_skipped_short = 0

    grouped = df.groupby('src_ip', sort=False)
    n_groups = len(grouped)
    last_progress_time = time.time()
    progress_iv = 5.0  # log progress every 5s

    for gi, (ip, sub) in enumerate(grouped, 1):
        now = time.time()
        if now - last_progress_time > progress_iv or gi == n_groups:
            print(f"    IP {gi}/{n_groups}  kept={cells_kept:,}  "
                  f"seen={cells_seen:,}  skipped_short={ips_skipped_short}",
                  flush=True)
            last_progress_time = now

        n_sub = len(sub)
        if n_sub < min_events_per_ip:
            ips_skipped_short += 1
            continue
        ips_processed += 1

        sub = sub.sort_values('ts').reset_index(drop=True)
        n = len(sub)

        ts = sub['ts'].values.astype('datetime64[s]').astype(np.int64)
        eid = sub['event_id'].values
        attacky = sub['is_attacky'].values.astype(np.int8)
        users = sub['attempted_user'].values
        ports = sub['port'].values
        ip0s  = sub['ip0'].values
        hours = sub['hour_str'].values
        hour_now_arr = sub['hour_now'].values

        t_start = int(ts[0])
        t_end   = int(ts[-1])
        anchors = np.arange(t_start, t_end + 1, bucket_s, dtype=np.int64)

        # Profile counters (events strictly before t_lb_lo)
        prof_user, prof_port, prof_ip0, prof_hour, prof_tmpl = (
            Counter(), Counter(), Counter(), Counter(), Counter())
        prof_total = 0
        p_idx = 0

        # Lookback queue [t_lb_lo, t_lb_hi)
        lb_user, lb_port, lb_ip0, lb_hour, lb_tmpl = (
            Counter(), Counter(), Counter(), Counter(), Counter())
        lb_eid_counts = Counter()
        lb_attacky_sum = 0
        lb_n = 0
        lb_left = 0   # next index to add (>= 0)
        lb_right = 0  # next index to remove (oldest)

        # Horizon cursors
        hz_left = 0
        hz_right = 0

        for anchor in anchors:
            cells_seen += 1
            t_lb_lo = anchor - lookback_s
            t_lb_hi = anchor
            t_hz_lo = anchor
            t_hz_hi = anchor + horizon_s

            # Profile <- events with ts < t_lb_lo
            while p_idx < n and ts[p_idx] < t_lb_lo:
                prof_user[users[p_idx]] += 1
                prof_port[ports[p_idx]] += 1
                prof_ip0[ip0s[p_idx]]   += 1
                prof_hour[hours[p_idx]] += 1
                prof_tmpl[eid[p_idx]]   += 1
                prof_total += 1
                p_idx += 1

            # Lookback right edge: add events with ts < t_lb_hi
            while lb_left < n and ts[lb_left] < t_lb_hi:
                lb_user[users[lb_left]] += 1
                lb_port[ports[lb_left]] += 1
                lb_ip0[ip0s[lb_left]]   += 1
                lb_hour[hours[lb_left]] += 1
                lb_tmpl[eid[lb_left]]   += 1
                lb_eid_counts[eid[lb_left]] += 1
                lb_attacky_sum += int(attacky[lb_left])
                lb_n += 1
                lb_left += 1
            # Lookback left edge: remove events with ts < t_lb_lo
            while lb_right < lb_left and ts[lb_right] < t_lb_lo:
                u = users[lb_right]; lb_user[u] -= 1
                if lb_user[u] == 0: del lb_user[u]
                p = ports[lb_right]; lb_port[p] -= 1
                if lb_port[p] == 0: del lb_port[p]
                i0 = ip0s[lb_right]; lb_ip0[i0] -= 1
                if lb_ip0[i0] == 0: del lb_ip0[i0]
                h = hours[lb_right]; lb_hour[h] -= 1
                if lb_hour[h] == 0: del lb_hour[h]
                e = eid[lb_right]; lb_tmpl[e] -= 1
                if lb_tmpl[e] == 0: del lb_tmpl[e]
                lb_eid_counts[e] -= 1
                if lb_eid_counts[e] == 0: del lb_eid_counts[e]
                lb_attacky_sum -= int(attacky[lb_right])
                lb_n -= 1
                lb_right += 1

            if lb_n == 0 or prof_total < min_profile:
                continue

            # Horizon
            while hz_left < n and ts[hz_left] < t_hz_lo:
                hz_left += 1
            while hz_right < n and ts[hz_right] < t_hz_hi:
                hz_right += 1
            if hz_right <= hz_left:
                continue

            target_count = int(attacky[hz_left:hz_right].sum())

            # 'Current' observation = last event added to lookback
            cur = lb_left - 1

            def score(v, prof, total):
                if total == 0 or v not in prof:
                    return 1.0
                return 1.0 - prof[v] / total

            row = {
                'ip': ip,
                't_anchor': int(anchor),
                'target_count': target_count,
                'target_any':   int(target_count > 0),
                'f_out_user':       score(users[cur], prof_user, prof_total),
                'f_out_port':       score(ports[cur], prof_port, prof_total),
                'f_out_ip0':        score(ip0s[cur],  prof_ip0,  prof_total),
                'f_out_hour':       score(hours[cur], prof_hour, prof_total),
                'f_out_template':   score(eid[cur],   prof_tmpl, prof_total),
                'f_n_events':              lb_n,
                'f_n_invalid_user':        lb_eid_counts.get('E13', 0),
                'f_n_failed_password':     lb_eid_counts.get('E10', 0),
                'f_n_pam_auth_fail':       lb_eid_counts.get('E19', 0),
                'f_n_break_in_warn':       lb_eid_counts.get('E27', 0),
                'f_n_distinct_users':      len(lb_user) - (1 if '' in lb_user else 0),
                'f_n_distinct_ports':      len(lb_port) - (1 if '' in lb_port else 0),
                'f_n_distinct_templates':  len(lb_tmpl),
                'f_fraction_attacky':      lb_attacky_sum / lb_n,
                'f_hour_now':              int(hour_now_arr[cur]),
            }
            rows.append(row)
            cells_kept += 1

    print(f"\n  Final: cells seen={cells_seen:,}, kept={cells_kept:,}, "
          f"IPs used={ips_processed:,}, IPs skipped (too short)={ips_skipped_short:,}",
          flush=True)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('t_anchor').reset_index(drop=True)


def eval_clf(y_true, y_score):
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {'auc_pr': float('nan'), 'auc_roc': float('nan'),
                'n': int(len(y_true)), 'n_pos': int(y_true.sum())}
    return {
        'auc_pr':  float(average_precision_score(y_true, y_score)),
        'auc_roc': float(roc_auc_score(y_true, y_score)),
        'n':       int(len(y_true)),
        'n_pos':   int(y_true.sum()),
    }

def eval_reg(y_true, y_pred):
    return {
        'mae':  float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2':   float(r2_score(y_true, y_pred)),
        'n':    int(len(y_true)),
    }

def chrono_split(cells: pd.DataFrame, train_frac: float):
    cut = int(len(cells) * train_frac)
    return cells.iloc[:cut].reset_index(drop=True), cells.iloc[cut:].reset_index(drop=True)


def run_one(cells, train_frac, seed):
    train, test = chrono_split(cells, train_frac)
    if len(train) < 50 or len(test) < 50 or test['target_any'].sum() < 5:
        return []

    fcols = [f'f_{n}' for n in FEATURE_NAMES]
    X_tr = train[fcols].values
    X_te = test[fcols].values
    y_tr_c = train['target_any'].values.astype(int)
    y_te_c = test['target_any'].values.astype(int)
    y_tr_r = train['target_count'].values.astype(float)
    y_te_r = test['target_count'].values.astype(float)

    stacks = {
        'RF[all]':     list(range(len(fcols))),
        'RF[outlier]': OUTLIER_IDX + CTX_IDX,
        'RF[agg]':     AGG_IDX + CTX_IDX,
    }
    rows = []
    for name, idx in stacks.items():
        Xtr, Xte = X_tr[:, idx], X_te[:, idx]
        clf = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed,
            class_weight='balanced_subsample', min_samples_leaf=2,
        ).fit(Xtr, y_tr_c)
        s = clf.predict_proba(Xte)[:, 1]
        r = eval_clf(y_te_c, s)
        r.update(method=name, kind='cls', seed=seed, train_frac=train_frac)
        rows.append(r)
        reg = RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=seed,
            min_samples_leaf=2,
        ).fit(Xtr, y_tr_r)
        yh = reg.predict(Xte)
        r = eval_reg(y_te_r, yh)
        r.update(method=name, kind='reg', seed=seed, train_frac=train_frac)
        rows.append(r)

    persist = (test['f_n_invalid_user'] + test['f_n_failed_password']
               + test['f_n_pam_auth_fail'] + test['f_n_break_in_warn']).values
    r = eval_clf(y_te_c, (persist > 0).astype(float))
    r.update(method='persist[>0]', kind='cls', seed=seed, train_frac=train_frac); rows.append(r)
    r = eval_reg(y_te_r, persist * 3.0)
    r.update(method='persist[×3]', kind='reg', seed=seed, train_frac=train_frac); rows.append(r)
    r = eval_clf(y_te_c, np.full_like(y_te_c, fill_value=y_tr_c.mean(), dtype=float))
    r.update(method='mean_baseline', kind='cls', seed=seed, train_frac=train_frac); rows.append(r)
    r = eval_reg(y_te_r, np.full_like(y_te_r, fill_value=y_tr_r.mean()))
    r.update(method='mean_baseline', kind='reg', seed=seed, train_frac=train_frac); rows.append(r)

    return rows


def main(args):
    print(f"\n=== Patent 2 forecasting (fast) — lookback={args.lookback}m, "
          f"horizon={args.horizon}m, bucket={args.bucket}m ===\n", flush=True)
    t0 = time.time()

    print("Parsing log ...", flush=True)
    df = prepare_events(args.data)
    print(f"  -> {len(df):,} events, {df['src_ip'].nunique():,} IPs, "
          f"span {df['ts'].min()} to {df['ts'].max()}")

    print(f"\nBuilding cells (streaming windowed aggregates, bucket={args.bucket}m) ...",
          flush=True)
    t1 = time.time()
    cells = build_cells_fast(
        df, lookback_min=args.lookback, horizon_min=args.horizon,
        bucket_min=args.bucket,
    )
    print(f"  done in {time.time()-t1:.1f}s; {len(cells):,} cells")
    if len(cells) == 0:
        print("\nNo cells generated. Try reducing --lookback or --bucket.")
        return
    if len(cells) < 100:
        print(f"\nVery few cells ({len(cells)}). Multi-split eval may be unstable.")

    print(f"  target_any rate: {cells['target_any'].mean()*100:.1f}%")
    print(f"  target_count: mean={cells['target_count'].mean():.2f}, "
          f"median={int(cells['target_count'].median())}, "
          f"max={int(cells['target_count'].max())}, "
          f"std={cells['target_count'].std():.1f}")

    if args.max_cells and len(cells) > args.max_cells:
        print(f"  -> downsampling to first {args.max_cells:,} chronological cells")
        cells = cells.iloc[:args.max_cells].reset_index(drop=True)

    print("\nMulti-split evaluation ...", flush=True)
    all_rows = []
    for i, tf in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
        print(f"  train_frac={tf}", flush=True)
        rs = run_one(cells, tf, seed=i)
        all_rows.extend(rs)

    if not all_rows:
        print("\nAll splits produced too few cells for evaluation.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cells.to_csv(args.out.parent / 'cells.csv', index=False)
        print(f"Wrote raw cells to {args.out.parent / 'cells.csv'} ({len(cells)} rows)")
        return

    res = pd.DataFrame(all_rows)

    print("\n" + "="*70)
    print("CLASSIFICATION (any attacky event in next horizon)")
    print("="*70)
    cls = res[res['kind']=='cls'].copy()
    if len(cls):
        agg = cls.groupby('method').agg(
            pr_mean=('auc_pr','mean'), pr_std=('auc_pr','std'),
            roc_mean=('auc_roc','mean'), roc_std=('auc_roc','std'),
            n_runs=('auc_pr','count'),
        ).sort_values('pr_mean', ascending=False)
        for name, row in agg.iterrows():
            print(f"  {name:<18s} AUC-PR={row['pr_mean']:.3f}±{row['pr_std']:.3f}  "
                  f"AUC-ROC={row['roc_mean']:.3f}±{row['roc_std']:.3f}  (n={int(row['n_runs'])})")
        if 'RF[all]' in agg.index:
            print("\nPaired t-tests (RF[all] vs each):")
            base = cls[cls.method=='RF[all]'].sort_values(['train_frac','seed'])['auc_pr'].values
            for m in agg.index:
                if m == 'RF[all]': continue
                other = cls[cls.method==m].sort_values(['train_frac','seed'])['auc_pr'].values
                if len(other) != len(base): continue
                mask = ~(np.isnan(base) | np.isnan(other))
                if mask.sum() < 3: continue
                b, o = base[mask], other[mask]
                t, p = stats.ttest_rel(b, o)
                wins = int((b > o).sum())
                print(f"  vs {m:<18s} wins={wins}/{len(b)}  Δ={b.mean()-o.mean():+.4f}  p={p:.4f}")

    print("\n" + "="*70)
    print("REGRESSION (count of attacky events in next horizon)")
    print("="*70)
    reg = res[res['kind']=='reg'].copy()
    if len(reg):
        agg = reg.groupby('method').agg(
            mae_mean=('mae','mean'), mae_std=('mae','std'),
            rmse_mean=('rmse','mean'), rmse_std=('rmse','std'),
            r2_mean=('r2','mean'), r2_std=('r2','std'),
            n_runs=('mae','count'),
        ).sort_values('mae_mean')
        for name, row in agg.iterrows():
            print(f"  {name:<18s} MAE={row['mae_mean']:6.2f}±{row['mae_std']:.2f}  "
                  f"RMSE={row['rmse_mean']:6.2f}±{row['rmse_std']:.2f}  "
                  f"R²={row['r2_mean']:+.3f}±{row['r2_std']:.3f}  (n={int(row['n_runs'])})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print(f"Total wall time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',      type=Path, required=True)
    ap.add_argument('--out',       type=Path, default=Path('results/forecast.csv'))
    ap.add_argument('--lookback',  type=int,  default=5)
    ap.add_argument('--horizon',   type=int,  default=15)
    ap.add_argument('--bucket',    type=int,  default=1)
    ap.add_argument('--max-cells', type=int,  default=0)
    args = ap.parse_args()
    main(args)
