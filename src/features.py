"""
Feature engineering for adaptive network attack prediction
(Patent 2 task: predict attack count in next H minutes from past L minutes
of activity).

Two feature families:

  1. OUTLIER SCORES per categorical attribute, computed against a per-IP
     behavioral profile built from the EARLIER half of an IP's history.
     Score in [0, 1]: 0 = value is common for this IP, 1 = value never
     seen before for this IP.
     Categorical attributes used: attempted_user, port, ip0 (first IP octet
     when treating destination/local context), hour_of_day, event_template.

  2. AGGREGATED COUNTS in the lookback window (default 5 min):
     - total_events
     - n_invalid_user (E13 template)
     - n_failed_password (E10)
     - n_pam_auth_fail (E19)
     - n_break_in_warn (E27)
     - n_distinct_users (attempted usernames)
     - n_distinct_ports
     - n_distinct_templates
     - fraction_attacky (proxy for incoming attack rate)

Target options (parameterized at evaluation time):
     - reg target: count of attacky events in next H minutes
     - cls target: indicator of >=1 attacky event in next H minutes
     - severity target: bucketed count (none/low/med/high)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


# IDs of templates considered 'attacky' (re-using Paper 1's heuristic).
ATTACKY_EVENTS = {'E10', 'E12', 'E13', 'E19', 'E21', 'E27'}


@dataclass
class FeatureBundle:
    """Container for features extracted from one (IP, t) prediction cell."""
    # outlier scores in [0,1]
    out_user: float
    out_port: float
    out_ip0: float
    out_hour: float
    out_template: float
    # aggregated counts over lookback
    n_events: int
    n_invalid_user: int
    n_failed_password: int
    n_pam_auth_fail: int
    n_break_in_warn: int
    n_distinct_users: int
    n_distinct_ports: int
    n_distinct_templates: int
    fraction_attacky: float
    # the latest event's hour-of-day (not a stat, just context)
    hour_now: int


def per_ip_profile(history: pd.DataFrame, attr: str) -> dict[str, int]:
    """Frequency table for one categorical attribute on this IP's history."""
    return history[attr].dropna().astype(str).value_counts().to_dict()


def outlier_score(value, profile: dict[str, int], total: int) -> float:
    """Normalized outlier score in [0,1].
    
    The patent says: 0 = frequently observed value not associated with attacks,
    1 = highly atypical for this user. Our implementation: relative frequency
    of the observed value, transformed so common = low score, rare = high score.
    Missing-from-history values get score 1.0.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.5  # missing data, neutral
    value = str(value)
    if total == 0 or value not in profile:
        return 1.0  # never seen before for this IP
    freq = profile[value] / total
    # smooth via -log mapping clipped to [0,1]
    # rare value (freq~0) -> score near 1; common value (freq~1) -> score near 0
    return float(np.clip(1.0 - freq, 0.0, 1.0))


def build_features(
    events: pd.DataFrame,
    ip: str,
    t_anchor: pd.Timestamp,
    lookback_min: int,
    profile_data: pd.DataFrame,
) -> FeatureBundle | None:
    """Build a feature vector at time t_anchor for IP `ip`, using
    activity in [t_anchor - lookback_min, t_anchor).

    profile_data is the slice of `ip`'s history used to fit the
    behavioral profile. Outlier scores are computed against it.
    """
    t_lo = t_anchor - pd.Timedelta(minutes=lookback_min)
    window = events[
        (events['src_ip'] == ip)
        & (events['ts'] >= t_lo)
        & (events['ts'] < t_anchor)
    ]
    if len(window) == 0:
        return None

    # ---- outlier scores against the IP's earlier behavioral profile ----
    user_prof = per_ip_profile(profile_data, 'attempted_user')
    port_prof = per_ip_profile(profile_data, 'port')
    ip0_prof  = per_ip_profile(profile_data, 'ip0')
    hour_prof = per_ip_profile(profile_data, 'hour_str')
    tmpl_prof = per_ip_profile(profile_data, 'event_id')
    n_user = sum(user_prof.values())
    n_port = sum(port_prof.values())
    n_ip0  = sum(ip0_prof.values())
    n_hour = sum(hour_prof.values())
    n_tmpl = sum(tmpl_prof.values())

    # Use the LATEST event in window as the 'current' observation
    last = window.iloc[-1]
    out_user = outlier_score(last['attempted_user'], user_prof, n_user)
    out_port = outlier_score(last['port'],           port_prof, n_port)
    out_ip0  = outlier_score(last['ip0'],            ip0_prof,  n_ip0)
    out_hour = outlier_score(last['hour_str'],       hour_prof, n_hour)
    out_tmpl = outlier_score(last['event_id'],       tmpl_prof, n_tmpl)

    # ---- aggregated counts ----
    n_events = len(window)
    eid = window['event_id']
    n_inv_user = int((eid == 'E13').sum())
    n_fail_pw  = int((eid == 'E10').sum())
    n_pam_fail = int((eid == 'E19').sum())
    n_break_in = int((eid == 'E27').sum())
    n_du = window['attempted_user'].dropna().nunique()
    n_dp = window['port'].dropna().nunique()
    n_dt = eid.nunique()
    frac_a = float(window['is_attacky'].mean())

    return FeatureBundle(
        out_user=out_user,
        out_port=out_port,
        out_ip0=out_ip0,
        out_hour=out_hour,
        out_template=out_tmpl,
        n_events=n_events,
        n_invalid_user=n_inv_user,
        n_failed_password=n_fail_pw,
        n_pam_auth_fail=n_pam_fail,
        n_break_in_warn=n_break_in,
        n_distinct_users=int(n_du),
        n_distinct_ports=int(n_dp),
        n_distinct_templates=int(n_dt),
        fraction_attacky=frac_a,
        hour_now=int(last['hour_now']),
    )


def fb_to_array(fb: FeatureBundle) -> np.ndarray:
    return np.array([
        fb.out_user, fb.out_port, fb.out_ip0, fb.out_hour, fb.out_template,
        fb.n_events, fb.n_invalid_user, fb.n_failed_password,
        fb.n_pam_auth_fail, fb.n_break_in_warn,
        fb.n_distinct_users, fb.n_distinct_ports, fb.n_distinct_templates,
        fb.fraction_attacky, fb.hour_now,
    ], dtype=float)


FEATURE_NAMES = [
    'out_user', 'out_port', 'out_ip0', 'out_hour', 'out_template',
    'n_events', 'n_invalid_user', 'n_failed_password',
    'n_pam_auth_fail', 'n_break_in_warn',
    'n_distinct_users', 'n_distinct_ports', 'n_distinct_templates',
    'fraction_attacky', 'hour_now',
]

# Mask flags for ablation studies: separate "outlier-only" from "aggregate-only"
OUTLIER_FEATURE_IDX = [0, 1, 2, 3, 4]
AGG_FEATURE_IDX     = [5, 6, 7, 8, 9, 10, 11, 12, 13]
CTX_FEATURE_IDX     = [14]  # just hour-of-day
