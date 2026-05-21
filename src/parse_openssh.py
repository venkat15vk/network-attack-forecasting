"""
Parser for the OpenSSH log dataset from logpai/loghub.

Supports two input formats:
  1) Loghub *structured* CSV: columns Date, Day, Time, Pid, Content, EventId,
     EventTemplate. This is what the 2K sample provides.
  2) Raw .log text file: one event per line, format
        "Dec 10 06:55:46 LabSZ sshd[24200]: <content>"
     This is what the full 70 MB Loghub OpenSSH download provides.

Use `parse_openssh(path)` as the unified entry point; it auto-detects
the format from the extension and the first line.

Output schema (one row per event):
  timestamp_unix : int      - unix seconds, synthesized from Date+Time
  pid            : int      - sshd PID
  src_ip         : str|None - extracted from Content (or None)
  attempted_user : str|None - extracted from Content (or None)
  port           : str|None - extracted from Content (or None)
  event_id       : str      - EventTemplate id E1..E27
  event_template : str      - the template text
  raw_content    : str      - the original log message body
  is_attacky     : bool     - heuristic label (see ATTACKY_EVENT_IDS)
"""

from __future__ import annotations
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ------------------------- extraction patterns ------------------------------

IP_RE = re.compile(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')
PORT_RE = re.compile(r'port\s+(\d+)')
INVALID_USER_RE = re.compile(r'[Ii]nvalid user (\S+)\s')
FROM_USER_RE = re.compile(r'Failed password for (?:invalid user )?(\S+) from')
PAM_USER_RE = re.compile(r'\buser=(\S+)')

# Raw-log line format: "Dec 10 06:55:46 LabSZ sshd[24200]: <content>"
RAW_LINE_RE = re.compile(
    r'^(?P<month>\w{3})\s+'
    r'(?P<day>\d{1,2})\s+'
    r'(?P<time>\d{2}:\d{2}:\d{2})\s+'
    r'(?P<host>\S+)\s+'
    r'(?P<proc>\S+?)\[(?P<pid>\d+)\]:\s+'
    r'(?P<content>.*)$'
)

# ------------------------- event templates ---------------------------------

# The full set of EventId -> regex patterns (matched in declaration order).
# We compile from the 2K sample's deduplicated templates so the IDs are
# guaranteed to line up between the structured CSV and the raw-log parses.
#
# Variables (<*>) become non-greedy ".+?" matches.  Anchors are added so
# matches must be whole-content.

_TEMPLATE_TEXTS = [
    ('E1',  'Accepted password for <*> from <*> port <*> ssh2'),
    ('E2',  'Connection closed by <*> [preauth]'),
    ('E3',  'Did not receive identification string from <*>'),
    ('E4',  'Disconnecting: Too many authentication failures for admin [preauth]'),
    ('E5',  'Disconnecting: Too many authentication failures for root [preauth]'),
    ('E6',  'error: Received disconnect from <*>: <*>: com.jcraft.jsch.JSchException: Auth fail [preauth]'),
    ('E7',  'error: Received disconnect from <*>: <*>: No more user authentication methods available. [preauth]'),
    ('E8',  'Failed none for invalid user <*> from <*> port <*> ssh2'),
    ('E10', 'Failed password for invalid user <*> from <*> port <*> ssh2'),   # before E9 (more specific)
    ('E9',  'Failed password for <*> from <*> port <*> ssh2'),
    ('E11', 'fatal: Write failed: Connection reset by peer [preauth]'),
    ('E12', 'input_userauth_request: invalid user <*> [preauth]'),
    ('E13', 'Invalid user <*> from <*>'),
    ('E14', 'message repeated <*> times: [ Failed password for root from <*> port <*>]'),
    ('E18', 'PAM service(sshd) ignoring max retries; <*> > <*>'),             # before E15/E16/E17 (specific phrase)
    ('E17', 'PAM <*> more authentication failures; logname= uid=<*> euid=<*> tty=ssh ruser= rhost=<*>  user=root'),
    ('E16', 'PAM <*> more authentication failures; logname= uid=<*> euid=<*> tty=ssh ruser= rhost=<*>'),
    ('E15', 'PAM <*> more authentication failure; logname= uid=<*> euid=<*> tty=ssh ruser= rhost=<*>'),
    ('E20', 'pam_unix(sshd:auth): authentication failure; logname= uid=<*> euid=<*> tty=ssh ruser= rhost=<*> user=<*>'),
    ('E19', 'pam_unix(sshd:auth): authentication failure; logname= uid=<*> euid=<*> tty=ssh ruser= rhost=<*>'),
    ('E21', 'pam_unix(sshd:auth): check pass; user unknown'),
    ('E22', 'pam_unix(sshd:session): session closed for user <*>'),
    ('E23', 'pam_unix(sshd:session): session opened for user <*> by (uid=<*>)'),
    ('E24', 'Received disconnect from <*>: <*>: Bye Bye [preauth]'),
    ('E25', 'Received disconnect from <*>: <*>: Closed due to user request. [preauth]'),
    ('E26', 'Received disconnect from <*>: <*>: disconnected by user'),
    ('E27', 'reverse mapping checking getaddrinfo for <*> [<*>] failed - POSSIBLE BREAK-IN ATTEMPT!'),
]

def _template_to_regex(t: str) -> re.Pattern:
    # Escape the literal text, then turn the escaped <\*> placeholder
    # back into a non-greedy ".+?" match.
    pat = re.escape(t).replace(r'<\*>', r'.+?')
    return re.compile('^' + pat + '$')

_COMPILED_TEMPLATES = [(eid, t, _template_to_regex(t)) for eid, t in _TEMPLATE_TEXTS]

ATTACKY_EVENT_IDS = {'E10', 'E12', 'E13', 'E19', 'E21', 'E27'}


def classify_content(content: str) -> tuple[str, str]:
    """Return (event_id, event_template) for a raw content string,
    or ('E0', content) if nothing matches."""
    for eid, tmpl, rx in _COMPILED_TEMPLATES:
        if rx.match(content):
            return eid, tmpl
    return 'E0', content


# --------------------- timestamp helpers -----------------------------------

def parse_timestamp(date_str: str, day: int, time_str: str, year: int) -> int:
    s = f"{year} {date_str} {int(day):02d} {time_str}"
    dt = datetime.strptime(s, "%Y %b %d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


# --------------------- structured-CSV path (2K sample) ---------------------

def _parse_csv(path: str | Path, year: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Day'] = df['Day'].astype(int)
    df['Pid'] = df['Pid'].astype(int)
    df['timestamp_unix'] = df.apply(
        lambda r: parse_timestamp(r['Date'], r['Day'], r['Time'], year=year),
        axis=1,
    )
    df = df.rename(columns={
        'Pid': 'pid',
        'EventId': 'event_id',
        'EventTemplate': 'event_template',
        'Content': 'raw_content',
    })
    return df


# --------------------- raw-log path (full Loghub OpenSSH) ------------------

def _parse_raw_log(path: str | Path, year: int) -> pd.DataFrame:
    path = Path(path)
    rows = []
    n_skipped = 0
    # Stream the file line-by-line to keep memory in check on 70 MB inputs.
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')
            if not line:
                continue
            m = RAW_LINE_RE.match(line)
            if not m:
                n_skipped += 1
                continue
            content = m.group('content')
            eid, tmpl = classify_content(content)
            try:
                ts = parse_timestamp(m.group('month'), int(m.group('day')),
                                     m.group('time'), year=year)
            except ValueError:
                n_skipped += 1
                continue
            rows.append({
                'timestamp_unix': ts,
                'pid': int(m.group('pid')),
                'raw_content': content,
                'event_id': eid,
                'event_template': tmpl,
            })
    if n_skipped:
        print(f"[parse_openssh] skipped {n_skipped:,} unparseable lines",
              file=sys.stderr)
    return pd.DataFrame(rows)


# --------------------- common post-processing -----------------------------

def _augment(df: pd.DataFrame) -> pd.DataFrame:
    content = df['raw_content'].astype(str)

    df['src_ip'] = content.apply(
        lambda s: (m.group(1) if (m := IP_RE.search(s)) else None)
    )
    df['port'] = content.apply(
        lambda s: (m.group(1) if (m := PORT_RE.search(s)) else None)
    )

    def extract_user(s: str) -> str | None:
        for rx in (FROM_USER_RE, INVALID_USER_RE, PAM_USER_RE):
            m = rx.search(s)
            if m:
                return m.group(1)
        return None

    df['attempted_user'] = content.apply(extract_user)
    df['is_attacky'] = df['event_id'].isin(ATTACKY_EVENT_IDS)

    out = df[[
        'timestamp_unix', 'pid', 'src_ip', 'attempted_user', 'port',
        'event_id', 'event_template', 'raw_content', 'is_attacky',
    ]]
    return out.sort_values('timestamp_unix').reset_index(drop=True)


# ------------------------- unified entry points ---------------------------

def parse_openssh(path: str | Path, year: int = 2015) -> pd.DataFrame:
    """Auto-detect format and parse. Returns the tidy DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.csv':
        df = _parse_csv(path, year=year)
    elif suffix == '.log' or suffix == '.txt':
        df = _parse_raw_log(path, year=year)
    else:
        # Peek at first line: starts with 3-letter month -> raw log,
        # otherwise assume CSV.
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            first = f.readline()
        if RAW_LINE_RE.match(first.strip()):
            df = _parse_raw_log(path, year=year)
        else:
            df = _parse_csv(path, year=year)
    return _augment(df)


# Backward compatibility for callers expecting the old name
def parse_openssh_csv(path: str | Path, year: int = 2015) -> pd.DataFrame:
    return parse_openssh(path, year=year)


# ------------------------------------------------------------- CLI ---------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('path', help='Input log path (.csv structured or .log raw)')
    ap.add_argument('--year', type=int, default=2015)
    ap.add_argument('--out', help='If given, write the augmented dataframe '
                                   'here as CSV')
    args = ap.parse_args()

    df = parse_openssh(args.path, year=args.year)
    print(f"Parsed {len(df):,} events")
    print(f"  events with src_ip        : {df['src_ip'].notna().sum():,}")
    print(f"  events with attempted_user: {df['attempted_user'].notna().sum():,}")
    print(f"  distinct src_ips          : {df['src_ip'].nunique():,}")
    print(f"  distinct event templates  : {df['event_id'].nunique():,}")
    print(f"  is_attacky rate           : {df['is_attacky'].mean()*100:.1f}%")
    print(f"  template coverage:")
    print(df['event_id'].value_counts().head(10).to_string())
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")
