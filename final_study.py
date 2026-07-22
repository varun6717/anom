"""
final_study.py — the §5c FINAL REVIEW TASK (REVIEW_OVERVIEW_FEEDBACK.md).

Eight measurements against REAL production history, run ONCE after the change
review, whose answers set the initial toggle positions (see the §5c toggle
inventory). READ-ONLY: one aggregate probe + one windowed SELECT; no writes, no
model training, no production side effects.

    python final_study.py --mock            # dry run on mock_v4 artifacts
    python final_study.py --days 90         # real pull (browser auth)

Questions (→ decides):
  1  center estimator sweep, groupby vs additive  → center_source (+ hybrid N)
  2  presence / gap distribution                  → C-2 threshold, C-2a statistic
  3  fee sign & zeros                             → C-26 ④ abs / sign-flip policy
  4  history depth available                      → history_days
  5  new permutations per day                     → C-15 ranking importance
  6  warehouse latency                            → C-5 --as-of default
  7  partition overlap                            → C-25 thin-group cost
  8  weekday effect size + stability              → weekday_adjust
"""
import argparse
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PERM = ['MOP_CD', 'SVC_CD', 'GEN_TXN_STR', 'ACT_CD']
PART = 'STRATUS_TANDEM'
SF_FEE_COL = os.environ.get("SF_FEE_COL", "")          # optional; q3 skipped if unset

DEPTH_PROBE_SQL = """
SELECT MIN(SUBM_DATE) AS FIRST_DAY, MAX(SUBM_DATE) AS LAST_DAY,
       COUNT(DISTINCT SUBM_DATE) AS N_DAYS
FROM {table}
WHERE SUBM_DATE >= DATEADD(day, -400, CURRENT_DATE)
"""

MAIN_SQL = """
SELECT
    STRATUS_TANDEM, MOP_CD, TO_CHAR(SUBM_DATE, 'MM-DD-YYYY') AS SUBM_DATE,
    SVC_CD, GEN_TXN_STR, ACT_CD,
    SUM({count_col}) AS TRANSACTION_COUNT{fee_select}
FROM {table}
WHERE SVC_CD IS NOT NULL AND SVC_CD != 'NONE'
  AND GEN_TXN_STR IS NOT NULL AND GEN_TXN_STR != 'NONE'
  AND ACT_CD IS NOT NULL AND ACT_CD != 'NONE'
  AND SUBM_DATE >= DATEADD(day, -{days}, CURRENT_DATE)
GROUP BY STRATUS_TANDEM, MOP_CD, SUBM_DATE, SVC_CD, GEN_TXN_STR, ACT_CD
"""


def load(args):
    if args.mock:
        print("[MOCK] reading dev_output/mock_v4_history_split.parquet")
        h = pd.read_parquet(Path(__file__).parent / "dev_output/mock_v4_history_split.parquet")
        return h, None
    from run_line_study import execute_sql_read_only, SF_TABLE, SF_COUNT_COL
    fee = f",\n    SUM({SF_FEE_COL}) AS FEE_AMOUNT" if SF_FEE_COL else ""
    probe = execute_sql_read_only(DEPTH_PROBE_SQL.format(table=SF_TABLE))
    h = execute_sql_read_only(MAIN_SQL.format(
        table=SF_TABLE, count_col=SF_COUNT_COL, fee_select=fee, days=args.days))
    return h, probe


def hdr(n, title):
    print(f"\n{'='*74}\nQ{n} — {title}\n{'='*74}")


def q1_center_sweep(h):
    hdr(1, "CENTER ESTIMATOR SWEEP  →  center_source")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import OneHotEncoder
    verdicts = []
    for part, g in h.groupby(PART):
        g = g.copy()
        L = np.log1p(g.TRANSACTION_COUNT)
        g['z'] = (L - L.mean()) / L.std()
        days = sorted(g.d.unique())
        test = g[g.d >= days[len(days)*2//3]]
        truth = test.groupby(PERM).z.mean()
        print(f"\n  {part}:  train pool {len(days)*2//3} days, test {len(days)-len(days)*2//3} days")
        print(f"  {'train days':>11}{'rows/perm':>11}{'groupby MAE':>13}"
              f"{'additive MAE':>14}{'winner':>10}")
        flat_probe = []
        for nd in [1, 2, 3, 5, 7, 10, 14, len(days)*2//3]:
            tr = g[g.d < days[min(nd, len(days)*2//3)]]
            if tr.empty:
                continue
            enc = OneHotEncoder(handle_unknown='ignore')
            mdl = Ridge(alpha=1.0).fit(enc.fit_transform(tr[PERM]), tr.z)
            plain = tr.groupby(PERM).z.mean()
            pt = tr.groupby(PERM).size().reset_index()[PERM]
            pred = pd.Series(mdl.predict(enc.transform(pt)),
                             index=pd.MultiIndex.from_frame(pt))
            common = truth.index.intersection(plain.index).intersection(pred.index)
            if len(common) < 50:
                continue
            a = float(np.abs(plain[common] - truth[common]).mean())
            b = float(np.abs(pred[common] - truth[common]).mean())
            rpp = len(tr) / tr.groupby(PERM).ngroups
            flat_probe.append(b)
            print(f"  {nd:>11}{rpp:>11.1f}{a:>13.4f}{b:>14.4f}"
                  f"{'groupby' if a < b else 'ADDITIVE':>10}")
            verdicts.append(a < b)
        if len(flat_probe) >= 3:
            spread = max(flat_probe) - min(flat_probe)
            print(f"  additive-MAE spread across windows: {spread:.4f}  "
                  f"({'FLAT — labels carry little signal' if spread < 0.05 else 'declining — real structure'})")
    if all(verdicts):
        print("\n  → PROPOSAL: center_source: groupby   (groupby won at every depth;")
        print("    NN remains attribution + novelty + features per the Phase 1 scope)")
    else:
        print("\n  → crossover detected — apply the agreed decision rule (≥5% of flag")
        print("    decisions moved) to set hybrid N; inspect the table above.")


def q2_presence_gaps(h):
    hdr(2, "PRESENCE / GAPS  →  C-2 threshold, C-2a statistic")
    for part, g in h.groupby(PART):
        days = sorted(g.d.unique()); idx = {d: i for i, d in enumerate(days)}
        pres = g.groupby(PERM).d.nunique()
        n = len(days)
        print(f"\n  {part}: {len(pres)} perms over {n} days")
        print(f"    median presence {pres.median():.0f}/{n} | full {(pres==n).mean()*100:.1f}% "
              f"| <10 days {(pres<10).mean()*100:.1f}%")
        print(f"    candidate Drop-1 thresholds:")
        for miss in [0, 1, 2, 3]:
            print(f"      missed ≤{miss} days  →  {(pres>=n-miss).sum():>5} perms "
                  f"({(pres>=n-miss).mean()*100:.1f}%) eligible")
        gaps_per_perm = []
        for _, rows in g.groupby(PERM):
            p = sorted({idx[d] for d in rows.d})
            gp = [b-a-1 for a, b in zip(p, p[1:]) if b-a-1 > 0]
            gaps_per_perm.append(len(gp))
        gp = pd.Series(gaps_per_perm)
        print(f"    perms with ≥3 completed gaps (own tolerance estimable): "
              f"{(gp>=3).mean()*100:.1f}%  |  0 gaps (earned tol 0): {(gp==0).mean()*100:.1f}%")


def q3_fee(h):
    hdr(3, "FEE SIGN & ZEROS  →  C-26 ④")
    if 'FEE_AMOUNT' not in h.columns:
        print("  FEE column not in pull (set SF_FEE_COL) — SKIPPED, decide ④ later")
        return
    neg = h[h.FEE_AMOUNT < 0]
    print(f"  negatives: {len(neg):,} rows ({len(neg)/len(h)*100:.2f}%)"
          + (f" — by ACT_CD: {neg.ACT_CD.value_counts().head(6).to_dict()}" if len(neg) else ""))
    print(f"  zeros    : {(h.FEE_AMOUNT==0).sum():,} rows")
    sign = h.assign(s=np.sign(h.FEE_AMOUNT)).groupby(PERM).s.nunique()
    print(f"  sign constant within perm: {(sign<=1).mean()*100:.1f}% of perms "
          f"({(sign>1).sum()} mixed-sign perms)")
    print("  → abs+sign-flip policy holds only if mixed-sign perms ≈ 0; else separate streams")


def q4_depth(probe, h):
    hdr(4, "HISTORY DEPTH  →  history_days")
    if probe is None:
        print(f"  [MOCK] window in artifact: {h.d.nunique()} days — probe needs the real table")
        return
    r = probe.iloc[0]
    print(f"  table holds {r.N_DAYS} distinct days: {r.FIRST_DAY} → {r.LAST_DAY}")
    print(f"  → if ≥90: raise history_days; offsets perm-thinning from added fields")


def q5_new_perms(h):
    hdr(5, "NEW PERMUTATIONS PER DAY  →  C-15 ranking importance")
    for part, g in h.groupby(PART):
        first = g.groupby(PERM).d.min()
        days = sorted(g.d.unique())
        warm = days[min(13, len(days)-1)]
        per_day = first[first > warm].groupby(first[first > warm]).size()
        print(f"  {part}: after a 14-day warm-up, new perms/day "
              f"mean {per_day.mean():.1f}, max {per_day.max() if len(per_day) else 0}"
              f"  ({'human-readable pile' if per_day.mean() < 10 else 'RANKING ESSENTIAL'})")


def q6_latency(h, is_mock):
    hdr(6, "WAREHOUSE LATENCY  →  --as-of default")
    if is_mock:
        print("  [MOCK] n/a — run against the real table")
        return
    latest = h.d.max().date()
    lag = (date.today() - latest).days
    print(f"  newest SUBM_DATE = {latest}  ({lag} day(s) behind today)")
    print(f"  → default --as-of {'CURRENT_DATE - 1 is safe' if lag <= 1 else f'must derive from MAX(SUBM_DATE); fixed offset unsafe (lag {lag}d)'}")


def q7_overlap(h):
    hdr(7, "PARTITION OVERLAP  →  C-25 thin-group cost")
    parts = h[PART].unique()
    if len(parts) < 2:
        print(f"  single partition value {list(parts)} — re-run after the prod switch")
        return
    sets = {p: set(map(tuple, h[h[PART] == p][PERM].drop_duplicates().values)) for p in parts}
    inter = set.intersection(*sets.values()); union = set.union(*sets.values())
    print(f"  perms in ALL partitions: {len(inter):,} of {len(union):,} "
          f"({len(inter)/len(union)*100:.1f}% overlap)")
    for p, s in sets.items():
        print(f"    {p}: {len(s):,} perms ({len(s - inter):,} exclusive)")


def q8_weekday(h):
    hdr(8, "WEEKDAY EFFECT  →  weekday_adjust")
    for part, g in h.groupby(PART):
        g = g.copy()
        L = np.log1p(g.TRANSACTION_COUNT)
        g['z'] = (L - L.mean()) / L.std()
        g['dc'] = g.z - g.groupby(PERM).z.transform('mean')   # de-centered: weekday residual
        g['wd'] = g.d.dt.day_name().str[:3]
        days = sorted(g.d.unique()); mid = days[len(days)//2]
        eff = g.groupby('wd').dc.mean()
        h1 = g[g.d < mid].groupby('wd').dc.mean(); h2 = g[g.d >= mid].groupby('wd').dc.mean()
        both = pd.concat([h1, h2], axis=1).dropna()
        r = both.corr(method='spearman').iloc[0, 1] if len(both) >= 5 else float('nan')
        span = float(np.exp((eff.max() - eff.min()) * L.std()))
        print(f"  {part}: weekday span {span:.2f}x (quietest→busiest) | "
              f"split-half stability r={r:.2f} | "
              f"{'→ pooled' if span >= 1.15 and r >= 0.6 else '→ none'}")
        print(f"    residual by weekday: "
              + "  ".join(f"{k}:{v:+.3f}" for k, v in eff.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mock', action='store_true')
    ap.add_argument('--days', type=int, default=90, help='real-pull window (default 90)')
    args = ap.parse_args()
    h, probe = load(args)
    h = h.copy()
    h['d'] = pd.to_datetime(h.SUBM_DATE, format='%m-%d-%Y')
    print(f"\n{'='*74}\n§5c FINAL STUDY — {len(h):,} rows | partitions "
          f"{sorted(h[PART].unique())} | {h.d.nunique()} days\n{'='*74}")
    q1_center_sweep(h)
    q2_presence_gaps(h)
    q3_fee(h)
    q4_depth(probe, h)
    q5_new_perms(h)
    q6_latency(h, args.mock)
    q7_overlap(h)
    q8_weekday(h)
    print(f"\n{'='*74}\nDONE — carry these answers into calibration recipe + input spec"
          f"\n(toggle inventory: REVIEW_OVERVIEW_FEEDBACK.md §5c)\n{'='*74}")


if __name__ == '__main__':
    main()
