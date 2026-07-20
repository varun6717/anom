# =============================================================================
# run_line_study.py
#
# READ-ONLY STUDY. Pulls real fee history from Snowflake and answers one
# question with measurements instead of assumptions:
#
#     Where should the anomaly threshold ("the line") come from?
#       flat        - one system-wide line for every fee code
#       own         - each SVC_CD's own quantile, from its own diffs
#       tiered@N    - own line when the code has >= N records, else a robust
#                     estimate borrowed from the pool
#       normalized  - diffs rescaled by each code's spread, one pooled line
#
# Writes nothing to Snowflake. Trains no model. Touches no pipeline code.
# Output: console tables + a timestamped .txt transcript + .csv results.
#
# USAGE
#   python run_line_study.py                    # last 57 days ending yesterday
#   python run_line_study.py --start 2026-04-20 --end 2026-06-15
#   python run_line_study.py --dim ACT_CD       # study a different dimension
#   python run_line_study.py --mock             # dry run on generated data
#
# REQUIREMENTS
#   Snowflake access via the same externalbrowser auth anomaly_fab.py uses.
#   A browser session is needed for login, so run this somewhere interactive.
# =============================================================================
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# NOTE: connection identifiers are placeholders. See CONNECTION_AND_QUERY.md.

OUT_DIR = Path(__file__).parent / "line_study_output"

# =============================================================================
# PLACEHOLDERS — replace before running against the real warehouse.
#
# Everything below uses mock identifiers so this file carries no production
# names. The real connection parameters and query live in
# CONNECTION_AND_QUERY.md; paste them in here (or export the env vars) at
# deploy time.
#
# The COLUMN names are real and must not change — the analysis depends on them:
#   STRATUS_TANDEM, MOP_CD, SUBM_DATE, SVC_CD, GEN_TXN_STR, ACT_CD,
#   TRANSACTION_COUNT
# =============================================================================
import os

# vvvvvvvvvvvvvvvv  EDIT BELOW THIS LINE  vvvvvvvvvvvvvvvv
SF_ACCOUNT = os.environ.get("SF_ACCOUNT", "MOCK_ACCOUNT")
SF_ROLE = os.environ.get("SF_ROLE", "MOCK_ROLE")
SF_WAREHOUSE = os.environ.get("SF_WAREHOUSE", "MOCK_WAREHOUSE")
SF_DATABASE = os.environ.get("SF_DATABASE", "MOCK_DATABASE")
SF_SCHEMA = os.environ.get("SF_SCHEMA", "MOCK_SCHEMA")
SF_TABLE = os.environ.get("SF_TABLE", "MOCK_DATABASE.MOCK_SCHEMA.MOCK_FEE_TABLE")
SF_COUNT_COL = os.environ.get("SF_COUNT_COL", "mock_fee_cnt")

HISTORY_SQL = """
SELECT
    STRATUS_TANDEM,
    MOP_CD,
    TO_CHAR(SUBM_DATE, 'MM-DD-YYYY') AS SUBM_DATE,
    SVC_CD,
    GEN_TXN_STR,
    ACT_CD,
    SUM({count_col}) AS TRANSACTION_COUNT
FROM
    {table}
WHERE
    SVC_CD IS NOT NULL AND SVC_CD != 'NONE'
    AND GEN_TXN_STR IS NOT NULL AND GEN_TXN_STR != 'NONE'
    AND ACT_CD IS NOT NULL AND ACT_CD != 'NONE'
    AND SUBM_DATE BETWEEN TO_TIMESTAMP('{start} 00:00:00', 'YYYY-MM-DD HH24:MI:SS')
                      AND TO_TIMESTAMP('{end} 23:59:59', 'YYYY-MM-DD HH24:MI:SS')
GROUP BY
    STRATUS_TANDEM, MOP_CD, SUBM_DATE, SVC_CD, GEN_TXN_STR, ACT_CD
"""
# ^^^^^^^^^^^^^^^^  EDIT ABOVE THIS LINE  ^^^^^^^^^^^^^^^^
# Nothing below needs changing. The SELECT must keep emitting these exact
# column names — alias in the query if the source table differs:
#   STRATUS_TANDEM, MOP_CD, SUBM_DATE, SVC_CD, GEN_TXN_STR, ACT_CD,
#   TRANSACTION_COUNT
# The {start} / {end} / {table} / {count_col} braces are filled at runtime.


# -----------------------------------------------------------------------------
# Snowflake read connection.
#
# DELIBERATELY duplicated from anomaly_fab.py rather than imported: that module
# executes its whole pipeline at import time (no __main__ guard) — both queries,
# 200 epochs of training per system, and write_to_snowflake(). Importing from it
# would trigger a production write. Keep this standalone.
# -----------------------------------------------------------------------------
def execute_sql_read_only(sql_query):
    from sqlalchemy import create_engine

    if SF_ACCOUNT == "MOCK_ACCOUNT":
        raise SystemExit(
            "\nSnowflake identifiers are still placeholders.\n"
            "Fill them in from CONNECTION_AND_QUERY.md — either edit the\n"
            "SF_* constants at the top of this file, or export the env vars:\n"
            "   export SF_ACCOUNT=...  SF_ROLE=...  SF_WAREHOUSE=...\n"
            "   export SF_DATABASE=... SF_SCHEMA=... SF_TABLE=... SF_COUNT_COL=...\n"
            "Or run with --mock to test the pipeline without a warehouse.\n"
        )

    user = os.environ.get("USER") or os.environ.get("USERNAME")
    engine = create_engine(
        f"snowflake://{user}:@{SF_ACCOUNT}/?database={SF_DATABASE}&schema={SF_SCHEMA}"
        f"&warehouse={SF_WAREHOUSE}&role={SF_ROLE}&authenticator=externalbrowser"
    )
    result = pd.read_sql_query(sql_query, engine)
    result.columns = [c.upper() for c in result.columns]
    return result


class Tee:
    """Mirror stdout to a transcript file so the run is reviewable later."""

    def __init__(self, path):
        self.file = open(path, 'w')
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def load_history(start, end, use_mock=False):
    if use_mock:
        print("[MOCK MODE] generating synthetic history from anomaly_dev.py")
        import io
        import contextlib
        src = open(Path(__file__).parent / 'anomaly_dev.py').read()
        ns = {'__file__': str(Path(__file__).parent / 'anomaly_dev.py')}
        exec(src[:src.index('merchant_id_column_name =')], ns)
        with contextlib.redirect_stdout(io.StringIO()):
            history, _ = ns['generate_mock_data']()
        return history

    sql = HISTORY_SQL.format(start=start, end=end,
                             table=SF_TABLE, count_col=SF_COUNT_COL)
    print(f"querying {SF_TABLE} for {start} .. {end}")
    print("(a browser window may open for authentication)")
    return execute_sql_read_only(sql)


def describe(history):
    print(f"\n{'='*72}\nDATA PULLED\n{'='*72}")
    print(f"  rows              : {len(history):,}")
    print(f"  systems           : {sorted(history.STRATUS_TANDEM.unique())}")
    print(f"  distinct days     : {history.SUBM_DATE.nunique()}")
    for col in ['MOP_CD', 'SVC_CD', 'GEN_TXN_STR', 'ACT_CD']:
        print(f"  {col:16s}: {history[col].nunique():,} distinct values")
    labels = ['MOP_CD', 'SVC_CD', 'GEN_TXN_STR', 'ACT_CD']
    for system, sdf in history.groupby('STRATUS_TANDEM'):
        perms = sdf.groupby(labels).ngroups
        days = sdf.groupby(labels).size()
        print(f"\n  {system}: {len(sdf):,} rows, {perms:,} permutations")
        print(f"     days per permutation: median {int(days.median())}, "
              f"{(days < 10).mean()*100:.0f}% have <10 days")
        bucket = sdf.groupby('SVC_CD').size()
        print(f"     records per SVC_CD  : median {int(bucket.median()):,}, "
              f"min {bucket.min():,}, max {bucket.max():,}")
        # the sample-size question that decides whether per-code lines are viable
        viable = (bucket >= 2000).sum()
        print(f"     SVC codes with >=2000 records (enough for a private 99.5th "
              f"pct): {viable} of {len(bucket)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    default_end = date.today() - timedelta(days=1)
    default_start = default_end - timedelta(days=56)
    ap.add_argument('--start', default=default_start.isoformat())
    ap.add_argument('--end', default=default_end.isoformat())
    ap.add_argument('--dim', default='SVC_CD',
                    help='dimension whose line-setting we study (default SVC_CD)')
    ap.add_argument('--quantile', type=float, default=0.995)
    ap.add_argument('--min-test', type=int, default=100, dest='min_test',
                    help='min held-out records for a group to be scored. Lower it '
                         '(e.g. 30) to include smaller groups — noisier, but it '
                         'reveals how strategies behave on the long tail that the '
                         'default threshold excludes entirely.')
    ap.add_argument('--measure-at', default='SVC_CD', dest='measure_at',
                    help='level at which calibration is judged in the '
                         'cross-evaluation (default SVC_CD — where billing '
                         'issues manifest). Line sources are compared on this '
                         'single yardstick so coarser groupings get no unfair '
                         'advantage.')
    ap.add_argument('--emit-quantile', type=float, default=None, dest='emit_q',
                    help='OVERRIDE the quantile the recommender chose. Omit to let '
                         'the study decide (the normal path).')
    ap.add_argument('--force-dim', default=None, dest='dim_override',
                    help='OVERRIDE the grouping dimension the recommender chose. '
                         'Omit to let the study decide (the normal path).')
    ap.add_argument('--cap-multiplier', type=float, default=1.0, dest='cap_mult',
                    help='cap groups at this multiple of the pooled line '
                         '(1.0 = exactly the pooled line; higher = looser cap, '
                         'fewer false alarms on the noisiest group but bigger '
                         'blind spots)')
    ap.add_argument('--min-catch-fold', type=float, default=5.0, dest='min_catch',
                    help='the smallest spike the business must never miss. The '
                         'recommender picks the loosest quantile whose WORST '
                         'group still catches a move this big (default 5x).')
    ap.add_argument('--min-records', type=int, default=100, dest='min_records',
                    help='a group needs this many historical records before its '
                         'OWN threshold is trusted. Below it, the pooled line is '
                         'used — a 3-record group otherwise gets a ~1.04x line '
                         'and fires on any day (default 100).')
    ap.add_argument('--yes', action='store_true', dest='assume_yes',
                    help='skip the confirmation prompt and write the config '
                         '(for scheduled/non-interactive runs)')
    ap.add_argument('--mock', action='store_true', help='dry run on synthetic data')
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    stamp = (f"{args.start}_to_{args.end}_{args.dim}_min{args.min_test}"
             + ("_MOCK" if args.mock else ""))
    tee = Tee(OUT_DIR / f"line_study_{stamp}.txt")
    sys.stdout = tee

    try:
        print(f"LINE-SETTING STUDY   window {args.start} .. {args.end}   dim={args.dim}")
        history = load_history(args.start, args.end, args.mock)
        if history.empty:
            print("\nNO ROWS RETURNED — check the date window against the source table.")
            return
        describe(history)

        from wobble_advisor import (compare_strategies, cross_evaluate,
                                    detectability, emit_calibration,
                                    quantile_sweep, recommend,
                                    recommend_config, validate_policy,
                                    volume_profile)

        print(f"\n{'='*72}\nPART 1 — WHICH LINE-SETTING STRATEGY CALIBRATES BEST?"
              f"\n{'='*72}")
        print("all strategies compute diffs per PERMUTATION; they differ only in")
        print("where the LINE comes from. evaluation is OUT-OF-SAMPLE.\n")
        print("read the three metrics together:")
        print("  in_band%  fairness  - share of groups near the intended alert rate")
        print("  worstFA%  tail risk - the single worst-calibrated group")
        print("  deaf      coverage  - groups where NOTHING ever crosses (catch nothing)")
        strat = compare_strategies(history, dim=args.dim, quantile=args.quantile,
                                   min_test=args.min_test)
        for system, table in strat.items():
            table.to_csv(OUT_DIR / f"strategies_{system}_{stamp}.csv", index=False)

        print(f"\n{'='*72}\nPART 1b — WHICH GROUPING SHOULD SUPPLY THE LINE?\n{'='*72}")
        print("Part 1 scored each grouping at its OWN level, which favours coarse")
        print("groupings unfairly (bigger buckets average out extremes). Here every")
        print(f"line source is judged on ONE yardstick: calibration at {args.measure_at}.")
        print("Detection is always per permutation; this only sets how false-alarm")
        print("rates are aggregated for scoring.\n")
        cross = cross_evaluate(history, measure_at=args.measure_at,
                               quantile=args.quantile, min_test=args.min_test)
        for system, table in cross.items():
            table.to_csv(OUT_DIR / f"crosseval_{system}_{stamp}.csv", index=False)

        print(f"\n{'='*72}\nPART 1c — HOW BIG A SPIKE DOES EACH GROUP NEED TO FLAG?"
              f"\n{'='*72}")
        print("Parts 1/1b score FALSE ALARMS on normal days — that is calibration,")
        print("not detection. This converts each strategy's thresholds into volume")
        print("multiples, which is what actually decides whether a real spike is")
        print("caught. A group that never fires on normal days is fine if its")
        print("threshold sits at 4x; a group needing 60x is the real blind spot.\n")
        det = detectability(history, dim=args.dim, quantile=args.quantile)

        print(f"\n{chr(61)*72}\nPART 1d — IS VOLUME THE REAL DRIVER OF THE THRESHOLD?\n{chr(61)*72}")
        print("fold-change = today / that perm's typical day (the global std cancels).")
        print("So a 7x threshold means normal days routinely swing 7x — which is what")
        print("tiny permutations do from Poisson noise alone. If high-volume perms are")
        print("far steadier, they are being judged by a threshold set by 2-txn-a-day")
        print("perms, and volume matters more than fee code.\n")
        vol = volume_profile(history, quantile=args.quantile)
        for system, table in vol.items():
            table.to_csv(OUT_DIR / f"volume_{system}_{stamp}.csv", index=False)
        for system, table in det.items():
            table.to_csv(OUT_DIR / f"detectability_{system}_{stamp}.csv", index=False)

        print(f"\n{'='*72}\nPART 1e — WHAT OPERATING POINT DO WE WANT?\n{'='*72}")
        print("Strategy is settled by 1b/1c; the quantile is the remaining dial, and")
        print("it is a business call. Lower quantile = tighter threshold = smaller")
        print("spikes caught, more alerts raised. Pick the row you can staff.\n")
        sweep = quantile_sweep(history, dim=args.dim)
        for system, table in sweep.items():
            table.to_csv(OUT_DIR / f"operating_point_{system}_{stamp}.csv", index=False)

        print(f"\n{'='*72}\nPART 2 — IS ANY DIMENSION WORTH NORMALIZING BY?\n{'='*72}")
        rec = recommend(history, test_all=True)

        print(f"\n{'='*72}\nPART 1f — RECOMMENDED CONFIG (with confidence)\n{'='*72}")
        print("Chains gates 1 -> cross-eval -> detectability -> quantile sweep into")
        print("one decision. Confidence is broken into components so you can see WHY")
        print("it is high or low. A dimension must BEAT flat pooling, not merely")
        print("qualify on heterogeneity.\n")
        reco = recommend_config(history, measure_at=args.measure_at,
                                min_catch_fold=args.min_catch)
        for system, r in reco.items():
            if r.get('group_dim'):
                print(f"\n    to emit this config:")
                print(f"      python run_line_study.py --start {args.start} --end {args.end} \\")
                print(f"             --dim {r['group_dim']} --emit-quantile {r['quantile']}")

        print(f"\n{'='*72}\nPART 1g — DOES THE POLICY SERVE THIN GROUPS?\n{'='*72}")
        print("Every other part scores idealised strategies. This scores the policy")
        print("as it will actually run, broken out by how each line was derived.")
        print("About half of all fee codes fall back to the pooled line; this checks")
        print("whether that fallback actually serves them.\n")
        _first = next(iter(reco.values())) if reco else {}
        validate_policy(history, dim=(args.dim_override or _first.get('group_dim') or args.dim),
                        quantile=(args.emit_q or _first.get('quantile') or args.quantile),
                        cap_multiplier=args.cap_mult, min_records=args.min_records)

        # ---- confirm, then emit ----------------------------------------
        # The study stops here and asks. Everything needed to decide is already
        # on screen; answering YES writes the config and finishes the run, so a
        # full calibration is one command plus one word.
        import json
        first = next(iter(reco.values())) if reco else {}
        chosen_dim = args.dim_override or first.get('group_dim')
        chosen_q = args.emit_q or first.get('quantile') or args.quantile
        overridden = bool(args.dim_override or args.emit_q)

        print(f"\n{'='*72}\nCONFIRM CALIBRATION\n{'='*72}")
        print(f"  proposed config")
        print(f"     group_dim    : {chosen_dim if chosen_dim else 'None (flat pooling)'}")
        print(f"     quantile     : {chosen_q}")
        print(f"     cap          : {args.cap_mult}x the pooled line")
        print(f"     min_records  : {args.min_records} (below this a group uses the pooled line)")
        if overridden:
            print(f"     source       : OVERRIDDEN on the command line")
        elif first.get('confidence') is not None:
            print(f"     confidence   : {first['confidence']}"
                  + ("   << LOW — review Parts 1b/1c first" if first['confidence'] < 0.6 else ""))
        if first.get('alerts_per_day') is not None:
            print(f"     expected load: {first['alerts_per_day']} alerts/day; typical group "
                  f"flags at {first.get('median_fold')}x, worst at {first.get('worst_fold')}x")
        print()

        if args.assume_yes:
            answer, how = 'yes', '(--yes)'
        else:
            try:
                print("  Write this calibration and finish?  [YES / no]")
                answer, how = input("  > ").strip().lower(), ''
            except EOFError:
                answer, how = 'no', '(no terminal — rerun interactively or pass --yes)'

        if answer not in ('y', 'yes', 'process', 'ok', 'go'):
            print(f"\n  not written {how}")
            print(f"  to write it later without re-running the analysis:")
            print(f"     python {Path(__file__).name} --start {args.start} --end {args.end} \\")
            print(f"            --force-dim {chosen_dim} --emit-quantile {chosen_q} --yes")
            return reco

        print(f"\n{'='*72}\nCALIBRATION CONFIG\n{'='*72}")
        cfg = emit_calibration(history, dim=chosen_dim, quantile=chosen_q,
                               cap_multiplier=args.cap_mult,
                               min_records=args.min_records,
                               window={'start': args.start, 'end': args.end})
        cfg['recommendation'] = reco
        stamped = OUT_DIR / f"calibration_{chosen_dim}_q{chosen_q}_{stamp}.json"
        latest = OUT_DIR / "calibration_latest.json"
        stamped.write_text(json.dumps(cfg, indent=2))
        latest.write_text(json.dumps(cfg, indent=2))
        print(f"    written to {stamped.name}")
        print(f"    and to    {latest.name}   <- the pipeline reads THIS path")

        print(f"\n{'='*72}\nWHAT TO DO WITH THIS\n{'='*72}")
        print("1. PART 1 picks the line-setting strategy. If one strategy wins all")
        print("   three metrics, take it. If not, weight `deaf` most heavily — a deaf")
        print("   group has zero detection capability, which is worse than a noisy one.")
        print("2. PART 2 sets WOBBLE_DIM / NORMALIZER. 'none' means flat pooling,")
        print("   which is the simplest and has no contamination surface.")
        print("3. Neither part measures whether REAL anomalies get caught — both score")
        print("   false alarms on normal days only. The injection harness (C-6 in")
        print("   REVIEW_OVERVIEW_FEEDBACK.md) is what settles detection power.")
        print(f"\nresults written to {OUT_DIR}/")
        return rec
    finally:
        sys.stdout = tee.stdout
        tee.file.close()


if __name__ == '__main__':
    main()
