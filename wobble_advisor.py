# =============================================================================
# wobble_advisor.py — step-zero diagnostic for the WOBBLE_DIM config flag
#
# Answers: "which categorical dimension (if any) should serve as the wobble-unit
# denominator in the pooled-diff threshold (C-10 / §5 A7)?"
#
# Method, per candidate dimension:
#   1. Compute per-perm diffs (§5 steps A1-A6) from a history DataFrame.
#   2. Bucket the diffs by the candidate dimension; per-bucket median = wobble.
#   3. HETEROGENEITY: ratio of max/min bucket wobble (buckets with n >= MIN_N).
#      A dimension only matters if fee groups genuinely differ (>= MIN_RATIO).
#   4. STABILITY: split each perm's days into chronological halves, recompute
#      bucket wobbles independently per half, Spearman-correlate across buckets.
#      Real heterogeneity reproduces itself (r >= MIN_STABILITY); sampling noise
#      doesn't.
#   5. Recommend the qualifying dimension with the largest stable ratio, else
#      None (flat system pool).
#
# Usage (dev): python wobble_advisor.py            <- runs on the mock generator
# Usage (prod): from wobble_advisor import advise; advise(history_df)
#   where history_df has columns STRATUS_TANDEM, MOP_CD, SVC_CD, GEN_TXN_STR,
#   ACT_CD, SUBM_DATE, TRANSACTION_COUNT (the historical_sql_code contract).
#
# Read-only diagnostic: makes no change to any model or pipeline behavior.
# =============================================================================
import numpy as np
import pandas as pd

LABELS = ['MOP_CD', 'SVC_CD', 'GEN_TXN_STR', 'ACT_CD']
CANDIDATES = ['MOP_CD', 'SVC_CD', 'ACT_CD']  # GEN_TXN_STR ~ perm-level; excluded
MIN_N = 30            # bucket must have >= this many diffs for a trusted median
MIN_RATIO = 1.5       # max/min bucket wobble must exceed this to matter
MIN_STABILITY = 0.6   # split-half Spearman r must exceed this to be believed
MIN_GAIN = 10         # gate 2: normalizer must add >= this many in-band points
MAX_CONTAM = 25       # gate 2: max % denominator inflation from one anomaly


def _diffs(df):
    """§5 steps A1-A6: log1p -> global z -> per-perm center -> |z - center|."""
    d = df.copy()
    lg = np.log1p(d['TRANSACTION_COUNT'])
    d['z'] = (lg - lg.mean()) / lg.std()
    d['center'] = d.groupby(LABELS)['z'].transform('mean')
    d['diff'] = (d['z'] - d['center']).abs()
    return d


def _bucket_wobble(d, dim, min_n=MIN_N):
    g = d.groupby(dim)['diff'].agg(n='size', wobble='median')
    return g[g['n'] >= min_n]


def _winsor_std(x, q=0.95):
    return x.clip(upper=x.quantile(q)).std()


NORMALIZERS = {
    'none':          None,
    'median':        lambda x: x.median(),          # retracted: does not align tails
    'MAD':           lambda x: (x - x.median()).abs().median(),
    'IQR':           lambda x: x.quantile(.75) - x.quantile(.25),
    'std':           lambda x: x.std(),             # best comparability, contamination-fragile
    'winsor95_std':  _winsor_std,                   # robust compromise
}


def compare_normalizers(history_df, dim='SVC_CD', quantile=0.995, min_n=200, verbose=True):
    """
    Does normalizing diffs by a per-`dim` dispersion statistic actually make the
    tails comparable? Reports, per candidate normalizer:
      - tail spread : max/min of per-bucket normalized `quantile` (1.0 = perfect)
      - FA band     : % of buckets whose false-alarm rate under the single shared
                      line lands in [0.5x, 2x] of target (the operational test)
      - contamination: how much one injected 20x anomaly inflates the denominator
    Recommends 'none' unless a normalizer materially beats it.
    """
    out = {}
    target = (1 - quantile) * 100
    for system, sdf in history_df.groupby('STRATUS_TANDEM'):
        d = _diffs(sdf)
        sizes = d.groupby(dim).size()
        big = sizes[sizes >= min_n].index
        b = d[d[dim].isin(big)]
        dirty = d.copy()
        dirty.loc[dirty.groupby(dim).head(1).index, 'diff'] *= 20

        rows = []
        for name, fn in NORMALIZERS.items():
            if fn is None:
                den = pd.Series(1.0, index=sizes.index)
                infl = 0.0
            else:
                den = d.groupby(dim)['diff'].apply(fn)
                den_c = dirty.groupby(dim)['diff'].apply(fn)
                infl = float(((den_c - den) / den).loc[big].max() * 100)
            den = den.replace(0, np.nan)
            line = (d['diff'] / d[dim].map(den)).quantile(quantile)
            nz = b['diff'] / b[dim].map(den)
            fa = nz.groupby(b[dim]).apply(lambda v: (v > line).mean() * 100)
            tails = nz.groupby(b[dim]).quantile(quantile)
            in_band = ((fa >= target / 2) & (fa <= target * 2)).mean() * 100
            rows.append((name, round(float(tails.max() / tails.min()), 1),
                         round(float(fa.median()), 2), round(float(fa.max()), 2),
                         round(in_band), round(infl, 1)))

        t = pd.DataFrame(rows, columns=['normalizer', 'tail_spread', 'FA_median%',
                                        'FA_max%', 'in_band%', 'contam_infl%'])
        flat = t[t.normalizer == 'none'].iloc[0]
        # require a real gain AND contamination resistance
        elig = t[(t.normalizer != 'none') & (t['in_band%'] >= flat['in_band%'] + MIN_GAIN)
                 & (t['contam_infl%'] <= MAX_CONTAM)]
        if len(elig):
            best = elig.sort_values('in_band%').iloc[-1]
            rec, gain = best['normalizer'], best['in_band%'] - flat['in_band%']
        else:
            rec, gain = 'none', 0
        out[system] = {'normalizer': rec, 'gain': gain,
                       'flat_in_band': flat['in_band%'], 'table': t}
        if verbose:
            print(f"\n=== GATE 2 — {system}: normalizer comparison on {dim} "
                  f"({len(big)} buckets >= {min_n} rows) ===")
            print(t.to_string(index=False))
            print(f"--> best for {dim}: {rec}"
                  f"{f' (+{gain:.0f} pts over flat)' if rec != 'none' else ' — no normalizer clears the bar'}")
    return out


def compare_strategies(history_df, dim='SVC_CD', quantile=0.995,
                       cutoffs=(200, 500, 1000, 2000), min_test=100, verbose=True):
    """
    THE decision test: which line-setting strategy calibrates best?

    Strategies compared (all compute diffs per PERMUTATION; they differ only in
    where the LINE comes from):
      flat            one system-wide line for every group
      own             each group's own `quantile` of its own diffs
      tiered@N        own line if the group has >= N records, else robust
                      estimate (pooled multiplier x that group's winsor-std)
      normalized      every diff divided by its group's winsor-std, one
                      pooled line over the normalized values

    Evaluation is OUT-OF-SAMPLE: lines are fitted on alternating days (half 0)
    and scored on the held-out days (half 1). This matters — a group's own
    quantile scores a perfect rate on its own fitting data by construction, so
    in-sample numbers systematically flatter per-group strategies.

    Reported per strategy:
      in_band%  share of groups whose false-alarm rate lands in [target/2, 2*target]
      worstFA%  the single worst-calibrated group (tail risk)
      deaf      groups where NOTHING crossed the line (zero detection capability)

    Returns a DataFrame per system.
    """
    target = (1 - quantile) * 100
    out = {}
    for system, sdf in history_df.groupby('STRATUS_TANDEM'):
        d = _diffs(sdf).sort_values('SUBM_DATE')
        d['_half'] = d.groupby(LABELS).cumcount() % 2
        fit, test = d[d._half == 0], d[d._half == 1]
        full_n = d.groupby(dim).size()
        ntest = test.groupby(dim).size()
        ev = ntest[ntest >= min_test].index
        if len(ev) < 2:
            continue

        flat = fit['diff'].quantile(quantile)
        own = fit.groupby(dim)['diff'].quantile(quantile)
        den = fit.groupby(dim)['diff'].apply(_winsor_std).replace(0, np.nan)
        mult = (fit['diff'] / fit[dim].map(den)).quantile(quantile)

        def score(line_of, label):
            fa = pd.Series({g: (test[test[dim] == g]['diff'] > line_of(g)).mean() * 100
                            for g in ev})
            fa = fa.dropna()
            return {'strategy': label,
                    'in_band%': round(((fa >= target / 2) & (fa <= target * 2)).mean() * 100),
                    'worstFA%': round(fa.max(), 2),
                    'deaf': int((fa == 0).sum()),
                    'groups': len(fa)}

        rows = [score(lambda g: flat, 'flat'),
                score(lambda g: own.get(g, flat), 'own'),
                score(lambda g: mult * den.get(g, np.nan) if pd.notna(den.get(g, np.nan))
                      else flat, 'normalized')]
        for c in cutoffs:
            rows.append(score(
                lambda g, c=c: (own.get(g, flat) if full_n.get(g, 0) >= c
                                else (mult * den.get(g, np.nan)
                                      if pd.notna(den.get(g, np.nan)) else flat)),
                f'tiered@{c}'))

        t = pd.DataFrame(rows)
        out[system] = t
        if verbose:
            # A group can only produce false-alarm rates of k/n. If no integer k
            # lands in the band, that group is guaranteed to score out-of-band no
            # matter how good its threshold is — a measurement limit, not a
            # detection failure. Surface it so the numbers aren't misread.
            need = int(np.ceil(100 / (target * 2)))
            n_meas = int((ntest.reindex(ev) >= need).sum())
            print(f"\n=== STRATEGY COMPARISON — {system} "
                  f"(dim={dim}, out-of-sample, {len(ev)} {dim} groups) ===")
            print(f"    target false-alarm rate {target:.2f}%, "
                  f"band [{target/2:.2f}%, {target*2:.2f}%]")
            if n_meas < len(ev):
                print(f"    !! only {n_meas} of {len(ev)} groups have the "
                      f">={need} test records needed for the band to be")
                print(f"       reachable at all. The other {len(ev)-n_meas} can "
                      f"only score 0% (counted deaf) or >={100/ (ntest.reindex(ev).min() or 1):.1f}%")
                print(f"       (counted noisy). Their in_band/deaf figures are "
                      f"measurement artifacts — compare")
                print(f"       strategies by their RELATIVE ordering, not absolute values.")
            print(t.to_string(index=False))
            best_fair = t.loc[t['in_band%'].idxmax(), 'strategy']
            best_cov = t.loc[t['deaf'].idxmin(), 'strategy']
            best_tail = t.loc[t['worstFA%'].idxmin(), 'strategy']
            print(f"    best fairness: {best_fair} | fewest deaf: {best_cov} "
                  f"| lowest tail risk: {best_tail}")
            if best_fair == best_cov == best_tail:
                print(f"    --> {best_fair} wins on all three")
            else:
                print("    --> no clean winner; trade-off. Deaf count is the "
                      "detection-relevant metric (a deaf group catches nothing).")
    return out


def cross_evaluate(history_df, measure_at='SVC_CD',
                   line_sources=('flat', 'SVC_CD', 'ACT_CD', 'MOP_CD'),
                   quantile=0.995, min_test=100, verbose=True):
    """
    THE grouping decision, measured honestly.

    Separates two things that are easy to conflate:
      - WHERE THE LINE COMES FROM  (line_sources: which dimension's groups
        supply each permutation's threshold)
      - WHERE CALIBRATION IS JUDGED (measure_at: the level at which false-alarm
        rates are computed and scored)

    Comparing `own@SVC_CD` to `own@ACT_CD` using each one's *own* level is not a
    fair fight — coarser groups aggregate more permutations, so extreme behavior
    averages out and calibration always looks cleaner. Holding `measure_at` fixed
    removes that artifact: every line source is scored on the same yardstick.

    Domain note: SVC_CD is where billing issues actually manifest, so it is the
    natural `measure_at`. A line source only earns its place if it calibrates
    well AT THAT LEVEL, regardless of how good it looks at its own granularity.

    Detection is always per permutation; `measure_at` affects only how the
    resulting false-alarm rates are aggregated for scoring.
    """
    target = (1 - quantile) * 100
    out = {}
    for system, sdf in history_df.groupby('STRATUS_TANDEM'):
        d = _diffs(sdf).sort_values('SUBM_DATE')
        d['_half'] = d.groupby(LABELS).cumcount() % 2
        fit, test = d[d._half == 0].copy(), d[d._half == 1].copy()

        ntest = test.groupby(measure_at).size()
        ev = ntest[ntest >= min_test].index
        if len(ev) < 2:
            continue
        scored = test[test[measure_at].isin(ev)].copy()
        flat = fit['diff'].quantile(quantile)

        def tally(lines, label):
            """lines: a per-row Series of thresholds aligned to `scored`."""
            hit = (scored['diff'] > lines).groupby(scored[measure_at]).mean() * 100
            hit = hit.dropna()
            return {'line_source': label,
                    f'in_band%@{measure_at}': round(
                        ((hit >= target / 2) & (hit <= target * 2)).mean() * 100),
                    'worstFA%': round(hit.max(), 2),
                    'deaf': int((hit == 0).sum()),
                    f'{measure_at}_groups': len(hit)}

        rows = [tally(pd.Series(flat, index=scored.index), 'flat')]
        for src in line_sources:
            if src == 'flat' or src not in scored.columns:
                continue
            own = fit.groupby(src)['diff'].quantile(quantile)
            den = fit.groupby(src)['diff'].apply(_winsor_std).replace(0, np.nan)
            mult = (fit['diff'] / fit[src].map(den)).quantile(quantile)
            own_rows = scored[src].map(own).fillna(flat)
            norm_rows = (scored[src].map(den) * mult).fillna(flat)
            rows.append(tally(own_rows, f'own@{src}'))
            rows.append(tally(norm_rows, f'normalized@{src}'))
            # never looser than the pooled line — see detectability() for why
            rows.append(tally(own_rows.clip(upper=flat), f'own@{src}_capped'))

        t = pd.DataFrame(rows)
        out[system] = t
        if verbose:
            band_col = f'in_band%@{measure_at}'
            print(f"\n=== CROSS-EVALUATION — {system} ===")
            print(f"    every line source scored on the SAME yardstick: "
                  f"calibration measured at {measure_at} "
                  f"({len(ev)} groups, out-of-sample)")
            print(f"    target false-alarm {target:.2f}%, "
                  f"band [{target/2:.2f}%, {target*2:.2f}%]")
            print(t.to_string(index=False))
            best = t.loc[t[band_col].idxmax()]
            fewest = t.loc[t['deaf'].idxmin()]
            print(f"    best fairness at {measure_at}: {best['line_source']} "
                  f"({best[band_col]}%) | fewest deaf: {fewest['line_source']} "
                  f"({fewest['deaf']})")
            base = t[t.line_source == 'flat'].iloc[0]
            print(f"    (flat baseline: {base[band_col]}% in band, "
                  f"{base['deaf']} deaf)")
    return out


def detectability(history_df, dim='SVC_CD', quantile=0.995, min_records=30,
                  fold_targets=(2, 3, 5, 10, 20), verbose=True):
    """
    THE question that actually matters: how big must a spike be before it flags?

    Everything else in this module scores FALSE ALARMS on normal days. That is
    calibration, not detection. This translates each strategy's thresholds into
    business units — "this code needs a 4.2x move to flag" — and reports how many
    groups can catch a spike of a given size.

    Mechanics: a diff of D in z-units corresponds to a volume fold-change of
    exp(D * global_log_std), because the diff lives in standardised log space.
    So a group's threshold maps directly to the smallest move it can detect.

    Why this matters more than in-band%: a group can be "deaf" (never fires on
    normal days) and still catch every spike that matters, if its threshold sits
    at 4x. A group whose threshold sits at 60x is the real blind spot — and the
    two look identical in the false-alarm metrics.

    Lines are fitted on the full history here (not split-half): we are reporting
    the thresholds the production pipeline would actually use, not validating
    calibration.
    """
    out = {}
    for system, sdf in history_df.groupby('STRATUS_TANDEM'):
        d = _diffs(sdf)
        sd = np.log1p(sdf['TRANSACTION_COUNT']).std()
        sizes = d.groupby(dim).size()
        keep = sizes[sizes >= min_records].index
        if not len(keep):
            continue

        flat = d['diff'].quantile(quantile)
        own = d.groupby(dim)['diff'].quantile(quantile)
        den = d.groupby(dim)['diff'].apply(_winsor_std).replace(0, np.nan)
        mult = (d['diff'] / d[dim].map(den)).quantile(quantile)

        own_k = own.reindex(keep).fillna(flat)
        norm_k = (den.reindex(keep) * mult).fillna(flat)
        strategies = {
            'flat': pd.Series(flat, index=keep),
            f'own@{dim}': own_k,
            f'normalized@{dim}': norm_k,
            # Capped variants: a group's own data may justify a TIGHTER threshold
            # than the pooled one, but never a looser one. Removes the blind spots
            # where a small group's history contains one huge day and its private
            # quantile lands at an absurd multiple — while keeping the extra
            # sensitivity wherever the group genuinely runs quieter than the pool.
            f'own@{dim}_capped': own_k.clip(upper=flat),
            f'normalized@{dim}_capped': norm_k.clip(upper=flat),
        }

        rows = []
        for name, lines in strategies.items():
            folds = np.exp(lines * sd)          # threshold -> volume multiple
            row = {'strategy': name,
                   'median_fold': round(float(folds.median()), 1),
                   'p90_fold': round(float(folds.quantile(0.90)), 1),
                   'worst_fold': round(float(folds.max()), 1)}
            for f in fold_targets:
                row[f'catch_{f}x%'] = round(float((folds <= f).mean() * 100))
            rows.append(row)

        t = pd.DataFrame(rows)
        out[system] = t
        if verbose:
            print(f"\n=== DETECTABILITY — {system} "
                  f"({len(keep)} {dim} groups with >={min_records} records) ===")
            print(f"    global log-std = {sd:.3f}; a threshold of D z-units means "
                  f"a spike of exp(D x {sd:.3f})")
            print(f"    'catch_Nx%' = share of groups whose threshold is at or "
                  f"below an N-fold move,")
            print(f"    i.e. groups that WOULD flag a spike of that size.")
            print(t.to_string(index=False))
            best = t.loc[t[f'catch_{fold_targets[-1]}x%'].idxmax()]
            print(f"    most groups covered at {fold_targets[-1]}x: "
                  f"{best['strategy']} ({best[f'catch_{fold_targets[-1]}x%']}%)")
            worst = t.loc[t['worst_fold'].idxmin()]
            print(f"    lowest blind-spot ceiling: {worst['strategy']} "
                  f"(worst group needs {worst['worst_fold']}x)")
    return out


def volume_profile(history_df, quantile=0.995, verbose=True):
    """
    Is the threshold being set by tiny permutations?

    fold-change = exp(diff * global_log_std) = today / that perm's typical day.
    The global std cancels, so a "7x threshold" literally means 99.5% of normal
    permutation-days land within 7x of their own centre. That is a lot of routine
    swing — and low-count permutations are the likely cause: a perm averaging 2
    txns/day going to 14 is a 7x move and pure Poisson noise.

    If so, high-volume permutations (where a 5x jump is a real incident) are being
    judged by a threshold set by permutations doing 2/day. The fix is not a better
    pooling strategy — it is separating permutations by volume, because relative
    variation is inherently volume-dependent.

    This reports, per volume band: how much permutations in that band naturally
    swing, and what threshold they would need on their own.
    """
    out = {}
    bands = [(0, 5), (5, 20), (20, 100), (100, 1000), (1000, np.inf)]
    for system, sdf in history_df.groupby('STRATUS_TANDEM'):
        d = _diffs(sdf)
        sd = np.log1p(sdf['TRANSACTION_COUNT']).std()
        perm_mean = d.groupby(LABELS)['TRANSACTION_COUNT'].transform('mean')
        rows = []
        for lo, hi in bands:
            m = (perm_mean >= lo) & (perm_mean < hi)
            if m.sum() < 30:
                continue
            sub = d.loc[m, 'diff']
            label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+"
            rows.append({
                'avg_txns_per_day': label,
                'perm_days': int(m.sum()),
                'share_of_rows%': round(m.mean() * 100),
                'median_swing': round(float(np.exp(sub.median() * sd)), 2),
                'own_threshold_fold': round(float(np.exp(sub.quantile(quantile) * sd)), 1),
            })
        if not rows:
            continue
        t = pd.DataFrame(rows)
        pooled = float(np.exp(d['diff'].quantile(quantile) * sd))
        out[system] = t
        if verbose:
            print(f"\n=== VOLUME PROFILE — {system} ===")
            print(f"    pooled threshold across ALL permutations: {pooled:.1f}x")
            print(f"    'median_swing'       = a typical day's move vs its own centre")
            print(f"    'own_threshold_fold' = the {quantile*100:.1f}th pct WITHIN that "
                  f"band, i.e. the threshold")
            print(f"                           those permutations would get on their own")
            print(t.to_string(index=False))
            big = t.iloc[-1]
            print(f"    -> highest-volume band would need only "
                  f"{big['own_threshold_fold']}x, but is judged at {pooled:.1f}x")
            print(f"    -> if that gap is large, volume — not fee code — is the "
                  f"dimension that matters")
    return out


def quantile_sweep(history_df, dim='SVC_CD',
                   quantiles=(0.999, 0.995, 0.99, 0.98, 0.95, 0.90),
                   verbose=True):
    """
    The operating point: alert volume vs detection sensitivity.

    Once the line-setting strategy is fixed, the quantile is the only remaining
    dial — and it is a business decision, not a statistical one. A lower quantile
    means a tighter threshold: smaller spikes get caught, more alerts get raised.

    Reported per quantile, using own@dim capped at the pooled line:
      alerts_per_day    what the team would actually absorb
      median_fold       the spike size a typical group needs
      worst_fold        the ceiling — nothing can hide above this
      catch_2x/3x/5x%   share of groups that would catch a spike of that size

    Pick the row where alerts_per_day is tolerable and catch_Nx% covers the
    smallest move the business needs to see. That is the operating point.
    """
    out = {}
    for system, sdf in history_df.groupby('STRATUS_TANDEM'):
        d = _diffs(sdf)
        sd = np.log1p(sdf['TRANSACTION_COUNT']).std()
        n_days = sdf['SUBM_DATE'].nunique()
        sizes = d.groupby(dim).size()
        keep = sizes[sizes >= 30].index

        rows = []
        for q in quantiles:
            flat = d['diff'].quantile(q)
            own = d.groupby(dim)['diff'].quantile(q).reindex(keep).fillna(flat)
            lines = own.clip(upper=flat)
            folds = np.exp(lines * sd)
            per_row_line = d[dim].map(lines).fillna(flat)
            n_flag = int((d['diff'] > per_row_line).sum())
            rows.append({
                'quantile': q,
                'alerts_per_day': round(n_flag / max(n_days, 1), 1),
                'median_fold': round(float(folds.median()), 1),
                'worst_fold': round(float(folds.max()), 1),
                'catch_2x%': round(float((folds <= 2).mean() * 100)),
                'catch_3x%': round(float((folds <= 3).mean() * 100)),
                'catch_5x%': round(float((folds <= 5).mean() * 100)),
            })
        t = pd.DataFrame(rows)
        out[system] = t
        if verbose:
            print(f"\n=== OPERATING POINT — {system} "
                  f"(own@{dim} capped, {len(keep)} groups, {n_days} days) ===")
            print(f"    alerts_per_day is measured on HISTORY, so it counts normal")
            print(f"    days crossing the line — the false-alarm load, not incidents.")
            print(t.to_string(index=False))
            print(f"    -> pick the row where alerts_per_day is absorbable AND")
            print(f"       catch_Nx% covers the smallest move the business must see")
    return out


def recommend_config(history_df, candidate_dims=None, measure_at='SVC_CD',
                     min_catch_fold=5, quantiles=(0.999, 0.995, 0.99, 0.98, 0.95),
                     verbose=True):
    """
    Pick the grouping dimension AND the quantile, with a confidence score.

    Chains the diagnostics into one decision so the config is not chosen by
    eyeballing tables:

      1. gate 1      screen dimensions for real, reproducible heterogeneity
      2. cross-eval  score every survivor on ONE yardstick (measure_at)
      3. detectability  reject anything whose blind spot exceeds min_catch_fold
      4. quantile sweep  pick the loosest quantile that still catches
                         min_catch_fold everywhere

    CONFIDENCE combines five signals, each 0-1, reported individually so a
    reviewer can see WHY it is high or low rather than trusting a single number:

      stability   split-half r of the winning dimension's wobble ranking
      margin      how far the winner beats the runner-up on in-band%
      separation  how far the winner beats flat (the do-nothing baseline)
      coverage    share of groups with enough records to calibrate honestly
      agreement   consistency across partitions (1.0 when only one partition)

    A low score is a signal to review the tables manually, not to abort.
    """
    if candidate_dims is None:
        candidate_dims = [c for c in CANDIDATES if c in history_df.columns]

    gate1 = advise(history_df, verbose=False)
    cross = cross_evaluate(history_df, measure_at=measure_at,
                           line_sources=tuple(['flat'] + list(candidate_dims)),
                           verbose=False)
    det = {sysname: detectability(history_df[history_df.STRATUS_TANDEM == sysname],
                                  dim=measure_at, verbose=False).get(sysname)
           for sysname in cross}

    out = {}
    for system, table in cross.items():
        band_col = f'in_band%@{measure_at}'
        qualified = gate1.get(system, [])

        # candidate rows: capped variants of qualifying dims (capping is the
        # settled policy — it bounds blind spots at the pooled line)
        cand = table[table.line_source.str.endswith('_capped')].copy()
        cand['dim'] = cand.line_source.str.replace('own@', '', regex=False) \
                                      .str.replace('_capped', '', regex=False)
        cand = cand[cand['dim'].isin(qualified)] if qualified else cand.iloc[0:0]
        flat_row = table[table.line_source == 'flat'].iloc[0]

        if cand.empty:
            out[system] = {'group_dim': None, 'confidence': 0.0,
                           'reason': 'no dimension passed gate 1; use flat pooling'}
            if verbose:
                print(f"\n=== CONFIG RECOMMENDATION — {system} ===")
                print("    no dimension shows reproducible heterogeneity -> flat pooling")
            continue

        cand = cand.sort_values(band_col, ascending=False)
        win = cand.iloc[0]
        win_dim = win['dim']
        runner = cand.iloc[1] if len(cand) > 1 else None

        # A dimension must actually BEAT the do-nothing baseline. Passing gate 1
        # only proves the heterogeneity is real, not that grouping by it helps —
        # exactly the trap that produced the retracted median-rescale proposal.
        if win[band_col] <= flat_row[band_col]:
            out[system] = {
                'group_dim': None, 'quantile': None, 'confidence': 0.0,
                'reason': (f"{win_dim} qualified on heterogeneity but scores "
                           f"{win[band_col]:.0f}% at {measure_at} vs flat's "
                           f"{flat_row[band_col]:.0f}% — grouping does not help here"),
            }
            if verbose:
                print(f"\n=== CONFIG RECOMMENDATION — {system} ===")
                print(f"    group_dim = None (flat pooling)")
                print(f"    {out[system]['reason']}")
            continue

        # ---- confidence components -------------------------------------
        d = _diffs(history_df[history_df.STRATUS_TANDEM == system]).sort_values('SUBM_DATE')
        d['_h'] = d.groupby(LABELS).cumcount() % 2
        h0 = _bucket_wobble(_diffs(d[d._h == 0]), win_dim, MIN_N // 2)
        h1 = _bucket_wobble(_diffs(d[d._h == 1]), win_dim, MIN_N // 2)
        common = h0.index.intersection(h1.index)
        stability = float(h0.loc[common, 'wobble'].corr(
            h1.loc[common, 'wobble'], method='spearman')) if len(common) >= 3 else 0.0

        margin = ((win[band_col] - runner[band_col]) / 100
                  if runner is not None else 0.5)
        separation = (win[band_col] - flat_row[band_col]) / 100
        sizes = d.groupby(win_dim).size()
        coverage = float((sizes >= 100).sum() / max(len(sizes), 1))
        agreement = 1.0 if len(cross) == 1 else None   # filled in after the loop

        comps = {'stability': max(0.0, min(1.0, stability)),
                 'margin': max(0.0, min(1.0, margin * 4)),
                 'separation': max(0.0, min(1.0, separation * 2)),
                 'coverage': coverage}
        conf = float(np.mean(list(comps.values())))

        # ---- pick the quantile: loosest that catches min_catch_fold ------
        sweep = quantile_sweep(history_df[history_df.STRATUS_TANDEM == system],
                               dim=win_dim, quantiles=quantiles, verbose=False)[system]
        ok = sweep[sweep['worst_fold'] <= min_catch_fold]
        chosen_q = float(ok.iloc[0]['quantile']) if len(ok) else float(sweep.iloc[-1]['quantile'])
        qrow = sweep[sweep['quantile'] == chosen_q].iloc[0]

        out[system] = {
            'group_dim': win_dim,
            # study-only provenance: the level at which this choice was VALIDATED.
            # Not a pipeline parameter — detection is always per permutation, and
            # group_dim alone decides which line a permutation faces.
            'validated_at': measure_at,
            'quantile': chosen_q,
            'confidence': round(conf, 2),
            'components': {k: round(v, 2) for k, v in comps.items()},
            'alerts_per_day': float(qrow['alerts_per_day']),
            'worst_fold': float(qrow['worst_fold']),
            'median_fold': float(qrow['median_fold']),
            'runner_up': (runner['dim'] if runner is not None else None),
        }

        if verbose:
            print(f"\n=== CONFIG RECOMMENDATION — {system} ===")
            print(f"    group_dim = {win_dim}   quantile = {chosen_q}")
            print(f"    confidence = {conf:.2f}")
            for k, v in comps.items():
                bar = '#' * int(v * 20)
                print(f"       {k:11s} {v:.2f}  {bar}")
            print(f"    why: {win_dim} scores {win[band_col]:.0f}% in band at "
                  f"{measure_at} vs flat's {flat_row[band_col]:.0f}%"
                  + (f", runner-up {runner['dim']} at {runner[band_col]:.0f}%"
                     if runner is not None else ""))
            print(f"    at q={chosen_q}: {qrow['alerts_per_day']} alerts/day, "
                  f"typical group needs {qrow['median_fold']}x, "
                  f"worst needs {qrow['worst_fold']}x "
                  f"(target: catch everything at {min_catch_fold}x)")
            if conf < 0.6:
                print(f"    !! confidence below 0.6 — review the Part 1b/1c tables "
                      f"before shipping this config")
    return out


def emit_calibration(history_df, dim='SVC_CD', quantile=0.99, cap_multiplier=1.0,
                     min_records=100, window=None, verbose=True):
    """
    Produce the calibration artifact the scoring pipeline consumes at run time.

    Separates CALIBRATION (which dimension, which quantile, what the lines are —
    decided by measurement, re-run when the data or the grouping changes) from
    SCORING (applied daily). Changing the grouping to INTERCHANGE_LVL_CODE or
    anything else becomes: re-run the study with --dim, review the diagnostics,
    ship the new config. No pipeline code changes.

    The config carries two things:

      RECIPE (authoritative)  - group_dim, quantile, cap policy. The pipeline
                                recomputes lines from its OWN history using these
                                rules, so it stays self-consistent even if its
                                window differs from the study's.
      REFERENCE (advisory)    - the lines this study computed, plus the record
                                counts behind each. Use to validate that the
                                pipeline's own numbers land in the same place;
                                a large divergence means the windows disagree.

    `dim` must exist as a column. Any categorical column works — the analysis
    never assumes SVC_CD semantics.
    """
    # dim=None is a valid config: flat pooling, one line per partition. It is what
    # the recommender returns when no grouping beats the pooled baseline.
    flat_only = dim is None
    if not flat_only and dim not in history_df.columns:
        raise ValueError(
            f"grouping column {dim!r} not present. Available: "
            f"{[c for c in history_df.columns if c not in ('SUBM_DATE','TRANSACTION_COUNT')]}"
        )

    cfg = {
        'schema_version': 1,
        'recipe': {
            'group_dim': dim,   # None = flat pooling, one line per partition
            'quantile': quantile,
            'cap_policy': 'min(group_own_quantile, cap_multiplier * pooled_quantile)',
            'cap_multiplier': cap_multiplier,
            'min_records': min_records,
            'thin_group_policy': (f'groups with < {min_records} records use the '
                                  f'pooled line, not their own quantile'),
            'center_level': 'permutation',
            'permutation_labels': LABELS,
            'transform': 'log1p',
            'standardize': 'one global mean/std per system, frozen from history',
        },
        'provenance': {
            'history_window': window,
            'source_rows': int(len(history_df)),
            'distinct_days': int(history_df['SUBM_DATE'].nunique()),
        },
        'systems': {},
    }

    for system, sdf in history_df.groupby('STRATUS_TANDEM'):
        d = _diffs(sdf)
        lg = np.log1p(sdf['TRANSACTION_COUNT'])
        sd = float(lg.std())
        pooled = float(d['diff'].quantile(quantile))
        cap = pooled * cap_multiplier
        if flat_only:
            sizes, lines = pd.Series(dtype=int), pd.Series(dtype=float)
        else:
            sizes = d.groupby(dim).size()
            own = d.groupby(dim)['diff'].quantile(quantile)
            # TWO guards, protecting opposite failures:
            #   cap         - a group's own quantile can be absurdly LOOSE when its
            #                 history happens to hold one huge day (seen: 266x).
            #   min_records - and absurdly TIGHT when it has almost no history at
            #                 all (seen: a 3-record group getting a 1.04x line,
            #                 which would fire on a 4% move). Below the threshold
            #                 we do not trust the group's own tail at all and fall
            #                 back to the pooled line.
            lines = own.clip(upper=cap)
            thin = sizes.index[sizes < min_records]
            lines.loc[lines.index.intersection(thin)] = cap

        cfg['systems'][str(system)] = {
            'log_mean': float(lg.mean()),
            'log_std': sd,
            'pooled_line_z': pooled,
            'pooled_line_fold': round(float(np.exp(pooled * sd)), 2),
            'cap_z': cap,
            'cap_fold': round(float(np.exp(cap * sd)), 2),
            'default_line_z': cap,      # groups absent from history fall back here
            'n_permutations': int(d.groupby(LABELS).ngroups),
            'lines': {} if flat_only else {
                str(g): {
                    'line_z': round(float(lines[g]), 6),
                    'line_fold': round(float(np.exp(lines[g] * sd)), 2),
                    'n_records': int(sizes[g]),
                    'source': ('pooled_thin' if sizes[g] < min_records
                               else ('own' if own[g] <= cap else 'capped')),
                }
                for g in lines.index
            },
        }

    if verbose:
        print(f"\n=== CALIBRATION CONFIG ===")
        print(f"    recipe: group by {dim}, quantile {quantile}, "
              f"cap at {cap_multiplier}x the pooled line")
        for system, s in cfg['systems'].items():
            if flat_only:
                print(f"    {system}: flat pooling — one line at "
                      f"{s['pooled_line_fold']}x for every permutation")
                continue
            src = pd.Series([v['source'] for v in s['lines'].values()]).value_counts()
            folds = [v['line_fold'] for v in s['lines'].values()]
            print(f"    {system}: {len(s['lines'])} {dim} lines "
                  f"({int(src.get('own',0))} own, {int(src.get('capped',0))} capped, "
                  f"{int(src.get('pooled_thin',0))} pooled-thin)")
            print(f"        cap = {s['cap_fold']}x | line folds: "
                  f"min {min(folds)}x, median {float(np.median(folds))}x, "
                  f"max {max(folds)}x")
            print(f"        groups not in history fall back to the cap")
    return cfg


def recommend(history_df, test_all=False, verbose=True):
    """
    Full two-gate recommendation.

    Gate 1 screens dimensions for real, reproducible wobble heterogeneity.
    Gate 2 then runs on EVERY qualifier and picks the winner by MEASURED
    effectiveness (in-band gain over flat, subject to contamination limits) —
    not by which had the biggest gate-1 ratio.

    test_all=True runs gate 2 on every candidate dimension regardless of gate 1,
    which validates that the screening didn't discard something useful.

    Returns {system: {'WOBBLE_DIM': dim_or_None, 'NORMALIZER': name_or_none}}.
    """
    qualifiers = advise(history_df, verbose=verbose)
    final = {}
    for system, dims in qualifiers.items():
        sdf = history_df[history_df.STRATUS_TANDEM == system]
        to_test = CANDIDATES if test_all else dims
        results = {}
        for dim in to_test:
            r = compare_normalizers(sdf, dim=dim, verbose=verbose)[system]
            results[dim] = r

        winners = {d: r for d, r in results.items() if r['normalizer'] != 'none'}
        if winners:
            best_dim = max(winners, key=lambda d: winners[d]['gain'])
            final[system] = {'WOBBLE_DIM': best_dim,
                             'NORMALIZER': winners[best_dim]['normalizer']}
        else:
            final[system] = {'WOBBLE_DIM': None, 'NORMALIZER': None}

        if verbose:
            print(f"\n{'='*70}\nFINAL RECOMMENDATION — {system}")
            if results:
                print("  gate-2 outcome per dimension tested:")
                for d, r in results.items():
                    tag = (f"{r['normalizer']} (+{r['gain']:.0f} pts)"
                           if r['normalizer'] != 'none' else 'nothing clears the bar')
                    print(f"     {d:12s} flat={r['flat_in_band']:.0f}% -> {tag}")
            else:
                print("  no dimension passed gate 1")
            f = final[system]
            print(f"  WOBBLE_DIM = {f['WOBBLE_DIM']}   NORMALIZER = {f['NORMALIZER']}")
            if f['WOBBLE_DIM'] is None:
                print("  -> flat system pool: no per-group denominator, no contamination surface")
            print('='*70)
    return final


def advise(history_df, verbose=True):
    """Return {system: recommended_dim_or_None}; print the evidence table."""
    out = {}
    for system, sdf in history_df.groupby('STRATUS_TANDEM'):
        d = _diffs(sdf)

        # chronological split-half per perm for the stability check
        d = d.sort_values('SUBM_DATE')
        d['half'] = d.groupby(LABELS).cumcount() % 2  # alternate days per perm

        rows, qualifying = [], []
        for dim in CANDIDATES:
            g = _bucket_wobble(d, dim)
            if len(g) < 2:
                continue
            ratio = g['wobble'].max() / max(g['wobble'].min(), 1e-9)

            h0 = _bucket_wobble(_diffs(d[d.half == 0]), dim, MIN_N // 2)
            h1 = _bucket_wobble(_diffs(d[d.half == 1]), dim, MIN_N // 2)
            common = h0.index.intersection(h1.index)
            stability = (h0.loc[common, 'wobble']
                         .corr(h1.loc[common, 'wobble'], method='spearman')
                         if len(common) >= 3 else np.nan)

            coverage = g['n'].sum() / len(d)
            ok = (ratio >= MIN_RATIO) and (stability >= MIN_STABILITY)
            rows.append((dim, len(g), int(g['n'].min()), round(ratio, 2),
                         round(float(stability), 2) if pd.notna(stability) else None,
                         f"{coverage:.0%}", 'QUALIFIES' if ok else 'no'))
            if ok:
                qualifying.append((ratio, dim))

        # ALL qualifiers, ranked by ratio. Gate 1 SCREENS; gate 2 DECIDES —
        # a big ratio only means "worth testing", never "will work" (the
        # median-rescale had a 5.6x ratio and still made fairness worse).
        ranked = [dim for _, dim in sorted(qualifying, reverse=True)]
        out[system] = ranked
        if verbose:
            print(f"\n=== GATE 1 — {system}: {len(d):,} history rows, "
                  f"{d.groupby(LABELS).ngroups:,} perms ===")
            print(pd.DataFrame(rows, columns=[
                'dimension', 'buckets>=min_n', 'smallest_bucket', 'wobble_ratio',
                'split_half_r', 'row_coverage', 'verdict']).to_string(index=False))
            print(f"--> qualifying dimensions for {system}: "
                  f"{ranked or 'none -> WOBBLE_DIM = None (flat system pool)'}")
    return out


if __name__ == '__main__':
    # Dev mode: pull the mock generator out of anomaly_dev.py without running
    # the training pipeline below it.
    src = open('anomaly_dev.py').read()
    ns = {'__file__': 'anomaly_dev.py'}
    exec(src[:src.index('merchant_id_column_name =')], ns)
    history_df, _ = ns['generate_mock_data']()
    # test_all=True: run gate 2 on every candidate dimension, not just gate-1
    # qualifiers, so we can verify the screening didn't discard something useful.
    recommend(history_df, test_all=True)
