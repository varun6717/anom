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
            print(f"\n=== STRATEGY COMPARISON — {system} "
                  f"(dim={dim}, out-of-sample, {len(ev)} {dim} groups) ===")
            print(f"    target false-alarm rate {target:.2f}%, "
                  f"band [{target/2:.2f}%, {target*2:.2f}%]")
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
            rows.append(tally(scored[src].map(own).fillna(flat), f'own@{src}'))
            rows.append(tally((scored[src].map(den) * mult).fillna(flat),
                              f'normalized@{src}'))

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
