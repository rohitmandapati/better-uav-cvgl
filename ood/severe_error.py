"""Test 1 severe-error analysis: if we only accept predictions whose confidence score is above a
threshold, how many accepted predictions are still badly wrong? (risk-coverage curve,
Geifman & El-Yaniv, NeurIPS 2017, https://arxiv.org/abs/1705.08500)

  coverage = fraction of predictions accepted
  risk     = fraction of accepted predictions with pos_err_m > severe threshold

Run from better-uav-cvgl/ (CPU is fine, takes seconds):
  python ood/severe_error.py --csv outputs/test1/shift_XXXX_grid.csv --out outputs/test1/severe_grid
Outputs: <out>/severe_summary.txt and fig_risk_coverage_<m>m.png / .pdf

Caveat: the overall severe rate is set by the shift sizes we chose in test 1, so absolute risk
values are only for comparing scores against each other, not real-flight failure rates.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_shift import add_scores, newest_csv, style, save, SERIES, INK, INK2

PLOT_SCORES = ['max softmax prob (model, T=1)', 'max cosine similarity',
               'similarity spread (max-min)', 'top-1 minus top-2 similarity']


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='')
    ap.add_argument('--out', default='outputs/test1/severe')
    ap.add_argument('--severe', type=float, nargs='+', default=[64.0, 32.0],
                    help='error thresholds in meters (64 m = one tile, 32 m = one unit)')
    ap.add_argument('--target-risk', type=float, nargs='+', default=[0.05, 0.10, 0.20],
                    help='report the highest coverage whose risk stays at or below each of these')
    ap.add_argument('--gate-risk', type=float, default=0.10,
                    help='risk target used for the "what passes the gate, by case" table')
    return ap.parse_args()


def risk_coverage(order, bad):
    """order: sample indices, most trusted first. Returns coverage and risk for accepting the top k."""
    b = bad[order].astype(float)
    k = np.arange(1, len(b) + 1)
    return k / len(b), np.cumsum(b) / k


def n_at_risk(risk, r):
    """Largest number of accepted samples k whose risk is still <= r (0 if never).
    Uses the largest such k, so a few unlucky samples at the very top do not end the curve early."""
    ok = np.nonzero(risk <= r + 1e-12)[0]
    return 0 if len(ok) == 0 else ok[-1] + 1


def main():
    args = parse_args()
    csv = args.csv or newest_csv()
    P = pd.read_csv(csv)
    os.makedirs(args.out, exist_ok=True)
    err = P.pos_err_m.to_numpy()
    case = P.case.to_numpy()
    sc = add_scores(P)
    pd.set_option('display.width', 200)
    lines = [f'source: {csv}   rows: {len(P)}', '',
             'risk = fraction of ACCEPTED predictions with position error > threshold',
             'coverage = fraction of predictions accepted (most confident first)',
             'cov@risk<=r = highest coverage whose risk stays at or below r (higher is better)',
             'AURC = mean risk over all coverages (lower is better)',
             'max useful coverage = fraction of predictions that are NOT severe; even a perfect gate',
             '  must accept severe errors beyond it, so cov@risk<=r can never exceed good/(1-r)', '']

    for thr in args.severe:
        bad = err > thr
        good = 1 - bad.mean()
        orders = {name: np.argsort(-s, kind='stable') for name, s in sc.items()}
        orders['oracle (sort by true error)'] = np.argsort(err, kind='stable')
        curves = {name: risk_coverage(o, bad) for name, o in orders.items()}
        rows = []
        for name, (cov, risk) in curves.items():
            r = {f'cov@risk<={t:g}': n_at_risk(risk, t) / len(risk) for t in args.target_risk}
            r['AURC'] = risk.mean()
            rows.append(dict(score=name, **r))
        tab = pd.DataFrame(rows).set_index('score')
        lines += [f'== Severe error > {thr:g} m   (severe rate {bad.mean():.3f} = random-gate risk; '
                  f'max useful coverage {good:.3f}) ==', tab.round(3).to_string(), '']

        # what passes the gate when each score is tuned to the same risk target?
        g = args.gate_risk
        comp = {'total': pd.Series(case).value_counts().reindex([1, 2, 3, 4], fill_value=0)}
        for name in PLOT_SCORES[:2]:
            short = 'softmax' if name.startswith('max softmax') else 'max sim'
            k = n_at_risk(curves[name][1], g)
            acc = orders[name][:k]
            comp[f'{short}: accepted'] = pd.Series(case[acc]).value_counts().reindex([1, 2, 3, 4], fill_value=0)
            comp[f'{short}: accepted severe'] = (pd.Series(case[acc[bad[acc]]]).value_counts()
                                                 .reindex([1, 2, 3, 4], fill_value=0))
        comp = pd.DataFrame(comp)
        comp.index = [f'Case {c}' for c in comp.index]
        comp.loc['all'] = comp.sum()
        lines += [f'-- What passes the gate at risk <= {g:g} (each score at its own threshold), '
                  f'count by case --', comp.to_string(), '']

        # figure: one axis, risk vs coverage
        plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        for name, col in zip(PLOT_SCORES, SERIES):
            cov, risk = curves[name]
            ax.plot(cov, risk, color=col, lw=2, label=f'{name} (AURC {risk.mean():.3f})')
        cov, risk = curves['oracle (sort by true error)']
        ax.plot(cov, risk, color=INK, lw=1.2, ls='--', label=f'oracle (AURC {risk.mean():.3f})')
        ax.axhline(bad.mean(), color=INK2, lw=1.2, ls=':', label=f'random gate ({bad.mean():.3f})')
        ax.set_xlim(0, 1); ax.set_ylim(0, None)
        ax.set_xlabel('Coverage (fraction of predictions accepted)')
        ax.set_ylabel(f'Risk (fraction accepted with error > {thr:g} m)')
        style(ax)
        ax.legend(fontsize=7.5, frameon=False, loc='upper left', bbox_to_anchor=(0, -0.16))
        fig.tight_layout(); save(fig, args.out, f'fig_risk_coverage_{thr:g}m')

    lines.append('note: temperature rows are tuned on the test set (optimistic); '
                 'absolute risk depends on the shift mix chosen in test 1.')
    with open(f'{args.out}/severe_summary.txt', 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'\nsaved summary and figures to {args.out}/')


if __name__ == '__main__':
    main()
