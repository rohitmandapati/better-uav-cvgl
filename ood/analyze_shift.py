"""Test 1 analysis: errors by severity/case, miss-detection scores (H1), and milestone figures.

Run from better-uav-cvgl/ (CPU is fine, takes seconds):
  python ood/analyze_shift.py                         # newest outputs/test1/shift_*.csv
  python ood/analyze_shift.py --csv outputs/test1/shift_XXXX.csv
Outputs: <out>/summary.txt and fig_*.png / fig_*.pdf
"""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

M_PER_UNIT = 32.0
CASE_NAME = {1: 'Case 1: in training region', 2: 'Case 2: in block, outside training region',
             3: 'Case 3: outside block, overlapping', 4: 'Case 4: no overlap'}
CASE_SHORT = {1: 'Case 1\nin training\nregion', 2: 'Case 2\nin block', 3: 'Case 3\noutside,\noverlap',
              4: 'Case 4\nno overlap'}
# validated categorical palette (dataviz reference, light mode), fixed order
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2, GRID = '#1f1f1e', '#5f5e5a', '#e6e5e0'
TEMPS = [0.1, 0.03, 0.01]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='')
    ap.add_argument('--out', default='outputs/test1/analysis')
    return ap.parse_args()


def newest_csv():
    files = [f for f in glob.glob('outputs/test1/shift_*.csv') if not f.endswith('_dry.csv')]
    if not files:
        raise SystemExit('no outputs/test1/shift_*.csv found; pass --csv')
    return max(files, key=os.path.getmtime)


def style(ax):
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)


def save(fig, out, name):
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}/{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def add_scores(P):
    S = P[['sim1', 'sim2', 'sim3', 'sim4']].to_numpy()
    srt = np.sort(S, axis=1)
    sc = {'max softmax prob (model, T=1)': P.max_prob.to_numpy(),
          'max cosine similarity': srt[:, -1],
          'similarity spread (max-min)': srt[:, -1] - srt[:, 0],
          'top-1 minus top-2 similarity': srt[:, -1] - srt[:, -2]}
    for T in TEMPS:
        e = np.exp((S - S.max(1, keepdims=True)) / T)
        sc[f'max softmax prob, T={T}'] = (e / e.sum(1, keepdims=True)).max(1)
    return sc


def detection(scores, mask_pos, mask_neg):
    """Score = confidence that the block is valid. Positive class for AUROC = valid (mask_pos).
    AUPR is reported with the *invalid* block as the positive class (what we want to catch)."""
    rows = []
    for name, s in scores.items():
        y = np.r_[np.ones(mask_pos.sum()), np.zeros(mask_neg.sum())]
        v = np.r_[s[mask_pos], s[mask_neg]]
        rows.append(dict(score=name, AUROC=roc_auc_score(y, v),
                         AUPR_invalid=average_precision_score(1 - y, -v),
                         base_rate_invalid=1 - y.mean()))
    return pd.DataFrame(rows).set_index('score')


def main():
    args = parse_args()
    csv = args.csv or newest_csv()
    P = pd.read_csv(csv)
    os.makedirs(args.out, exist_ok=True)
    lines = [f'source: {csv}   rows: {len(P)}   test samples: {P["sample"].nunique()}', '']
    pd.set_option('display.width', 200)

    def section(title, obj):
        lines.extend([f'== {title} ==', obj if isinstance(obj, str) else obj.round(3).to_string(), ''])

    # ---------------- errors
    section('Errors by shift distance d (m)', P.groupby('d_m').agg(
        n=('case', 'size'), pos_err_mean=('pos_err_m', 'mean'), pos_err_median=('pos_err_m', 'median'),
        head_err_mean=('head_err_deg', 'mean'), head_err_median=('head_err_deg', 'median')))
    section('Errors by case', P.groupby('case').agg(
        n=('case', 'size'), pos_err_mean=('pos_err_m', 'mean'), pos_err_median=('pos_err_m', 'median'),
        head_err_mean=('head_err_deg', 'mean'), head_err_median=('head_err_deg', 'median')))

    # does the prediction follow the UAV, or stay near the block center?
    P['true_from_center_m'] = np.hypot(P.x_new, P.y_new) * M_PER_UNIT
    P['pred_from_center_m'] = np.hypot(P.pred_x, P.pred_y) * M_PER_UNIT
    P['pred_in_train_region'] = (P.pred_x.abs() <= 1) & (P.pred_y.abs() <= 1)
    section('Where the predictions land, by case', P.groupby('case').agg(
        true_from_center_m=('true_from_center_m', 'mean'), pred_from_center_m=('pred_from_center_m', 'mean'),
        frac_pred_in_training_region=('pred_in_train_region', 'mean'), pos_err_m=('pos_err_m', 'mean')))

    # Case 1 check: is the higher error for shifted blocks due to off-grid crops or to edge positions?
    c1 = P[P.case == 1].copy()
    c1['r'] = np.maximum(c1.x_new.abs(), c1.y_new.abs())
    c1['r_bin'] = pd.cut(c1.r, [0, 0.5, 0.75, 1.0], include_lowest=True)
    c1['block'] = np.where(c1.d_m == 0, 'unshifted (d=0)', 'shifted (d>0)')
    section('Case 1 position error (m) by distance from block center (max(|x|,|y|), units) and shift',
            c1.pivot_table(index='r_bin', columns='block', values='pos_err_m', aggfunc=['mean', 'size'],
                           observed=False))

    # ---------------- miss detection (H1)
    sc = add_scores(P)
    tasks = {
        'in block (Cases 1-2) vs outside (Cases 3-4)': (P.case.isin([1, 2]), P.case.isin([3, 4])),
        'Case 2 vs Case 3 (both partially overlap)': (P.case.eq(2), P.case.eq(3)),
        'Case 1 vs Case 4 (easiest)': (P.case.eq(1), P.case.eq(4)),
    }
    det = {}
    for t, (mp, mn) in tasks.items():
        det[t] = detection(sc, mp.to_numpy(), mn.to_numpy())
        section(f'Miss detection: {t}   (AUROC 0.5 = chance; < 0.5 = score points the wrong way)', det[t])
    lines.append('note: temperatures here are swept on the test set, so the best-T row is optimistic; '
                 'a real baseline must pick T on validation data.')

    with open(f'{args.out}/summary.txt', 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))

    # ---------------- figures
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})

    # 1) error vs shift (two panels, no dual axis)
    g = P.groupby('d_m')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, col, ylab in [(axes[0], 'pos_err_m', 'Position error (m)'),
                          (axes[1], 'head_err_deg', 'Heading error (deg)')]:
        q = g[col].quantile([0.25, 0.75]).unstack()
        ax.fill_between(q.index, q[0.25], q[0.75], color=SERIES[0], alpha=0.18, lw=0)
        ax.plot(g[col].mean().index, g[col].mean(), color=SERIES[0], lw=2, marker='o', ms=4)
        ax.set_xlabel('Reference-block shift d (m)'); ax.set_ylabel(ylab); style(ax)
    d = np.array(sorted(P.d_m.unique()))
    axes[0].plot(d, d, ls='--', color=INK2, lw=1)
    axes[0].text(d[-1] * 0.62, d[-1] * 0.62 - 0.12 * d[-1], 'dashed: error = shift', color=INK2, fontsize=8, ha='left')
    axes[0].text(0.02, 0.97, 'mean, band = IQR', transform=axes[0].transAxes, fontsize=8, color=INK2, va='top')
    fig.tight_layout(); save(fig, args.out, 'fig_error_vs_shift')

    # 2) position error by case
    fig, ax = plt.subplots(figsize=(6, 3.6))
    data = [P.loc[P.case == c, 'pos_err_m'].clip(lower=0.1) for c in range(1, 5)]
    bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.5))
    for patch, col in zip(bp['boxes'], SERIES):
        patch.set_facecolor(col); patch.set_alpha(0.75); patch.set_edgecolor(col)
    ax.set_xticks(range(1, 5), [CASE_SHORT[c] for c in range(1, 5)], fontsize=8)
    ax.set_yscale('log'); ax.set_ylabel('Position error (m, log scale)'); style(ax)
    fig.tight_layout(); save(fig, args.out, 'fig_error_by_case')

    # 3) confidence distributions by case
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (key, xlab) in zip(axes, [('max softmax prob (model, T=1)', 'Max softmax probability (model)'),
                                      ('max cosine similarity', 'Max cosine similarity')]):
        s = sc[key]
        bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 50)
        for c in range(1, 5):
            ax.hist(s[P.case.to_numpy() == c], bins=bins, histtype='step', lw=1.8, density=True,
                    color=SERIES[c - 1], label=CASE_NAME[c])
        ax.set_xlabel(xlab); ax.set_ylabel('Density'); style(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); save(fig, args.out, 'fig_confidence_by_case')

    # 4) ROC, in-block vs outside
    mp, mn = tasks['in block (Cases 1-2) vs outside (Cases 3-4)']
    y = np.r_[np.ones(mp.sum()), np.zeros(mn.sum())]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    keys = ['max softmax prob (model, T=1)', 'max cosine similarity', 'similarity spread (max-min)',
            'top-1 minus top-2 similarity', f'max softmax prob, T={TEMPS[1]}']
    for k, col in zip(keys, SERIES):
        v = np.r_[sc[k][mp.to_numpy()], sc[k][mn.to_numpy()]]
        fpr, tpr, _ = roc_curve(y, v)
        ax.plot(fpr, tpr, color=col, lw=2, label=f'{k} (AUROC {roc_auc_score(y, v):.2f})')
    ax.plot([0, 1], [0, 1], ls='--', color=INK2, lw=1)
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); style(ax)
    ax.set_title('Valid (Cases 1-2) vs invalid (Cases 3-4) block', fontsize=10, color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc='upper left', bbox_to_anchor=(0, -0.16))
    fig.tight_layout(); save(fig, args.out, 'fig_roc_in_vs_out')

    print(f'\nsaved summary and figures to {args.out}/')


if __name__ == '__main__':
    main()
