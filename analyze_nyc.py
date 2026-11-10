# Summarise the NYC runs written by run_nyc.py: learning curves, coverage, summary table, paired tests.
#
# python analyze_nyc.py                       -> analysis/nyc/{returns.png, coverage.png, summary.md}
import os
import glob
import pickle
import argparse
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# fixed method order = fixed colour slot (categorical palette slots 1-4)
METHODS = [('neo', 'NEO', '#2a78d6'), ('neo1', 'NEO (first ev.)', '#eb6834'),
           ('sp', 'SP-novelty', '#1baf7a'), ('qlnov', 'Q-learning-novelty', '#eda100')]
INK, MUTED, GRID = '#1a1a19', '#6b6a63', '#e4e3dc'
DELTA = 0.5  # NEO scaling delta; Algorithm text: eigenvectors with lambda < delta give gradients
RNG = np.random.default_rng(0)


def load(results_dir, graph):
    runs = defaultdict(dict)
    for f in glob.glob(os.path.join(results_dir, f'{graph}_*_seed*.pkl')):
        with open(f, 'rb') as fh:
            r = pickle.load(fh)
        runs[r['method']][r['seed']] = r
    return runs


def boot_ci(x, n=5000, axis=0):
    """95% bootstrap CI of the mean over seeds (axis 0)."""
    x = np.asarray(x, float)
    idx = RNG.integers(0, x.shape[0], size=(n, x.shape[0]))
    means = x[idx].mean(axis=1)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def style(ax, ylabel):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel('environment steps (thousands)', color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)


def curve_plot(runs, key, ylabel, title, out, scale=1.0, ylim=None):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=160)
    ends = []
    for m, label, color in METHODS:
        if m not in runs:
            continue
        seeds = sorted(runs[m])
        y = np.array([runs[m][s][key] for s in seeds], float) * scale
        x = np.arange(y.shape[1]) * 2  # one evaluation every 2000 steps
        lo, hi = boot_ci(y)
        mean = y.mean(axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=2, label=f'{label} (n={len(seeds)})')
        ends.append([mean[-1], label, color])
    # direct labels at the line ends, nudged apart so they do not collide
    ends.sort()
    span = (ylim[1] - ylim[0]) if ylim else max(e[0] for e in ends) or 1
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.06 * span)
    for yv, label, color in ends:
        ax.text(x[-1] + 4, yv, label, color=INK, fontsize=8.5, va='center')
        ax.plot([x[-1] + 1, x[-1] + 3.5], [yv, yv], color=color, linewidth=2)
    style(ax, ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(0, x[-1] + 55)
    ax.set_title(title, color=INK, fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', labelcolor=INK)
    fig.text(0.01, 0.01, 'Line: mean over seeds. Band: 95% bootstrap CI of the mean.', color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def summarise(runs):
    rows, per_seed = [], {}
    for m, label, _ in METHODS:
        if m not in runs:
            continue
        seeds = sorted(runs[m])
        ev = np.array([runs[m][s]['eval_returns'] for s in seeds], float)
        nodes = np.array([runs[m][s]['node_counts'] for s in seeds], float)
        auc = ev.mean(axis=1)                 # mean eval return over the whole run
        final = ev[:, -10:].mean(axis=1)      # last 10 evaluations = last 20k steps
        solved = (ev > 0).any(axis=1)
        first = [2 * int(np.argmax(r > 0)) if (r > 0).any() else np.nan for r in ev]
        per_seed[m] = dict(zip(seeds, auc))
        lo, hi = boot_ci(auc[:, None])
        rows.append({
            'method': label, 'n': len(seeds),
            'AUC (mean eval return)': f'{auc.mean():.1f} [{lo[0]:.1f}, {hi[0]:.1f}]',
            'sigma of AUC': f'{auc.std():.1f}',
            'final return (last 20k steps)': f'{final.mean():.1f}',
            'seeds ever solving eval': f'{solved.sum()}/{len(seeds)}',
            'first solved eval (k steps, median)': f'{np.nanmedian(first):.0f}' if solved.any() else '-',
            'final coverage (nodes)': f'{nodes[:, -1].mean():.0f} ± {nodes[:, -1].std():.0f}',
            'wall clock (min, mean)': f'{np.mean([runs[m][s]["wall_clock_s"] for s in seeds]) / 60:.1f}',
        })
    tests = []
    for a, b in [('neo', 'sp'), ('neo', 'qlnov'), ('neo', 'neo1')]:
        if a in per_seed and b in per_seed:
            common = sorted(set(per_seed[a]) & set(per_seed[b]))
            d = np.array([per_seed[a][s] - per_seed[b][s] for s in common])
            p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            tests.append((a, b, len(common), d.mean(), p))
    return rows, tests


def eigen_report(runs):
    """Theory link (WP1/WP2): are the NEO eigenvalues real, and below delta?"""
    out = []
    for m in ('neo', 'neo1'):
        if m not in runs:
            continue
        lam = np.concatenate([np.asarray(e[0]) for s in runs[m].values() for e in s['info'] if e[0] is not None])
        out.append((m, lam.size, int((np.abs(lam.imag) > 1e-8).sum()), float(np.abs(lam.imag).max()),
                    int((lam.real < DELTA).sum()), float(lam.real.min()), float(lam.real.max())))
    return out


def by_goal(runs):
    """Seeds share goals (the start is random, the goal is the farthest reachable node), so break AUC down by goal."""
    from nyc_graph import load_graph
    from Many_Streets_Eval import get_start_and_goal_state
    import contextlib, io
    G = load_graph('nyc')
    goal = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for s in sorted(set().union(*[set(r) for r in runs.values()])):
            goal[s] = get_start_and_goal_state(s, None, G)[1]
    table = defaultdict(dict)
    for m, label, _ in METHODS:
        for s, r in runs.get(m, {}).items():
            table[goal[s]].setdefault(m, []).append(np.mean(r['eval_returns']))
    return goal, table


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', default=os.path.join('results', 'nyc'))
    p.add_argument('--graph', default='nyc')
    p.add_argument('--out', default=os.path.join('analysis', 'nyc'))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load(a.results, a.graph)

    curve_plot(runs, 'eval_returns', 'evaluation return (goal = 100)',
               'NYC (directed): greedy evaluation return', os.path.join(a.out, 'returns.png'), ylim=(-2, 102))
    curve_plot(runs, 'node_counts', 'unique nodes visited (of 10,000)',
               'NYC (directed): exploration coverage', os.path.join(a.out, 'coverage.png'), ylim=(0, 10000))

    rows, tests = summarise(runs)
    lines = ['# NYC reproduction: summary', '',
             'Evaluation every 2,000 steps, greedy (epsilon = 0), 250k steps, frozen graph `data/nyc`. '
             'AUC = mean evaluation return over the run, with a 95% bootstrap CI over seeds.', '']
    cols = list(rows[0])
    lines += ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(str(r[c]) for c in cols) + ' |' for r in rows]
    lines += ['', '## Paired comparisons (same seed = same start/goal), Wilcoxon signed-rank on AUC', '',
              '| comparison | n | mean AUC difference | p |', '|---|---|---|---|']
    lines += [f'| {a_} vs {b_} | {n} | {d:+.1f} | {pv:.2g} |' for a_, b_, n, d, pv in tests]
    lines += ['', '## NEO eigenvalues at option updates (all seeds)', '',
              '| variant | eigenvalues | complex (abs(Im) > 1e-8) | max abs(Im) | Re < delta = 0.5 | min Re | max Re |',
              '|---|---|---|---|---|---|---|']
    lines += [f'| {m} | {n} | {c} | {im:.1e} | {b} | {lo:.4f} | {hi:.4f} |' for m, n, c, im, b, lo, hi in eigen_report(runs)]
    goal, table = by_goal(runs)
    lines += ['', '## AUC by goal node (seeds sharing a goal are not independent instances)', '',
              '| goal | seeds | ' + ' | '.join(l for m, l, _ in METHODS) + ' |', '|---|---|' + '---|' * len(METHODS)]
    for g in sorted(table, key=lambda g: -len(next(iter(table[g].values())))):
        seeds = sorted(s for s, gg in goal.items() if gg == g)
        lines.append(f'| {g} | {", ".join(map(str, seeds))} | ' +
                     ' | '.join(f'{np.mean(table[g][m]):.1f}' if m in table[g] else '-' for m, _, _ in METHODS) + ' |')
    with open(os.path.join(a.out, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
