# %%
# Extra figures for the Many_Streets_Eval results.
# Usage: python viz_many_streets.py                 (all cities that have results)
#        python viz_many_streets.py Paris London
#        python viz_many_streets.py --instances 5   (which multi-seed results to use for seeds/, default 5)
# Writes into plots_many_streets/:
#   maps/city_maps.pdf                          street graph of every city, with start, goal and requested city center
#   discovery/neo_discovery_all_cities.pdf      NEO: % of states discovered vs training steps, one line per city
#   discovery/discovery_per_city.pdf            all methods, one panel per city
#   coverage/final_coverage.pdf                 % of states discovered at the end of training
#   visitation/visitation_<city>.pdf            where each method spent its steps, drawn on the map
#   options/neo_eigenvalues.pdf                 eigenvalues of NEO's modified Laplacian at every option update
#   seeds/discovery_per_city_seeds<N>.pdf       mean and 95% CI over N seeds (needs the multi-seed runs)
#   seeds/final_coverage_seeds<N>.pdf           final coverage over N seeds, one dot per seed
import os
import sys
import io
import math
import pickle
import argparse
import contextlib
import matplotlib
INTERACTIVE = 'ipykernel' in sys.modules
if not INTERACTIVE:
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats
from Many_Streets_Eval import CITIES, get_save_name, get_start_and_goal_state

EVAL_INTERVAL = 2000  # node counts are recorded every eval_interval training steps (run_training_on_env default)
OUT_ROOT = 'plots_many_streets'

METHODS = {  # option name in code -> (label, color)
    'hotspot_options': ('NEO', '#2a78d6'),
    'oracle': ('SP-novelty', '#eb6834'),
    'Qlearning-novelty': ('Qlearning-novelty', '#1baf7a'),
}
# fixed color per city (follows the city, not its rank)
CITY_COLORS = dict(zip(CITIES, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']))
SEQ_BLUE = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
EIG_COLORS = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']  # ordinal: 1st..4th smallest eigenvalue
TEXT, TEXT_2, GRID, STREET, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#cfcec9', '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.usetex': False, 'font.size': 10, 'axes.titlesize': 11,
    'text.color': TEXT, 'axes.labelcolor': TEXT_2, 'xtick.color': TEXT_2, 'ytick.color': TEXT_2,
    'axes.edgecolor': GRID, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
    'lines.linewidth': 1.6, 'legend.frameon': False,
})


# ---------------------------------------------------------------- data helpers
def load_results(cities, instances=1):
    results = {}
    for city in cities:
        for option_name in METHODS:
            save_name = get_save_name(option_name, city, instances=instances)
            if not os.path.exists(save_name):
                continue
            with open(save_name, 'rb') as f:
                results.setdefault(city, {})[option_name] = pickle.load(f)['env_results'][(city, option_name)]
    return results


def city_graph(city_res):
    # every method of a city runs on the same (renumbered) graph
    return next(iter(city_res.values()))['final_agent'].G


def nx_pos(G):
    return {n: d['pos'] for n, d in G.nodes(data=True)}


def start_and_goal(G, run=0):
    # same deterministic choice as in training; silence its prints
    with contextlib.redirect_stdout(io.StringIO()):
        return get_start_and_goal_state(run, None, G)


def discovered_fraction(res, G):
    # -> steps (n_points,), % discovered (n_seeds, n_points)
    counts = np.array(res['node_instance_list:'], dtype=float)
    steps = np.arange(counts.shape[1]) * EVAL_INTERVAL
    return steps, 100 * counts / len(G)


def mean_ci(x, level=0.95):
    # mean and half-width of the t-based confidence interval over axis 0 (seeds)
    n = x.shape[0]
    m = x.mean(axis=0)
    if n < 2:
        return m, np.zeros_like(m)
    return m, stats.t.ppf(0.5 + level / 2, n - 1) * x.std(axis=0, ddof=1) / np.sqrt(n)


# ---------------------------------------------------------------- drawing helpers
def grid_axes(n, w, h, **kw):
    ncol = min(4, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), squeeze=False, **kw)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes


def draw_streets(ax, G, color=STREET, lw=0.35):
    pos = nx_pos(G)
    ax.add_collection(LineCollection([(pos[u], pos[v]) for u, v in G.edges()], colors=color, linewidths=lw, zorder=1))
    xs, ys = zip(*pos.values())
    # small margin so start/goal markers on the border are not clipped
    pad_x, pad_y = 0.04 * (max(xs) - min(xs)), 0.04 * (max(ys) - min(ys))
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    # lon/lat -> roughly equal meters on both axes
    ax.set_aspect(1 / math.cos(math.radians(np.mean(ys))))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def mark_start_goal(ax, G, center=None):
    pos = nx_pos(G)
    start, goal = start_and_goal(G)
    for node, marker, text in [(start, 'o', 'start'), (goal, 's', 'goal')]:
        ax.scatter(*pos[node], s=60, marker=marker, color=TEXT, edgecolor=SURFACE, linewidth=1.5, zorder=5)
        ax.annotate(text, pos[node], xytext=(5, 5), textcoords='offset points', fontsize=8, color=TEXT, zorder=6,
                    bbox=dict(boxstyle='round,pad=0.1', fc=SURFACE, ec='none', alpha=0.8))
    if center is not None:
        lat, lon = center
        ax.scatter(lon, lat, s=80, marker='+', color=TEXT_2, linewidth=1.5, zorder=5)


def label_line_ends(ax, ends, min_gap):
    # direct labels at the right end of each line, nudged apart so they don't overlap
    placed = []
    for x, y, text in sorted(ends, key=lambda e: e[1]):
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords='offset points', va='center', fontsize=9, color=TEXT)


def method_legend(fig, y=1.0):
    handles = [Line2D([], [], color=c, label=l) for l, c in METHODS.values()]
    fig.legend(handles=handles, loc='lower center', ncol=len(METHODS), bbox_to_anchor=(0.5, y))


def save(fig, subdir, name):
    os.makedirs(os.path.join(OUT_ROOT, subdir), exist_ok=True)
    path = os.path.join(OUT_ROOT, subdir, name)
    fig.savefig(path, bbox_inches='tight')
    print('wrote', path)
    if not INTERACTIVE:
        plt.close(fig)


# ---------------------------------------------------------------- 1) maps
def plot_city_maps(results):
    cities = list(results)
    fig, axes = grid_axes(len(cities), 3.6, 3.9)
    for ax, city in zip(axes.flat, cities):
        G = city_graph(results[city])
        draw_streets(ax, G, color='#8a8984', lw=0.3)
        mark_start_goal(ax, G, CITIES[city]['center'])
        ax.set_title(f"{city}\n{len(G):,} nodes · {G.number_of_edges():,} edges · r = {CITIES[city]['initial_radius'] / 1000:g} km", fontsize=10)
    fig.tight_layout(h_pad=2.5)
    fig.text(0.5, 0.0, '●  start     ■  goal     +  requested city center', ha='center', va='top', fontsize=9, color=TEXT_2)
    save(fig, 'maps', 'city_maps.pdf')


# ---------------------------------------------------------------- 2) discovery curves
def plot_neo_all_cities(results):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ends = []
    for city, city_res in results.items():
        if 'hotspot_options' not in city_res:
            continue
        steps, frac = discovered_fraction(city_res['hotspot_options'], city_graph(city_res))
        ax.plot(steps / 1000, frac[0], color=CITY_COLORS[city], label=city)
        ends.append((steps[-1] / 1000, frac[0][-1], city))
    label_line_ends(ax, ends, min_gap=3.5)
    ax.set_xlabel('Training steps (thousands)')
    ax.set_ylabel('States discovered (%)')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, None)
    ax.set_title('NEO: exploration of the street graph', loc='left')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    save(fig, 'discovery', 'neo_discovery_all_cities.pdf')


def plot_discovery_per_city(results, subdir='discovery', suffix=''):
    # one panel per city; with several seeds: mean line and 95% CI band
    cities = list(results)
    fig, axes = grid_axes(len(cities), 3.4, 2.8, sharex=True, sharey=True)
    n_seeds = 1
    for ax, city in zip(axes.flat, cities):
        G = city_graph(results[city])
        for option_name, (label, color) in METHODS.items():
            if option_name not in results[city]:
                continue
            steps, frac = discovered_fraction(results[city][option_name], G)
            n_seeds = frac.shape[0]
            m, ci = mean_ci(frac)
            ax.plot(steps / 1000, m, color=color)
            if n_seeds > 1:
                ax.fill_between(steps / 1000, np.clip(m - ci, 0, 100), np.clip(m + ci, 0, 100), color=color, alpha=0.18, linewidth=0)
        ax.set_title(city, loc='left')
        ax.set_ylim(0, 100)
    n = len(cities)
    for i, ax in enumerate(axes.flat[:n]):
        if i % axes.shape[1] == 0:
            ax.set_ylabel('States discovered (%)')
        if i + axes.shape[1] >= n:  # lowest visible panel of its column
            ax.set_xlabel('Training steps (thousands)')
            ax.xaxis.set_tick_params(labelbottom=True)
    fig.tight_layout()
    method_legend(fig)
    if n_seeds > 1:
        fig.text(0.5, -0.01, f'line: mean over {n_seeds} seeds (each seed has its own start and goal) · band: 95% confidence interval',
                 ha='center', va='top', fontsize=9, color=TEXT_2)
    save(fig, subdir, f'discovery_per_city{suffix}.pdf')


# ---------------------------------------------------------------- 3) final coverage
def plot_final_coverage(results, subdir='coverage', suffix=''):
    cities = list(results)
    fig, ax = plt.subplots(figsize=(7.5, 0.75 * len(cities) + 1.2))
    bar_h = 0.26
    n_seeds = 1
    for i, (option_name, (label, color)) in enumerate(METHODS.items()):
        for j, city in enumerate(cities):
            if option_name not in results[city]:
                continue
            _, frac = discovered_fraction(results[city][option_name], city_graph(results[city]))
            final = frac[:, -1]
            n_seeds = len(final)
            m, ci = mean_ci(final[:, None])
            y = j + (i - 1) * (bar_h + 0.03)
            ax.barh(y, m[0], height=bar_h, color=color, label=label if j == 0 else None)
            if n_seeds > 1:
                ax.errorbar(m[0], y, xerr=ci[0], color=TEXT_2, lw=1, capsize=2, zorder=4)
                ax.scatter(final, np.full(n_seeds, y), s=10, color=SURFACE, edgecolor=TEXT, linewidth=0.7, zorder=5)
                x_label = min(105, max(final.max(), m[0] + ci[0]))
            else:
                x_label = m[0]
            ax.annotate(f'{m[0]:.0f}%', (x_label, y), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8, color=TEXT_2)
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels(cities)
    ax.tick_params(axis='y', length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel('States discovered after training (%)')
    ax.grid(axis='y', visible=False)
    title = 'Final exploration coverage'
    if n_seeds > 1:
        title += f'  (mean of {n_seeds} seeds, 95% CI, dots = single seeds)'
    ax.set_title(title, loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1 - 0.4 / len(cities)), ncol=len(METHODS), fontsize=9)
    save(fig, subdir, f'final_coverage{suffix}.pdf')


# ---------------------------------------------------------------- 4) visitation maps
def plot_visitation(results, city):
    city_res = results[city]
    G = city_graph(city_res)
    pos = nx_pos(G)
    vmax = max(r['visitation_mat'].max() for r in city_res.values())
    norm = LogNorm(vmin=1, vmax=max(vmax, 10))
    fig, axes = plt.subplots(1, len(METHODS), figsize=(4.2 * len(METHODS), 4.8))
    sc = None
    for ax, (option_name, (label, _)) in zip(axes, METHODS.items()):
        draw_streets(ax, G)
        if option_name in city_res:
            vm = city_res[option_name]['visitation_mat']
            idx = np.flatnonzero(vm > 0)
            idx = idx[np.argsort(vm[idx])]  # most visited drawn on top
            xy = np.array([pos[n] for n in idx])
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=vm[idx], cmap=SEQ_BLUE, norm=norm, s=4, linewidths=0, zorder=3)
            ax.set_title(f'{label}\n{len(idx) / len(G) * 100:.0f}% of states visited', fontsize=10)
        else:
            ax.set_title(f'{label}\nno results', fontsize=10)
        mark_start_goal(ax, G)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, orientation='horizontal', fraction=0.04, pad=0.03, aspect=50)
        cbar.set_label('Visits per state (log scale)', color=TEXT_2)
        cbar.outline.set_visible(False)
    fig.suptitle(f'{city}: where each method spent its training steps', x=0.02, ha='left', fontsize=11)
    save(fig, 'visitation', 'visitation_' + city.replace(' ', '_') + '.pdf')


# ---------------------------------------------------------------- 5) NEO option updates
def plot_neo_eigenvalues(results):
    # info_instance_list[seed] = [[eigenvalues, visit-count quantiles, step], ...] at every NEO option update
    cities = [c for c in results if 'hotspot_options' in results[c]]
    fig, axes = grid_axes(len(cities), 3.4, 2.8, sharex=True)
    for ax, city in zip(axes.flat, cities):
        info = results[city]['hotspot_options']['info_instance_list'][0]
        steps = np.array([u[2] for u in info]) / 1000
        eig = np.array([np.asarray(u[0])[:len(EIG_COLORS)] for u in info])
        for k in range(eig.shape[1]):
            real = np.abs(eig[:, k].imag) < 1e-8
            ax.plot(steps, eig[:, k].real, color=EIG_COLORS[k], marker='o', ms=5, label=f'λ{k + 1}')
            ax.scatter(steps[~real], eig[~real, k].real, s=40, facecolor=SURFACE, edgecolor=EIG_COLORS[k], zorder=4)
        ax.set_title(city, loc='left')
    n = len(cities)
    for i, ax in enumerate(axes.flat[:n]):
        if i % axes.shape[1] == 0:
            ax.set_ylabel('Re(eigenvalue)')
        if i + axes.shape[1] >= n:
            ax.set_xlabel('Training step of option update (thousands)')
            ax.xaxis.set_tick_params(labelbottom=True)
    fig.tight_layout()
    handles = [Line2D([], [], color=c, marker='o', ms=5, label=f'λ{k + 1}') for k, c in enumerate(EIG_COLORS)]
    handles.append(Line2D([], [], color=TEXT_2, marker='o', ms=6, ls='', markerfacecolor=SURFACE, label='complex eigenvalue'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, -0.01, "Smallest eigenvalues of NEO's modified Laplacian on the explored subgraph, at each option update (seed 0)",
             ha='center', va='top', fontsize=9, color=TEXT_2)
    save(fig, 'options', 'neo_eigenvalues.pdf')


if __name__ == '__main__' or INTERACTIVE:
    parser = argparse.ArgumentParser()
    parser.add_argument('cities', nargs='*', default=list(CITIES))
    parser.add_argument('--instances', type=int, default=5, help='number of seeds of the multi-seed runs used in seeds/')
    args = parser.parse_args([] if INTERACTIVE else None)

    results = load_results(args.cities)
    missing = [c for c in args.cities if c not in results]
    if missing:
        print('no single-seed results yet for:', ', '.join(missing))
    if results:
        plot_city_maps(results)
        plot_neo_all_cities(results)
        plot_discovery_per_city(results)
        plot_final_coverage(results)
        plot_neo_eigenvalues(results)
        for city in results:
            plot_visitation(results, city)

    seed_results = load_results(args.cities, instances=args.instances)
    if seed_results:
        suffix = f'_seeds{args.instances}'
        plot_discovery_per_city(seed_results, subdir='seeds', suffix=suffix)
        plot_final_coverage(seed_results, subdir='seeds', suffix=suffix)
    else:
        print(f'no {args.instances}-seed results yet, skipping seeds/')

# %%
