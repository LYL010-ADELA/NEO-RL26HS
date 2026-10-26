# %%
import os
import sys
import pickle
import matplotlib
# when run as a plain script (no notebook), draw off-screen and save to file instead
INTERACTIVE = 'ipykernel' in sys.modules
if not INTERACTIVE:
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
_show = plt.show
import numpy as np
import chart_utils
from chart_utils import make_plots
# chart_utils turns on LaTeX text rendering; the tier3 LaTeX install lacks type1cm.sty, so use matplotlib's own renderer
matplotlib.rcParams['text.usetex'] = False
from Many_Streets_Eval import CITIES, get_save_name

# option name in code -> label in the plot
option_labels = {
    'hotspot_options': 'NEO',
    'oracle': 'SP-novelty',
    'Qlearning-novelty': 'Qlearning-novelty',
}

# python plots_many_streets.py                         -> all cities, single-seed results
# python plots_many_streets.py "New Delhi" --instances 5  -> only New Delhi, 5-seed results (mean + confidence band)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('cities', nargs='*', default=list(CITIES))
parser.add_argument('--instances', type=int, default=1)
args = parser.parse_args([] if INTERACTIVE else None)
cities_to_plot, instances = args.cities, args.instances

OUT_DIR = os.path.join('plots_many_streets', 'per_city' if instances == 1 else f'per_city_seeds{instances}')
os.makedirs(OUT_DIR, exist_ok=True)
for city in cities_to_plot:
    print(f"City: {city}")
    option_names = []
    alg = []
    for option_name, label in option_labels.items():
        save_name = get_save_name(option_name, city, instances=instances)
        if not os.path.exists(save_name):
            print(f"  missing {save_name}, skipping")
            continue
        with open(save_name, "rb") as f:
            results = pickle.load(f)
        res = results['env_results'][(city, option_name)]
        print(f"  Option Name: {option_name}")
        print('  number of nodes: ', len(res['final_agent'].G.nodes()))
        print('  number of edges: ', len(res['final_agent'].G.edges()))
        alg.append(res['node_instance_list:'])
        #alg.append(res['edge_instance_list'])
        option_names.append(label)
    if len(alg) == 0:
        continue
    plt.rcParams['figure.figsize'] = [6, 6]
    out_file = os.path.join(OUT_DIR, city.replace(' ', '_') + '.pdf')

    def finish_and_show(out_file=out_file):
        # chart_utils.plot() calls show() and then closes the figure, so fix the layout and save at show() time
        ax = plt.gca()
        ax.set_xlabel('Evaluation (every 2000 training steps)')
        # chart_utils puts the legend lower left, on top of the curves -> move it below the plot
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=11)
        if INTERACTIVE:
            _show()
        else:
            plt.savefig(out_file, bbox_inches='tight')
    chart_utils.pyplot.show = finish_and_show
    make_plots(alg, option_names, plot_title=city + ' (Directed)', cumulative=False, episodic=False, track_disc_reward=False, open_plot=False, plot_y='States discovered')

# %%
