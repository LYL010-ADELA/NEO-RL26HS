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
import numpy as np
from chart_utils import make_plots
from Many_Streets_Eval import CITIES

# must match the settings used in Many_Streets_Eval.main()
RW_LAP = True
nsa_greedy = False
det_option_policy = True
num_options = 4
option_eps = 0.1
OUT_DIR = 'plots_many_streets'

# option name in code -> label in the plot
option_labels = {
    'hotspot_options': 'NEO',
    'oracle': 'SP-novelty',
    'Qlearning-novelty': 'Qlearning-novelty',
}

def get_save_name(option_name, city):
    # same naming scheme as in Many_Streets_Eval.main()
    return 'NEO_n_' + option_name + '_street_' + city.replace(' ', '_') + '_' + str(RW_LAP) + '_' + str(nsa_greedy) + '_' + str(det_option_policy) + '__' + str(num_options) + '_' + str(option_eps) + '.pkl'

os.makedirs(OUT_DIR, exist_ok=True)
for city in CITIES:
    print(f"City: {city}")
    option_names = []
    alg = []
    for option_name, label in option_labels.items():
        save_name = get_save_name(option_name, city)
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
    make_plots(alg, option_names, plot_title=city + ' (Directed)', cumulative=False, episodic=False, track_disc_reward=False, open_plot=False)
    if not INTERACTIVE:
        plt.savefig(os.path.join(OUT_DIR, city.replace(' ', '_') + '.pdf'), bbox_inches='tight')
        plt.close('all')

# %%
