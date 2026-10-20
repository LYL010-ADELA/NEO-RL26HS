# %%
import copy
import numpy as np
import pickle
import copy
from all_functions import *
from chart_utils import make_plots
from utils import create_graphs
# %load_ext autoreload
# %autoreload 2
import random
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import collections
from PIL import Image
import networkx as nx
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

CITIES = {
    "New York":  {"center": (40.7128, -74.0060), "initial_radius": 7000},
    "Zurich":    {"center": (47.3769,   8.5417), "initial_radius": 9000},
    "Beijing":   {"center": (39.9042, 116.4074), "initial_radius": 8000},
    "Cape Town": {"center": (-33.9249, 18.4241), "initial_radius": 7000},
    "New Delhi": {"center": (28.6139,  77.2090), "initial_radius": 5000},
    "London":    {"center": (51.5074,  -0.1278), "initial_radius": 4000},
    "Paris":     {"center": (48.8566,   2.3522), "initial_radius": 7000},
}

cities = CITIES.keys()

def find_furthest_states(G: nx.Graph):
    # Precompute shortest‐path lengths from every node
    all_lens = dict(nx.all_pairs_shortest_path_length(G))
    
    max_d = -1
    far_pair = (None, None)
    for u, dist_dict in all_lens.items():
        for v, d in dist_dict.items():
            if d > max_d:
                max_d = d
                far_pair = (u, v)
    return far_pair

def get_street_graph(
    center_point=(40.7128, -74.0060),  # New York City Hall
    desired_nodes=10000,
    initial_radius=7000,               # meters; adjust to hit ~6000 nodes
    network_type='drive',
    simplify=True
):
    # 1) Fetch directed graph
    G = ox.graph_from_point(center_point,
                            dist=initial_radius,
                            network_type=network_type,
                            simplify=simplify)
    # 2) Keep only the largest weakly connected component
    G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
    # 3) If it’s too big, sample ~desired_nodes by BFS
    if len(G) > desired_nodes:
        def bfs_sample(G, n):
            start = next(iter(G))
            visited = {start}
            queue = deque([start])
            while queue and len(visited) < n:
                u = queue.popleft()
                for nbr in list(G.successors(u)) + list(G.predecessors(u)):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
                        if len(visited) >= n:
                            break
            return G.subgraph(visited).copy()
        G = bfs_sample(G, desired_nodes)
    # 4) Build pos attribute
    pos = {n: (data['x'], data['y']) for n, data in G.nodes(data=True)}
    nx.set_node_attributes(G, pos, name='pos')
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def evaluate_agent(agent, env, start_state, goal_state, max_time_steps):
    eval_return = 0
    state = env.reset_world(start_state, goal_state)
    for t in range(max_time_steps):
        mas = len(env.get_next_states(state))
        if mas == 0:
            break
        action = agent.option_action(state, False)
        ns, r, done = env.step(action)
        eval_return += r
        state = ns
        if done:
            print('done..')
            break
    return eval_return

def get_start_and_goal_state(run, gw, G):
    # 0) seed the right RNG
    random.seed(run)
    # 1) collect only nodes with at least one successor
    nodes_with_succ = [n for n in G.nodes() if G.out_degree(n) > 0]
    if not nodes_with_succ:
        raise ValueError("Graph has no nodes with successors")
    # 2) pick a random source from that filtered list
    source = random.choice(nodes_with_succ)
    print(f"Source node: {source}")
    # 3) compute shortest‐path lengths (in meters)
    lengths = nx.single_source_dijkstra_path_length(G, source, weight='None')
    if len(lengths) == 1:
        # only itself is reachable!
        raise ValueError(f"No other nodes reachable from source {source}")
    # 4) find the furthest‐away reachable node
    target, max_dist = max(lengths.items(), key=lambda item: item[1])
    print(f"Furthest node: {target}")
    print(f"Distance      : {max_dist:.1f} meters")

    return source, target

def run_training_on_env(G, env_name, option_name, num_options=4, instances=1, max_steps_mul=250, ep_time_horizon=1000, eval_interval=2000, directed=False, option_eps=0.1, nsa_greedy=False, det_option_policy=True, reward_eval=True, RW_LAP=True):
    max_steps_mul = 250
    stochastic = False
    grid_world = True
    G = copy.deepcopy(G)
    directed = True
    grid_world = False
    G = renumber_graph(G)
    pos = nx.get_node_attributes(G,'pos')
    gw = GraphWorld(grid_world, G, stochastic)
    ep_time_horizon = len(G.nodes)
    update_exploration_option_rate = len(G.nodes)*5# 
    if update_exploration_option_rate < 1000: 
        update_exploration_option_rate = 1000
    if len(G.nodes) < 1000: #smaller graph's should need less time
         ep_time_horizon = 500
         max_steps_mul = max_steps_mul * ep_time_horizon #ep_time_horizon
    elif len(G.nodes) < 3000: 
         ep_time_horizon = 1000
         max_steps_mul = max_steps_mul * ep_time_horizon #ep_time_horizon
    elif len(G.nodes) >= 6000: 
         ep_time_horizon = len(G.nodes)
         max_steps_mul = max_steps_mul * 1000 #ep_time_horizon
    elif len(G.nodes) >= 3000: 
         ep_time_horizon = len(G.nodes)
         max_steps_mul = max_steps_mul * 1000 #ep_time_horizon
    print('max_steps_mul: ', max_steps_mul)
    if option_name == 'Qlearning-novelty' or option_name == 'Qlearning':
        option_limit = 0
    else:
        option_limit = 64
    agent = AgentQ2(G, gw, option_limit, policy="epsilon_greedy", epsilon=0.1, alpha=0.4, gamma=0.99,initialization_states=None, option_eps = option_eps, det_option_policy=det_option_policy)
    if reward_eval:
        agent.update_options = True #-------------------------update
    else:
        agent.update_options = False 
    instance_list     = []  # shape: [instances][episodes]
    eval_instance_list = [] # shape: [instances][# of evaluations]
    node_instance_list = []
    edge_instance_list = []
    info_instance_list = []
    for run in range(instances):
        print('len(G.nodes()): ', len(G.nodes()))
        visitation_mat = np.zeros(len(G.nodes()))
        visitation_mat_2 = np.zeros(len(G.nodes()))
        visitation_sa = np.zeros(shape=(gw.num_nodes, gw.max_action_size))
        #go through statespace and set all q_values beyond the number of neighbors to -1
        for node in G.nodes:
            mas = len(gw.get_next_states(node))
            visitation_sa[node][mas: ] = -1
        new_G = nx.DiGraph()
        # Make a fresh copy of 'agent' for each run
        running_agent = copy.deepcopy(agent)
        start_state, goal_state = get_start_and_goal_state(run,gw,G)  #switched
        step_count = 0
        run_rewards = []
        run_eval_rewards = []
        node_counts = []
        edge_counts = []
        info = []
        found_goal = False
        prior_state = start_state
        while(step_count < max_steps_mul):
            state = gw.reset_world(start_state,goal_state) #1500 (large)
            e_rewards = []
            backward_updates_data = []
            for t in range(ep_time_horizon):
                running_agent.visitation_mat = visitation_mat #or step_count == ep_time_horizon
                if (len(new_G.nodes()) > 10) and ((step_count % update_exploration_option_rate == 0 and step_count > 1000)) and option_name != 'Qlearning' and option_name != 'Qlearning-novelty':
                    replace = False
                    _e, count_vec = running_agent.add_option(directed, new_G,visitation_mat, option_name, num_options, replace, option_limit,RW_LAP)
                    if count_vec is not None:
                        q_levels = np.linspace(0, 1, 21)
                        quantile_values = np.quantile(count_vec, q_levels)
                    else:
                        quantile_values = None
                    info.append([_e,quantile_values,step_count])
                visitation_mat[state] += 1
                mas = len(gw.get_next_states(state))
                if mas == 0:
                    print('_i_')
                    break
                action = running_agent.option_action(state, True) 
                ns, reward, done = gw.step(action)
                visitation_sa[state][action] += 1
                if option_name == 'Qlearning-novelty':
                    reward += 0.01/(np.sqrt(visitation_sa[state][action])+0.0000000000000001)
                backward_updates_data.append([state, ns, reward, done, action])
                if reward_eval:
                    running_agent.update_q_values(state, ns, reward, done, action) #-------------------------update
                e_rewards.append(reward)
                    #add edge to new graph
                if state not in new_G.nodes():
                    new_G.add_node(state, pos = pos[state])
                if ns not in new_G.nodes():
                    new_G.add_node(ns, pos = pos[ns])
                if not new_G.has_edge(state, ns) and state != ns:
                    new_G.add_edge(state, ns)
                state = ns
                if step_count % eval_interval == 0:
                    print('step_count: ', step_count, 'number of nodes: ', len(new_G.nodes()) ,'/', len(G.nodes()), 'number of edges: ', len(new_G.to_undirected().edges()) ,'/', len(G.edges()))
                    eval_ret = evaluate_agent(copy.deepcopy(running_agent), copy.deepcopy(gw), start_state, goal_state, ep_time_horizon)
                    run_eval_rewards.append(eval_ret)
                    node_counts.append(len(new_G.nodes()))
                    edge_counts.append(len(new_G.edges()))

                step_count += 1
                mas = len(gw.get_next_states(state))
                if done or len(list(G.successors(state))) == 0 or mas == 0:
                    if not found_goal and reward > 0:
                        found_goal = True
                        print("found goal ", step_count)
                        print('_', running_agent.option_eps)
                        if option_name == 'Qlearning-novelty':
                            epsilon = 0.1
                    if reward_eval:
                        break #-------------------------update
                if step_count >= max_steps_mul:
                    break
            if reward_eval:
                # #in reverse
                for i in range(len(backward_updates_data)-1, -1, -1):
                            old_state, new_state, reward, done, action = backward_updates_data[i]
                            running_agent.update_q_values(old_state, new_state, reward, done, action) #-------------------------update
                
            run_rewards.append(np.sum(e_rewards))
        instance_list.append(run_rewards)
        eval_instance_list.append(run_eval_rewards)
        node_instance_list.append(node_counts)
        edge_instance_list.append(edge_counts)
        info_instance_list.append(info)
    #return instance_list, eval_instance_list, running_agent
        print('len: ', len(new_G.nodes()))
    return instance_list, eval_instance_list, node_instance_list, edge_instance_list, running_agent, visitation_mat, info_instance_list

def main():
    env_names = ['street']
    #option_name = 'Qlearning'
    #option_name = 'hotspot_options' (we refer to our NEO options as hotspot_options in code)
    #option_name = 'oracle' (we refer to SPNovelty options as oracle options in code)
    #option_name = 'cover_options'
    #option_name = 'eigen_options'
    #option_name = 'Qlearning-novelty'
    option_sets = [4]
    reward_eval_set = [True]
    option_names = ['hotspot_options','oracle','Qlearning-novelty']
    nsa_greedy = False
    stochastic = False
    det_option_policy = True
    RW_LAP = True
    results = {'env_results': {}}
    option_eps_set = [0.1]
    for city in cities:
        # build the street graph once per city so every method runs on the same graph
        city_G = get_street_graph(center_point=CITIES[city]['center'], initial_radius=CITIES[city]['initial_radius'])
        for reward_eval in reward_eval_set:
            for option_name in option_names:
                for option_eps in option_eps_set:
                    for num_options in option_sets:
                        for env_name in env_names:
                            print(f"Running training on environment: {env_name}")
                            print('option_eps: ', option_eps)
                            # Train & get results for each environment
                            instance_list, eval_instance_list, node_instance_list, edge_instance_list, last_agent, visitation_mat, info_instance_list = run_training_on_env(city_G, env_name, option_name=option_name, num_options=num_options, directed=False, option_eps=option_eps, nsa_greedy=nsa_greedy, det_option_policy=det_option_policy, reward_eval=reward_eval, RW_LAP=RW_LAP)
                            results['env_results'][(city, option_name)] = {'city': city, 'node_instance_list:': node_instance_list, 'edge_instance_list': edge_instance_list, 'stochastic': stochastic, 'option_name': option_name,'num_options': num_options, 'training_returns': instance_list,'eval_returns': eval_instance_list,'final_agent': last_agent,'visitation_mat': visitation_mat,'info_instance_list': info_instance_list}
                            for k in range(len(eval_instance_list)):
                                print('eval_instance_list: ', np.shape(instance_list[k]))
                            save_name = 'NEO_n_' + option_name + '_street_' + city.replace(' ', '_') + '_' + str(RW_LAP) + '_' + str(nsa_greedy) + '_' + str(det_option_policy) + '__' + str(num_options) + '_' + str(option_eps) + '.pkl'
                            with open(save_name, "wb") as f:
                                pickle.dump({'env_results': {(city, option_name): results['env_results'][(city, option_name)]}}, f)
        
if __name__ == "__main__":
    main()

