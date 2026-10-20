import itertools
import math
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering, k_means
from sklearn.dummy import check_random_state
from sklearn.kernel_approximation import pairwise_kernels
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import tqdm
from utils import convert_to_degree_row
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import networkx as nx
import copy
from numpy import linalg as LA
import scipy as sp
from scipy.linalg import polar
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering
from utils import convert_to_degree_row
import networkx as nx
from sklearn.cluster import KMeans
import scipy
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
import networkx as nx
from collections import defaultdict
import copy

def add_stoc(M, G):
    #get g nodes 
    g_nodes = []
    for i in G.nodes:
        g_nodes.append(i)
    #get positions 
    pos = nx.get_node_attributes(G,'pos')
    #seperate to x and y 
    x = []
    y = []
    for i in g_nodes:
        x.append(pos[i][0])
        y.append(pos[i][1])
    #go through adhcent nodes, if an adjacent node has a higher y value, add a stoc weight to the edge
    for i in range(len(M)):
        stoc_prob = (1/4)*1
        prior_sum = np.sum(M[i])
        prior_probs = M[i] * (1-stoc_prob)
        for k in range(len(M)):
            if M[i][k] != 0 and i != k:
                if y[k] > y[i]:
                    M[i][k] += prior_sum * stoc_prob
                    break
    # for i in range(len(M)):
    #     #prev_sum = np.sum(M[i])
    #     for k in range(len(M)):
    #         if M[i][k] != 0 and i == k-1:
    #             stoc_weight = (1/3)*(4)
    #             M[i][k] += stoc_weight
    return M




def renumber_graph(G_in):
    mapping = dict(zip(G_in.nodes(), range(0, len(G_in.nodes()))))
    G_in = nx.relabel_nodes(G_in, mapping)
    return G_in

def calculate_shortest_path_matrix(G, include_all_paths = True):
    SP_list = []
    l = len(list(G.nodes))
    for r_state in range(l):
        SP_list.append([])
        for c_state in range(l):
            SP_list[r_state].append([])
    SPD = np.empty(  (l, l)  )
    print('calculating shortest path distances...')
    for r_state in range(l):
        for c_state in range(l):
            try:
                if include_all_paths:
                    SP_list[r_state][c_state].append([p for p in nx.all_shortest_paths(G, source=r_state, target=c_state)])
                SPD[r_state][c_state] = -nx.shortest_path_length(G, source=r_state, target=c_state)
            except:
                SPD[r_state][c_state] = 0
                if include_all_paths:
                    SP_list[r_state][c_state] = [[[9999],[9999]]]
    if include_all_paths:
        return SPD, SP_list
    else:
        return SPD

def get_all_shortest_path_dists(env_name, G):
    try:
        with open(env_name + '.SPD.data', 'rb') as handle:
            SPD = pickle.load(handle)
    except OSError as err:
        SPD = calculate_shortest_path_matrix(G, False)
        with open(env_name + '.SPD.data', 'wb') as handle:
            pickle.dump(SPD, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved..')
    return SPD

def exponentiate(M, exp):
		numRows = len(M)
		numCols = len(M[0])
		expM = np.zeros((numRows, numCols))
		for i in range(numRows):
			for j in range(numCols):
				if M[i][j] != 0:
					expM[i][j] = M[i][j]**exp
		return expM

class GraphWorld:
    def __init__(self,gridworld, G ,stochastic=False,goal_reward=100, M=None):
        self.G = G
        self.G = renumber_graph(self.G)
        if stochastic:
            M = nx.adjacency_matrix(G).todense().astype(np.double)
            M = add_stoc(M,G)
        self.num_nodes = len(list(self.G.nodes()))
        self.agent_pos = np.random.randint(0, self.num_nodes)
        self.observation_space = np.arange(self.num_nodes)
        self.goal = len(self.G.nodes()) 
        self.gridworld = gridworld
        if self.gridworld:
            for i in self.G.nodes:
                mas = len(list(self.G.neighbors(i)))
                if mas < 4:
                    for k in range(4 - mas):
                        self.G.add_edge(i, i)

        self.max_action_size = max([len(self.get_next_states(node)) for node in self.G.nodes()])
        self.action_space = np.arange(self.max_action_size)
        self.goal_reward = goal_reward
        self.M = M
        self.stochastic = stochastic
        pass


    def reset_world(self, start_state = None, goal_state = None):
        if start_state is None:
            self.agent_pos = np.random.randint(0, self.num_nodes)
        else:
            self.agent_pos = start_state

        if goal_state is None:
             self.goal = np.random.randint(0, self.num_nodes)
        else:
             self.goal = goal_state

        return self.agent_pos
    
    def step(self, action):
        neighbor_nodes = list(self.G.neighbors(self.agent_pos))
        
        if self.stochastic:
            rnd = np.random.random()
            if rnd <= 1/4:
                #get action where self.M[self.agent_pos][neighbor_nodes[action]] is highest
                max_val = -np.inf
                for i in range(len(neighbor_nodes)):
                    if self.M[self.agent_pos][neighbor_nodes[i]] > max_val:
                        max_val = self.M[self.agent_pos][neighbor_nodes[i]]
                        action = i

        if action < len(neighbor_nodes):
            self.agent_pos = neighbor_nodes[action]

        observation = self.agent_pos

        reward = 0
        done = False
        if self.agent_pos == self.goal:
            reward = self.goal_reward
            done = True
        return observation, reward, done
    
    def get_next_states(self, state, gw=None):
        mas = list(self.G.neighbors(state))
        if self.gridworld:
            if len(mas) < 4:
                for k in range(4 - len(mas)):
                    mas.append(state) 
        return mas

    def get_next_state_action(self, state, next_state, gw=None):
        next_states = list(self.G.neighbors(state))
        mas = list(self.G.neighbors(state))
        if self.gridworld:
            if len(mas) < 4:
                for k in range(4 - len(mas)):
                    next_states.append(state) 
        if next_state in list(self.G.neighbors(state)):
            return list(self.G.neighbors(state)).index(next_state)  
    
    def get_list_of_neighbors(self, state):
        return list(self.G.neighbors(state))
    
    def random_action(self):
        #get the number of neighbors of self.agent_pos
        neighbor_nodes = list(self.G.neighbors(self.agent_pos))
        #randomly select one of the neighbors
        action = np.random.randint(0, len(neighbor_nodes))
        return action
    
    def stochastic_action(self, action):
        current_state = self.agent_pos
        next_states_ = self.get_next_states(current_state)
        potential_action = action
        act = 0
        for ns in next_states_:
            if ns == (current_state -1):
                potential_action = act
            act += 1
        rnd = np.random.random()
        if rnd <= 1/3:
            return potential_action
        else:       
            return action
            
def convert_to_sums_of_discounted_returns(returns):
    sums = np.zeros(len(returns))
    for k in range(len(returns)):
        for i in range(k, len(returns)):
            sums[k] += returns[i] * 0.99 ** (i - k)
    return sums

def streetmap_graph():
    import networkx as nx
    import osmnx as ox
    import matplotlib.pyplot as plt
    ox.config(use_cache=True, log_console=True)
    point = (51.469058, -0.018071)
    G = ox.graph_from_point(point, dist=750)
    G = renumber_graph(G)
    print('streetmap: ', len(list(G.nodes)))
    pos = nx.spectral_layout(G)
    nx.set_node_attributes(G, pos, "pos")
    return G.to_undirected()
    
class TowersOfHanoi:
    def __init__(self, state):
        self.state = state              # "State" is a tuple of length N, where N is the number of discs, and the elements are peg indices in [0,1,2]
        self.discs = len(self.state)
    def discs_on_peg(self, peg):
        return [disc for disc in range(self.discs) if self.state[disc] == peg]
    def move_allowed(self, move):
        discs_from = self.discs_on_peg(move[0])
        discs_to = self.discs_on_peg(move[1])
        if discs_from:
            return (min(discs_to) > min(discs_from)) if discs_to else True
        else:
            return False
    def get_moved_state(self, move):
        if self.move_allowed(move):
            disc_to_move = min(self.discs_on_peg(move[0]))
        moved_state = list(self.state)
        moved_state[disc_to_move] = move[1]
        return tuple(moved_state)
def generate_reward_matrix(G,N):      # N is the number of discs
    states = list(itertools.product(list(range(3)), repeat=N))
    moves = list(itertools.permutations(list(range(3)), 2))
    R = pd.DataFrame(index=states, columns=states, data=-np.inf)
    for state in states:
        tower = TowersOfHanoi(state=state)
        for move in moves:
            if tower.move_allowed(move):
                next_state = tower.get_moved_state(move)
                G.add_edges_from([(str(state), str(next_state))])    
                R[state][next_state] = 0
    final_state = tuple([2]*N)          # Define final state as all discs being on the last peg
    R[final_state] += 100               # Add a reward for all moves leading to the final state
    return R.values
def get_tow_graph():
    G = nx.DiGraph()
    generate_reward_matrix(G,6)
    G = renumber_graph(G)
    print('toh: ', len(list(G.nodes)))
    pos = nx.spectral_layout(G)
    nx.set_node_attributes(G, pos, "pos")
    return G.to_undirected()
def renumber_graph(G):
    mapping = dict(zip(G.nodes(), range(0, len(G.nodes()))))
    G = nx.relabel_nodes(G, mapping)
    return G
def get_internet_graph():
    G = nx.random_internet_as_graph(100, seed=999)
    print('internet: ', len(list(G.nodes)))
    pos = nx.spectral_layout(G)
    nx.set_node_attributes(G, pos, "pos")
    return G.to_undirected()
def get_sudoku_graph():
    G = nx.sudoku_graph(n=6)
    G = renumber_graph(G)
    print('sudoku: ', len(list(G.nodes)))
    pos = nx.spectral_layout(G)
    nx.set_node_attributes(G, pos, "pos")
    return G.to_undirected()


def image_to_graph_loader(env_name):
    #load a 2d black and white image and display it as a graph 
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    from all_functions import renumber_graph
    from utils import create_graphs
    if env_name == 'Hex':
        image_path = 'Hex.png'
    if env_name == 'DoubleHex':
            image_path = 'DoubleHex.png'
    if env_name == 'Hex_half':
        image_path = 'Hex_half.png'
    if env_name == 'Office_half':
        image_path = 'Office_half.png'
    if env_name == 'large_maze':
        image_path = 'large_maze.png'
    if env_name == 'double_large_maze_portion':
        image_path = 'double_large_maze_portion.png'
    if env_name == 'double_large_maze':
        image_path = 'double_large_maze.png'
    if env_name == 'quad_maze':
        image_path = 'quad_maze.png'
    if env_name == 'office':
        image_path = '2_floor_image.png'
    if env_name == 'double_office':
        image_path = '4_floor_image.png'
    img = mpimg.imread(image_path)
    print(img.shape)
    img = img * 255
    matrix = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j][0] == 0 and img[i][j][1] == 0 and img[i][j][2] == 0:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0
    G, G2 = create_graphs(env_name, matrix)
    G2 = G2.to_undirected()
    G = G.to_undirected()
    states_to_add = []
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j][0] == 0 and img[i][j][1] == 100 and img[i][j][2] == 0:
                states_to_add.append((i*img.shape[1] + j))
    states_to_add = np.array(states_to_add)
    #connect state together from states_to_add in the graph G
    for i in range(0, len(states_to_add)):
        for j in range(0, len(states_to_add)):
            if i != j:
                G.add_edge(states_to_add[i], states_to_add[j])
    G = renumber_graph(G)
    return G.to_undirected()

class option:
    def __init__(self, env, G, _v, new_G, det_option_policy, initialization_states=None):
        self.num_actions = env.max_action_size
        self.q_table = np.zeros(shape=(env.num_nodes, self.num_actions))
        self.env = env
        self.G = G
        self._v = _v
        self.max_v = max(_v)
        self.new_G = copy.deepcopy(new_G)
        self.old_states = []
        self.det_option_policy = det_option_policy
        self.associated_Q_updated = False
        i = 0
        for node in new_G.nodes:
            self.new_G.nodes[node]['ind'] = i
            i += 1
        self.term_cond = True
        pass
        
    def print_new_G_nodes_and_state(self, state):
        print('new_G: ', self.new_G.nodes)
        print('state: ', state)

    def plot_v(self):
        pos = nx.get_node_attributes(self.new_G, 'pos')
        nx.draw(self.new_G, pos, node_color=self._v**(1/20), node_size=50, with_labels=False)
        plt.show()

    def take_option_action(self, state):
        if state in self.new_G.nodes:
            if self._v[self.new_G.nodes[state]['ind']] == np.max(self._v):
                return -1 
        else:
            return -1
        self.old_states.append(state)
        mas = len(self.env.get_next_states(state))
        action_probs = np.ones(mas)*0 #-1000000# + 0.000000000001
        i = 0
        found_state = False
        for next_state in self.env.get_next_states(state):
            diff = 0 
            if state in self.new_G.nodes and next_state in self.new_G.nodes:
                s_indx = self.new_G.nodes[state]['ind']
                ns_indx = self.new_G.nodes[next_state]['ind']
                diff = self._v[ns_indx] - self._v[s_indx]
                if self._v[ns_indx] > self._v[s_indx]:
                    found_state = True  
                if diff < 0:
                    diff = 0
                action_probs[i] = diff# + 0.000000000000001
            i += 1
        if found_state:    
            if self.det_option_policy:
                argmax = np.argmax(action_probs)
                action = argmax
            else:
                action_list = np.arange(0, len(action_probs))
                action = np.random.choice(action_list, p=action_probs / np.sum(action_probs))
        else:
            return -1
        return action

class AgentQ2:
    def __init__(self, G, env, num_options, policy="epsilon_greedy", epsilon=0.1, alpha=0.4, gamma=0.99,initialization_states=None, options=None, option_eps = 0.1, det_option_policy = True):
        self.G = G
        self.num_actions = env.max_action_size
        self.num_options = num_options
        self.q_table = np.zeros(shape=(env.num_nodes, self.num_actions + self.num_options))
        #go through statespace and set all q_values beyond the number of neighbors to -1
        for node in G.nodes:
            mas = len(env.get_next_states(node))
            self.q_table[node][mas: ] = -0.00001
        self.q_table = self.q_table.tolist()
        self.policy = policy
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.one_m_alpha = (1 - self.alpha)
        self.initialization_states = initialization_states
        self.option_type = policy
        self.eps = epsilon
        from collections import deque
        self.option_list = deque(maxlen=num_options)
        self.option_representations = []
        self.active_option = -1
        self.option_eps = option_eps
        self.option_start_states = []
        self.option_rewards_list = []
        self.env = env
        self.act_randomly = True
        self.internal_steps = 0
        self.option_init_step = 0
        self.update_options = True
        self.det_option_policy = det_option_policy
        self.prev_co_links = []

    def option_init(self,action, state):
        if self.active_option == -1 and action >= self.num_actions: #does there need to be a -1 here?
            self.option_start_states = [state]
            self.option_init_step = self.internal_steps
            self.active_option = action - self.num_actions
            return self.option_list[self.active_option].take_option_action(state)
        else:
            return action

    def option_termination(self, action, state, reward=0):
        if action != -1:
            if self.active_option != -1:
                self.option_start_states.append(state)
            return action
        else:
            if state != self.option_start_states[0] and (self.update_options == True):
                    if len(self.option_list) > 0 and self.active_option != -1:
                        self.option_list[self.active_option].associated_Q_updated = True
                    for i in range(1):
                        step_diff = self.internal_steps - self.option_init_step + i
                        old_state = self.option_start_states[i]
                        max_q_value_in_new_state = max(self.q_table[state][0:self.num_actions + self.num_options])
                        option_action = self.active_option + self.num_actions
                        current_q_value = self.q_table[old_state][option_action]
                        self.q_table[old_state][option_action] = (self.one_m_alpha) * current_q_value + self.alpha * (
                                    (self.gamma**step_diff) * reward + (self.gamma**(step_diff+1)) * max_q_value_in_new_state)
                            
            self.active_option = -1
            return self.option_action(state)
        
    def option_action(self, state, option_explore = True):
        self.internal_steps += 1
        if self.active_option != -1:
            action = self.option_list[self.active_option].take_option_action(state)
        else:
            action = self.take_prim_action(state)
            if option_explore:
                if np.random.random() < (self.option_eps) and len(self.option_list) > 0:
                        action = np.random.choice(range(len(self.option_list))) + self.num_actions


        return self.option_termination(self.option_init(action,state), state)
    
    def take_prim_action(self, state):
        mas = len(self.env.get_next_states(state))
        rnd = np.random.random()
        if self.act_randomly and rnd <= self.eps:
            action = np.random.randint(0, mas)
        else:
            q_values_of_state = self.q_table[state]
            max_val = max(q_values_of_state)
            arg_lists = []
            added_lists = []
            for i in range(0,len(q_values_of_state)):
                if q_values_of_state[i] == max_val:
                    added_lists.append(i)
            action = np.random.choice(added_lists)
        return action


    def update_q_values(self, old_state,new_state,reward,done,action):
        if done:
            current_q_value = self.q_table[old_state][action]
            self.q_table[old_state][action] = (self.one_m_alpha) * current_q_value + self.alpha * (reward)
            if len(self.option_list) > 0 and self.active_option != -1:
                self.option_list[self.active_option].associated_Q_updated = True
                self.option_termination(-1,new_state,reward)
        else:
            mas = len(list(self.G.neighbors(new_state)))
            max_q_value_in_new_state = max(self.q_table[new_state][0:self.num_actions + self.num_options])
            current_q_value = self.q_table[old_state][action]
            self.q_table[old_state][action] = (self.one_m_alpha) * current_q_value + self.alpha * (
                    reward + self.gamma * max_q_value_in_new_state)
            

    def add_option(self, directed, new_G, visitation_mat, method, num_options=4, replace=False, option_limit=64, RW_LAP=True, beta=0.5,k_val=64):
        visit_mat_float = []
        for i in range(len(visitation_mat)):
            visit_mat_float.append(visitation_mat[i])
        visit_mat_float = np.array(visit_mat_float) 
        self.visit_mat_float = visit_mat_float / np.max(visit_mat_float)
        self.method = method
        #print('adding options')
        num_of_nodes = len(new_G.nodes)
        if not directed:
            new_G = new_G.to_undirected()
        print('is_directed:', nx.is_directed(new_G))
        node_idxs = []
        for node in new_G.nodes:
            node_idxs.append(node)
            if new_G.has_edge(node, node):
                new_G.remove_edge(node, node)
        M = nx.adjacency_matrix(new_G).todense().astype(np.double)
        for j in range(len(M)):
            M[j] = convert_to_degree_row(M[j])  
        I = np.identity(len(M))
        P = M
        L = I - P
        if method == 'hotspot_options': #hotspot_options = NEO options
            if RW_LAP or directed:
                L = (I - P).astype(np.double)
            else:
                L = nx.laplacian_matrix(new_G).todense().astype(np.double)
        else:
            L = nx.normalized_laplacian_matrix(new_G).toarray().astype(np.double)
        i = 0
        count_vec = []
        count_vec_2d = []
        for node in new_G.nodes:
            count_vec.append(visitation_mat[node])
            count_vec_2d.append([visitation_mat[node],node])
            i += 1
        sorted_visitation_count = sorted(count_vec_2d, key=lambda x: x[0], reverse=False)
        first_N = sorted_visitation_count[0:num_options]
        arg_lowest = np.argsort(count_vec)[0:num_options]#num_options*4]
        rand = np.random.choice([0,])
        count_vec = np.array(count_vec)
        count_vec = (count_vec / np.max(count_vec))
        rand = np.random.random()
        if rand < 0.01:
            rand = 0.01
        if method == 'hotspot_options':
            for i in range(len(count_vec)):
                    L[i][i] += ((count_vec[i])**(1/k_val))*beta#
        if method == 'eigen_options' or method == 'hotspot_options':
            if method == 'eigen_options' and directed:
                U, L = polar(L,side='left')
            if method == 'hotspot_options' or method == 'eigen_options':# and not directed:
                L_sparse = csr_matrix(L.astype(np.double))
                e, v = eigs(L_sparse, k=num_options, which='SR')
            else:
                e, v = LA.eig(np.array(L).astype(np.double))
            o = e.argsort()
            _e = e[o]
            _v = v[:, o].T

        if method  == 'cover_options':
            cover_new_G = copy.deepcopy(new_G)
            options = []
            N = num_options
            fiedler_vecs = []
            while(len(fiedler_vecs)<N):
                        vals = nx.fiedler_vector(cover_new_G, weight='weight', normalized=False, tol=1e-08, method='tracemin_pcg', seed=None)
                        highest_point = node_idxs[np.argmax(vals)]
                        lowest_point = node_idxs[np.argmin(vals)]
                        for bool in [False, True]:
                            if len(options) < N:
                                if bool:
                                    fiedler_vecs.append(vals)
                                else:
                                    fiedler_vecs.append(-vals)
                        cover_new_G.add_edge(lowest_point, highest_point, weight=1)
                        cover_new_G.add_edge(highest_point, lowest_point, weight=1)
        vec_list = []

        for i in range(num_options):
            if method == 'eigen_options':
                if i % 2 == 0 and i != 0:
                    vec = -_v[i-1]
                else:
                    vec = _v[i]
            if method == 'hotspot_options':
                vec = np.abs(_v[i])
                vec = vec
                vec = ((vec / np.max(vec)))#**(1/32)
            if method == 'cover_options':
                vec = np.array(fiedler_vecs[i])

            if method == 'oracle': #oracle_options = SPNovelty options
                lengths = dict(nx.single_target_shortest_path_length(new_G, first_N[i][1]))
                lengths_def = defaultdict(int, lengths)
                vec = np.array([lengths_def[node] for node in new_G.nodes()])
                vec = -vec
                blank_option = option(self.env, self.G, vec, new_G, self.det_option_policy)  
            else:
                blank_option = option(self.env, self.G, vec, new_G, self.det_option_policy)
            vec_list.append(vec)          
            if replace and len(self.option_list) >= i + 1:
                self.option_list[i] = blank_option
                self.option_representations[i] = vec
            else:
                self.option_list.append(blank_option)
                self.option_representations.append(vec)

        pos=nx.get_node_attributes(new_G,'pos')
        for node in self.G.nodes:
            for op in range(len(self.option_list)):
                if self.option_list[op].associated_Q_updated == False:
                    self.q_table[node][self.num_actions + op] = -0.001
        if method == 'hotspot_options':
            return _e,count_vec
        else:
            return None,None

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
import networkx as nx
import numpy as np

def check_reals(e_sorted,count_vec):
    first20 = e_sorted[:4]
    tol = 1e-8
    real_mask = np.abs(first20.imag) < tol
    # 5) count and compute percentage
    num_real = np.count_nonzero(real_mask)
    total     = first20.shape[0]
    percent_real = num_real / total * 100
    real_eigenvalues = first20[real_mask].real
    print(f"\n{num_real} out of {total} are (approximately) real — {percent_real:.2f}%")
    q_levels = np.linspace(0, 1, 21)            # [0.00, 0.05, …, 0.95, 1.00]
    quantile_values = np.quantile(count_vec, q_levels)


def find_furthest_states(G: nx.Graph):
    """
    Returns a tuple (u, v) of two nodes in G whose shortest-path distance
    is maximal among all pairs. If there are multiple such pairs, returns
    one of them arbitrarily.
    """
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
    
def torus():
    def image_to_directed_torus_graph(image_path):
        from PIL import Image
        """
        Load an 80x80 black-and-white PNG and convert to a directed torus graph with one-way wrap:
        - White pixels (>128) become nodes; black pixels are walls (no node).
        - Directed edges to the 4-neighbors (East, West, North, South).
        - Wrap-around only from the rightmost -> leftmost and topmost -> bottommost.
        - Node positions stored in 'pos' attribute as (x, y).
        """
        img = Image.open(image_path).convert("L")
        width, height = img.size
        assert width == height == 80, "Image must be 80x80"
        data = img.load()
        
        G = nx.DiGraph()
        # Add nodes for white pixels
        for x in range(width):
            for y in range(height):
                if data[x, y] > 128:
                    G.add_node((x, y), pos=(x, y))
        
        # Add directed edges with one-way wrap
        for x, y in list(G.nodes()):
            # East (wrap only at rightmost)
            nx_x = (x + 1) if (x + 1) < width else 0
            if (nx_x, y) in G:
                G.add_edge((x, y), (nx_x, y))
            # West (no wrap at leftmost)
            if x - 1 >= 0 and ((x - 1), y) in G:
                G.add_edge((x, y), (x - 1, y))
            # North (wrap only at topmost)
            nx_y = (y + 1) if (y + 1) < height else 0
            if (x, nx_y) in G:
                G.add_edge((x, y), (x, nx_y))
            # South (no wrap at bottommost)
            if y - 1 >= 0 and (x, y - 1) in G:
                G.add_edge((x, y), (x, y - 1))
        
        return G
    demo_path = "torus.png"
    G = image_to_directed_torus_graph(demo_path)
    # Verify wraps
    print("Successors of (79, 23) [should include (0, 23)]:", sorted(G.successors((79, 23))))
    print("Successors of (0, 23) [should NOT include (79, 23)]:", sorted(G.successors((0, 23))))
    print("Successors of (10, 79) [should include (10, 0)]:", sorted(G.successors((10, 79))))
    print("Successors of (10, 0) [should NOT include (10, 79)]:", sorted(G.successors((10, 0))))
    return G

def four_rooms_directed():
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    # Helper: normalize row to sum to 1
    def normalize_row(row):
        s = row.sum()
        return row / s if s > 0 else row
    # 1. Build directed four-rooms domain
    G = nx.grid_2d_graph(8, 8)
    # remove walls, keep doorways
    for r in [3]:
        for c in range(8):
            if c not in [1, 5] and G.has_edge((r, c), (r+1, c)):
                G.remove_edge((r, c), (r+1, c))
    for c in [3]:
        for r in range(8):
            if r not in [1, 5] and G.has_edge((r, c), (r, c+1)):
                G.remove_edge((r, c), (r, c+1))
    mapping = {coord: i for i, coord in enumerate(G.nodes())}
    inverse_mapping = {i: coord for coord, i in mapping.items()}
    G = nx.relabel_nodes(G, mapping)
    pos = {i: (coord[1], 7-coord[0]) for coord, i in mapping.items()}
    # one-way door edges
    door_edges = {
        (mapping[(3,1)], mapping[(4,1)]),
        (mapping[(4,5)], mapping[(3,5)]),
        (mapping[(1,4)], mapping[(1,3)]),
        (mapping[(5,3)], mapping[(5,4)])
    }
    G_dir = nx.DiGraph()
    for u, v in G.edges():
        if (u, v) in door_edges:
            G_dir.add_edge(u, v)
        elif (v, u) in door_edges:
            G_dir.add_edge(v, u)
        else:
            G_dir.add_edge(u, v)
            G_dir.add_edge(v, u)
    # Assign 'pos' attribute to each node
    nx.set_node_attributes(G_dir, pos, 'pos')
    
    return G_dir

def rubiks():
    import networkx as nx
    import matplotlib.pyplot as plt
    from collections import deque
    # Build the 2×2×2 cube graph up to depth 3
    solved_state = (tuple(range(8)), (0,)*8)
    cp_ccw = {
        'U': [3,0,1,2,4,5,6,7], 'R': [4,1,2,0,7,5,6,3],
        'F': [1,5,2,3,0,4,6,7], 'D': [0,1,2,3,5,6,7,4],
        'L': [0,2,6,3,4,1,5,7], 'B': [0,1,3,7,4,5,2,6]
    }
    co_ccw = {
        'U':[0]*8, 'R':[2,0,0,1,1,0,0,2], 'F':[1,2,0,0,2,1,0,0],
        'D':[0]*8, 'L':[0,1,2,0,0,2,1,0], 'B':[0,0,1,2,0,0,2,1]
    }
    def apply_move(state, cp, co):
        perm, orient = state
        return (
            tuple(perm[cp[i]] for i in range(8)),
            tuple((orient[cp[i]] + co[i]) % 3 for i in range(8))
        )
    # Define quarter-turn moves
    move_fns = {}
    for face in cp_ccw:
        cp, co = cp_ccw[face], co_ccw[face]
        move_fns[f"{face}'"] = lambda s, cp=cp, co=co: apply_move(s, cp, co)
    for face in cp_ccw:
        fn = move_fns[f"{face}'"]
        move_fns[face] = lambda s, fn=fn: fn(fn(fn(s)))
    # BFS to depth 3
    dist = {solved_state: 0}
    edges = []
    queue = deque([solved_state])
    while queue:
        s = queue.popleft()
        if dist[s] == 3: continue
        for fn in move_fns.values():
            ns = fn(s)
            if ns not in dist:
                dist[ns] = dist[s] + 1
                queue.append(ns)
            edges.append((s, ns))
    # Create graph
    G = nx.Graph()
    G.add_edges_from(edges)
    # Compute spring layout positions
    pos = nx.spring_layout(G)
    # Assign 'pos' attribute to each node
    nx.set_node_attributes(G, pos, 'pos')
    # Draw using the stored 'pos' attribute
    stored_pos = nx.get_node_attributes(G, 'pos')
    return G

def threed_graph():
    import re
    import networkx as nx
    import matplotlib.pyplot as plt
    # Read and parse the data file
    filename = 'c_graph.txt'
    G = nx.DiGraph()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(
                r'FaceNum:\s*(\d+)\s+Center:\s*\[\s*([-0-9.,\s]+)\]\s+AllowedCandidates:\s*(.*)',
                line
            )
            if not m:
                continue
            face = int(m.group(1))
            center_vals = [float(v) for v in m.group(2).split(',')]
            #print('center_vals:', center_vals)
            cand_str = m.group(3).strip()
            candidates = [int(x.strip()) for x in cand_str.split(',')] if cand_str and cand_str.lower() != 'none' else []
            
            # Add node with position attribute
            G.add_node(face, pos=tuple(center_vals[0:3]))
            
            # Add directed edges for each allowed candidate
            for c in candidates:
                G.add_edge(face, c)
    
    # Information
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    print("\nFirst 10 nodes with positions:")
    for n, pos in list(G.nodes(data='pos'))[:10]:
        print(f"Node {n}: pos={pos}")
    
    # Optional: draw the graph (may be large!)
    pos = nx.get_node_attributes(G, 'pos')
    G.remove_edges_from(nx.selfloop_edges(G))
    return G
    

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

