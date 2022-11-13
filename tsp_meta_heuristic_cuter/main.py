import math
import os
import time
from pprint import pprint
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

# the particular packages developed in this project
import load_tsp_data
from algorithms import *
# from tsp_meta_heuristic_cuter.algorithms import ga, tb, sa
from evolution_views import gen_search_evolution

# from tsp_meta_heuristic_cuter.algorithms.tsp import *

if not os.path.exists('results'):
    os.makedirs('results')

# generate and load tsp data for searching
# pos, adj_mat, opt_cost, no_loc = load_tsp_data.distance_matrix_org()

problem_id = 3  #1: swiss; 2:a280(c); 3:berlin(c); 4:ch130(c); 5:brg180; 6: ulysses22(c)
pos, adj_mat, opt_cost, no_loc,  = load_tsp_data.distance_matrix_frm_tsplib(problem_id)


# initialization
num_tests = 1  # number of iid tests
test_result = []
result = {'best_sol': [], 'best_cost': math.inf,
          'optimal_cost': opt_cost, 'best_gap': math.inf,
          'cost': [0] * num_tests, 'time': [0] * num_tests,
          'avg_cost': math.inf, 'avg_gap': math.inf, 'cost_std': math.inf,
          'avg_time': math.inf, 'time_std': math.inf}
best_cost = math.inf
best_gap = math.inf
best_sol = []
data = {}

# set method
method = 'ts'  # tabu search
# method = 'ga'  # genetic algorithm
# method = 'sa'  # simulated annealing

# set mutation method (random perturbation)
# mut_md = [get_new_sol_swap, get_delta_swap]
mut_md = [get_new_sol_2opt, get_delta_2opt]

# start run the search process
for _ in tqdm(range(num_tests)):
    start = time.process_time()
    match method:
        case 'ts':
            method_name = 'Tabu Search'
            best_sol, best_cost, data = tb(no_loc, adj_mat,
                                           tb_size=20,  # tabu solutions in tb_list
                                           max_tnm=100,  # how many candidates picked in tournament selection
                                           mut_md=mut_md,  # [get_sol, get delta], method of mutation, e.g. swap, 2-opt
                                           term_count=200  # terminate threshold if best_cost not change
                                           )
        case 'ga':
            method_name = 'Genetic Algorithm'
            best_sol, best_cost, data = ga(no_loc, adj_mat,
                                           n_pop=200,
                                           r_cross=0.7,
                                           r_mut=0.9,
                                           selection_md='rw',
                                           # 'rw' : roulette-wheel selection
                                           # 'tnm' : tournament selection
                                           # 'elt' : elitism
                                           max_tnm=3,
                                           term_count=500
                                           )
        case 'sa':
            method_name = 'Simulated Annealing'
            best_sol, best_cost, data = sa(no_loc, adj_mat,
                                           tb_size=0,  # tabu solutions in tb_list
                                           max_tnm=20,  # how many candidates picked in tournament selection
                                           mut_md=mut_md,  # [get_sol, get delta], method of mutation, e.g. swap, 2-opt
                                           term_count_1=25,  # inner loop termination flag
                                           term_count_2=25,  # outer loop termination flag
                                           t_0=1200,  # starting temperature, calculated by init_temp.py
                                           alpha=0.9  # cooling parameter
                                           )
    end = time.process_time()
    result['time'][_] = end - start
    result['cost'][_] = best_cost
    if best_cost < result['best_cost']:
        result['best_sol'] = best_sol
        result['best_cost'] = best_cost
        best_gap = round(best_cost / opt_cost - 1, 5)
        result['best_gap'] = best_gap
    plt.figure(figsize=(16, 10))  # witdth:heigth = 16 : 10
    plt.plot(range(len(data['cost'])), data['cost'], color='b')
    plt.plot(range(len(data['cost'])), data['best_cost'], color='r')
    test_result.append(result)

plt.title('Solving TSP with {}'.format(method_name),
          fontdict={'fontsize': 24,
                    'fontweight': 'bold'}
          )
plt.xlabel('Number of Iteration',
           fontdict={'fontsize': 20,
                     'fontweight': 'light'}
           )
plt.ylabel('Cost(distance), m',
           fontdict={'fontsize': 20,
                     'fontweight': 'light'}
           )
plt.legend(['Distance of A Sample Tour', 'Least Tour Distance'], fontsize=16)
text_loc_x = 0.8*len(data['cost'])
text_loc_y = 0.85*np.max(data['cost'])
plt.text(text_loc_x, text_loc_y, 'best cost = ' + str(best_cost),
         fontdict={'fontsize': 16, 'fontweight': 'bold'},
         bbox=dict(edgecolor='red', alpha=0.8, boxstyle='round'))
plt.text(text_loc_x, text_loc_y*0.95, 'optimal cost = ' + str(opt_cost),
         fontdict={'fontsize': 16, 'fontweight': 'bold'},
         bbox=dict(edgecolor='red', alpha=0.8, boxstyle='round'))
plt.text(text_loc_x, text_loc_y*0.9, 'best gap = ' + str(best_gap),
         fontdict={'fontsize': 16, 'fontweight': 'bold'},
         bbox=dict(edgecolor='red', alpha=0.8, boxstyle='round'))
plt.show()
plt.savefig('results/{}.png'.format(method))

# print results
result['avg_cost'] = np.mean(result['cost'])
result['avg_gap'] = result['avg_cost'] / opt_cost - 1
result['cost_std'] = np.std(result['cost'])
result['avg_time'] = np.mean(result['time'])
result['time_std'] = np.std(result['time'])
pprint(result)

# Visualization of search process
# generate and store the evolution during the search process in the directory /result
if pos:  # pos coordinate is available, plot the tour
    gen = True
else:
    gen = False
gen_search_evolution(num_tests, method_name, data, pos, gen)  # gen=False : no evolution generated
