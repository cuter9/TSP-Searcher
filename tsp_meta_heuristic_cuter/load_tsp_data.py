import numpy as np
from python_tsp.distances import tsplib_distance_matrix
import tsplib95


def distance_matrix_frm_tsplib(problem_id):
    match problem_id:
        case 1:
            tsplib_file = "assets/tsplib_data/swiss42.tsp" # best solution 1273
            optimal_sol = 1273
        case 2:
            tsplib_file = "assets/tsplib_data/a280.tsp"  # best solution 2579
            optimal_sol = 2579
        case 3:
            tsplib_file = "assets/tsplib_data/berlin52.tsp"  # best solution 7542
            optimal_sol = 7542
        case 4:
            tsplib_file = "assets/tsplib_data/ch130.tsp"  # best solution 6110
            optimal_sol = 6110
        case 5:
            tsplib_file = "assets/tsplib_data/brg180.tsp"  # best solution 1950
            optimal_sol = 1950
        case 6:
            tsplib_file = "assets/tsplib_data/ulysses22.tsp"  # best solution 7013
            optimal_sol = 7013


    tsp_problem = tsplib95.load(tsplib_file)
    distance_matrix = tsplib_distance_matrix(tsplib_file)
    pos_dict = tsp_problem.node_coords
    pos_name = list(pos_dict.keys())
    pos_coords = list(pos_dict.values())
    # pos_lst= list(dict.items(pos_dict))
    # pos = np.array(pos_lst)
    opt_cost = optimal_sol
    no_loc = tsp_problem.dimension
    return pos_coords, distance_matrix, opt_cost, no_loc


def distance_matrix_org():
    # load data
    pos = [[float(x) for x in s.split()[1:]] for s in open('data/dj38.txt').readlines()]
    no_loc = len(pos)

    # calculate adjacency matrix
    adj_mat = np.zeros([no_loc, no_loc])
    for i in range(no_loc):
        for j in range(i, no_loc):
            adj_mat[i][j] = adj_mat[j][i] = np.linalg.norm(np.subtract(pos[i], pos[j]))

    opt_cost = 6659.439330623091  # get result from tsp_gurobi.py

    return pos, adj_mat, opt_cost, no_loc
