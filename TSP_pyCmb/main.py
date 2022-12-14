# Required Libraries
from pyCombinatorial_rev.algorithm import brute_force_analysis, stochastic_hill_climbing
from pyCombinatorial_rev.algorithm import genetic_algorithm, tabu_search, simulated_annealing_tsp
from pyCombinatorial_rev.algorithm import montecarlo_tree_search
from pyCombinatorial_rev.utils import graphs, util
from load_tsp_data import *
import time


def tour_searcher():
    # Use a breakpoint in the code line below to debug your script.

    search_times = 2  # times for searching
    problem_id = 3  # 1: swiss; 2:a280(c); 3:berlin(c); 4:ch130(c); 5:brg180; 6: ulysses22(c)
    coordinates, distance_matrix, Optimal_cost, no_loc = distance_matrix_frm_tsplib(problem_id)

    search_method = 3
    search_method_name = ''
    best_solution = []
    best_search_idx = 0
    best_search = float('+inf')
    evolution_profile = []
    for n in range(0, search_times):
        start = time.process_time()
        match search_method:
            case 1:
                tsp_bf(distance_matrix)
                search_method_name = 'Brute Force Method'
            case 2:
                best_solution_n, evolution_profile_n = tsp_shc(distance_matrix)
                search_method_name = 'Hill Climbing Method'
            case 3:
                best_solution_n, evolution_profile_n = tsp_tabu(distance_matrix)
                search_method_name = 'Tabu Search Method'
            case 4:
                best_solution_n, evolution_profile_n = tsp_sa(distance_matrix)
                search_method_name = 'Simulating Annealing Method'
            case 5:
                best_solution_n, evolution_profile_n = tsp_ga(distance_matrix)
                search_method_name = 'Genetic Algorithm'
            case 6:
                edges_n, cost_n = tsp_mcts(coordinates, distance_matrix)    # refer to mcts-travel-salesman-master
                best_solution_n = [e[0]+1 for e in edges_n]
                best_solution_n.append(edges_n[-1][1]+1)
                best_solution_n = [best_solution_n, cost_n]
                evolution_profile_n = [cost_n]
                search_method_name = 'Monte Carlo Tree Search'

        end = time.process_time()
        search_time = end - start
        best_solution.append(best_solution_n)
        cost_gap_n = round((best_solution[-1][1] / Optimal_cost - 1) * 100, 2)
        best_solution[-1].append(cost_gap_n)
        if cost_gap_n < best_search:
            best_search = cost_gap_n
            best_search_idx = n

        evolution_profile_n = [evolution_profile_n, search_time]
        evolution_profile.append(evolution_profile_n)

    if search_method != 6:      # MCTS has no convergence plot
        graphs.plot_evolution(evolution_profile, best_solution, Optimal_cost, best_search_idx, search_method_name)
    graphs.plot_tour(coordinates, best_solution[best_search_idx], Optimal_cost, search_method_name, view='browser',
                     size=10)

    print('Best Total Distance: ', round(best_solution[best_search_idx][1], 2))
    print('Best Cost_gap: ', best_solution[best_search_idx][2])
    print("???????????????%f ???" % evolution_profile[best_search_idx][1])


def tsp_bf(distance_matrix):
    best_solution = brute_force_analysis(distance_matrix)
    evolution_profile = []
    return best_solution, evolution_profile


def tsp_shc(distance_matrix):
    # Stochastic Hill Climbing - Parameters
    parameters = {'iterations': 20,
                  'verbose': True
                  }

    city_tour = util.seed_function(distance_matrix)
    best_solution, evolution_profile = stochastic_hill_climbing(distance_matrix, city_tour, **parameters)
    return best_solution, evolution_profile


def tsp_sa(distance_matrix):
    # simulated annealing - Parameters
    parameters = {'initial_temperature': 1.0,       # t_0
                  'temperature_iterations': 20,     # max_tnm
                  'final_temperature': 0.8,         # t_final
                  'alpha': 0.9,                     # alpha
                  'verbose': True
                  }

    best_solution, evolution_profile = simulated_annealing_tsp(distance_matrix, **parameters)
    return best_solution, evolution_profile


def tsp_tabu(distance_matrix):
    # Tabu - Parameters
    parameters = {
        'tabu_tenure': 5,  # tb_tenure; 10: the number of iterations prohibiting move actions
        'iterations': 10,  # iter_no
        'verbose': True
    }
    city_tour = util.seed_function(distance_matrix)
    best_solution, evolution_profile = tabu_search(distance_matrix, city_tour, **parameters)
    return best_solution, evolution_profile


def tsp_ga(distance_matrix):
    # GA - Parameters
    parameters = {
        'population_size': 10,  # n_pop; 10; 20 : the population for breeding, 1: for elite
        'elite': 1,             # elite_no
        'crossover_rate': 0.5,  # r_cross; 0.5
        'mutation_rate': 0.01,  # r_mut; 0.1
        'mutation_search': -1,  # 8; -1 : no recursive breading
        'generations': 5,       # max_gen_no
        'verbose': True
    }

    # GA - Algorithm
    best_solution, evolution_profile = genetic_algorithm(distance_matrix, **parameters)
    return best_solution, evolution_profile


def tsp_mcts(coordinates, distance_matrix):
    # MCTS - Parameters
    parameters = {
        'roll_policy': 'greedy',     # 'greedy' or 'random'
        'prob_greedy': 0.4,  # probability of greedy
        'num_of_expand': 10,  # 50
        'num_of_simulate': 10,  # 20
        'verbose': True
    }

    # MCTS - Algorithm
    best_solution, evolution_profile = montecarlo_tree_search(coordinates, distance_matrix, **parameters)
    return best_solution, evolution_profile


if __name__ == '__main__':
    print('Hi! Let start searching for the best tour')
    tour_searcher()
