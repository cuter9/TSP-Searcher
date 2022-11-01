# Required Libraries
from pyCombinatorial_rev.algorithm import brute_force_analysis, stochastic_hill_climbing
from pyCombinatorial_rev.algorithm import genetic_algorithm, tabu_search, simulated_annealing_tsp
from pyCombinatorial_rev.utils import graphs, util
from load_tsp_data import *

import time


def tour_searcher():
    # Use a breakpoint in the code line below to debug your script.

    search_times = 3  # times for searching
    problem_id = 3  # 1: swiss; 2:a280(c); 3:berlin(c); 4:ch130(c); 5:brg180; 6: ulysses22(c)
    coordinates, distance_matrix, Optimal_cost, no_loc = distance_matrix_frm_tsplib(problem_id)

    search_method = 2
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
        # evolution_profile[-1].append(search_time)

    # graphs.plot_evolution(evolution_profile[-1][0], best_solution[-1], Optimal_cost, search_method_name)
    graphs.plot_evolution(evolution_profile, best_solution, Optimal_cost, best_search_idx, search_method_name)
    # graphs.plot_tour(coordinates, city_tour=route, view='browser', size=10)
    graphs.plot_tour(coordinates, best_solution[-1], Optimal_cost, search_method_name, view='browser', size=10)

    print('Total Distance: ', round(best_solution[-1][1], 2))
    print('Cost_gap: ', best_solution[-1][2])
    print("執行時間：%f 秒" % evolution_profile[-1][1])


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
    parameters = {'initial_temperature': 1.0,
                  'temperature_iterations': 20,
                  'final_temperature': 0.8,
                  'alpha': 0.9,
                  'verbose': True
                  }

    best_solution, evolution_profile = simulated_annealing_tsp(distance_matrix, **parameters)

    return best_solution, evolution_profile


def tsp_tabu(distance_matrix):
    # Tabu - Parameters
    parameters = {
        'tabu_tenure': 5,  # 10: the number of iterations prohibiting move actions
        'iterations': 10,
        'verbose': True
    }
    city_tour = util.seed_function(distance_matrix)
    best_solution, evolution_profile = tabu_search(distance_matrix, city_tour, **parameters)

    return best_solution, evolution_profile


def tsp_ga(distance_matrix):
    # GA - Parameters
    parameters = {
        'population_size': 10 + 1,  # 10; 20 : the population for breeding, 1: for elite
        'elite': 1,
        'crossover_rate': 0.5,  # 0.5
        'mutation_rate': 0.01,  # 0.1
        'mutation_search': -1,  # 8; -1 : no recursive breading
        'generations': 5,
        'verbose': True
    }

    # GA - Algorithm
    best_solution, evolution_profile = genetic_algorithm(distance_matrix, **parameters)

    return best_solution, evolution_profile


if __name__ == '__main__':
    print('Hi! Let start searching for the best tour')
    tour_searcher()
