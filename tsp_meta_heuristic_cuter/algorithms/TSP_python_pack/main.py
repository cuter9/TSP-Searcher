import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming, solve_tsp_brute_force
from python_tsp.distances import tsplib_distance_matrix
from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing
import time


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    # Get corresponding distance matrix
    # tsplib_file = "tests/tsplib_data/a280.tsp"
    # tsplib_file = "tests/tsplib_data/ulysses22.tsp"
    tsplib_file = "tests/tsplib_data/brazil58.tsp"
    distance_matrix = tsplib_distance_matrix(tsplib_file)
    start = time.process_time()

    # Solve with  using default parameters

    # permutation, distance = solve_tsp_local_search(distance_matrix)
    permutation, distance = solve_tsp_simulated_annealing(distance_matrix)

    # permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    # permutation, distance = solve_tsp_brute_force(distance_matrix)

    end = time.process_time()
    print(permutation)
    print(distance)
    print("執行時間：%f 秒" % (end - start))


if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
