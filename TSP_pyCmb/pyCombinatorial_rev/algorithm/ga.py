############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Lesson: pyCombinatorial - Genetic Algorithm

# GitHub Repository: <https://github.com/Valdecy>

############################################################################

# Required Libraries
import copy
import numpy as np
import random
import os


# from pyCombinatorial_rev.utils import graphs, util


############################################################################

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0]) - 1):
        m = k + 1
        distance = distance + distance_matrix[city_tour[0][k] - 1, city_tour[0][m] - 1]
    return distance


# Function: 2_opt
def local_search_2_opt_1(distance_matrix, city_tour, recursive_seeding=-1):
    if recursive_seeding < 0:   # no recursive seeding, do 2-op once
        count = -2
    else:               # recursive seeding and do 2-op for mutation_search times (i.e. recursive_seeding)
        count = 0
    city_list = copy.deepcopy(city_tour)
    distance = city_list[1] * 1
    while count < recursive_seeding:
        best_route = copy.deepcopy(city_list)
        seed = copy.deepcopy(city_list)
        for i in range(1, len(city_list[0]) - 1):
            for j in range(i + 1, len(city_list[0])):
                best_route[0][i:j] = list(reversed(best_route[0][i:j]))
                best_route[0][-1] = best_route[0][0]
                best_route[1] = distance_calc(distance_matrix, best_route)
                if city_list[1] > best_route[1]:
                    city_list = copy.deepcopy(best_route)
                best_route = copy.deepcopy(seed)
        count = count + 1
        if distance > city_list[1] and recursive_seeding < 0:   # repeat 2-op search till no_more best solution
            distance = city_list[1]
            count = -2
            recursive_seeding = -1
        elif city_list[1] >= distance and recursive_seeding < 0:    # stop the 2-op search
            count = -1
            recursive_seeding = -2
    return city_list


# Function: 2_opt
def local_search_2_opt_10(distance_matrix, city_tour, recursive_seeding=-1):
    if recursive_seeding < 0:
        count = -2
    else:
        count = 0
    city_list = copy.deepcopy(city_tour)
    distance = city_list[1] * 2
    while count < recursive_seeding:
        best_route = copy.deepcopy(city_list)
        seed = copy.deepcopy(city_list)
        for i in range(1, len(city_list[0]) - 1):
            for j in range(i + 1, len(city_list[0])):
                best_route[0][i:j] = list(reversed(best_route[0][i:j]))
                best_route[0][-1] = best_route[0][0]
                best_route[1] = distance_calc(distance_matrix, best_route)
                if city_list[1] > best_route[1]:
                    city_list = copy.deepcopy(best_route)
                best_route = copy.deepcopy(seed)
        count = count + 1
        if distance > city_list[1] and recursive_seeding < 0:
            distance = city_list[1]
            count = -2
            recursive_seeding = -1
        elif city_list[1] >= distance and recursive_seeding < 0:
            count = -1
            recursive_seeding = -2
    return city_list


# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding=-1):
    if recursive_seeding < 0:
        count = -2
    else:
        count = 0
    city_list = copy.deepcopy(city_tour)
    distance = city_list[1] * 2
    while count < recursive_seeding:
        best_route = copy.deepcopy(city_list)
        seed = copy.deepcopy(city_list)
        for i in range(0, len(city_list[0]) - 2):
            for j in range(i + 1, len(city_list[0]) - 1):
                best_route[0][i:j + 1] = list(reversed(best_route[0][i:j + 1]))
                best_route[0][-1] = best_route[0][0]
                best_route[1] = distance_calc(distance_matrix, best_route)
                if city_list[1] > best_route[1]:
                    city_list = copy.deepcopy(best_route)
                best_route = copy.deepcopy(seed)
        count = count + 1
        if distance > city_list[1] and recursive_seeding < 0:
            distance = city_list[1]
            count = -2
            recursive_seeding = -1
        elif city_list[1] >= distance and recursive_seeding < 0:
            count = -1
            recursive_seeding = -2
    return city_list


############################################################################

# Function: Initial Seed
def seed_function(distance_matrix):
    seed = [[], float("inf")]
    sequence = random.sample(list(range(2, distance_matrix.shape[0] + 1)), distance_matrix.shape[0] - 1)
    sequence.append(1)
    sequence.insert(0, 1)
    seed[0] = sequence
    seed[1] = distance_calc(distance_matrix, seed)
    return seed


# Function: Initial Seed
def seed_function_0(distance_matrix):
    seed = [[], float("inf")]
    sequence = random.sample(list(range(1, distance_matrix.shape[0] + 1)), distance_matrix.shape[0])
    sequence.append(sequence[0])
    seed[0] = sequence
    seed[1] = distance_calc(distance_matrix, seed)
    return seed


# Function: Initial Population
def initial_population(population_size, distance_matrix):
    population = []
    for i in range(0, population_size):
        seed = seed_function(distance_matrix)
        population.append(seed)
    return population  # population = [tour sequence. tour length]


############################################################################

# Function: Fitness
def fitness_function(cost, population_size):
    fitness = np.zeros((population_size, 2))
    for i in range(0, fitness.shape[0]):
        fitness[i, 0] = 1 / (1 + cost[i] + abs(np.min(cost)))
    fit_sum = fitness[:, 0].sum()
    fitness[0, 1] = fitness[0, 0]
    for i in range(1, fitness.shape[0]):
        fitness[i, 1] = (fitness[i, 0] + fitness[i - 1, 1])
    for i in range(0, fitness.shape[0]):
        fitness[i, 1] = fitness[i, 1] / fit_sum
    return fitness


# Function: Selection
def roulette_wheel(fitness):
    ix = 0
    random = int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if random <= fitness[i, 1]:
            ix = i
            break
    return ix


# Function: TSP Crossover - BCR (Best Cost Route Crossover)
#
def crossover_tsp_bcr_1(distance_matrix, parent_1, parent_2):
    cut_no = random.sample(range(1, len(parent_1[0]) - 1), 1)[0]  # random gen no of cut
    cut_p1 = random.sample(list(range(1, len(parent_1[0]) - 1)), cut_no)  # end and beginning your point keep no change
    cut_ind_p1 = [parent_1[0].index(c) for c in cut_p1]
    child_2 = copy.deepcopy(parent_2)
    for i in range(len(cut_ind_p1)):  # cut points from p1 and insert to the same position in p2
        A = cut_p1[i]
        child_2[0].remove(A)
        child_2[0].insert(cut_ind_p1[i], A)
    d_1 = distance_calc(distance_matrix, [child_2[0], child_2[1]])
    child_2[1] = d_1
    individual = copy.deepcopy(child_2)
    return individual


# Function: TSP Crossover - BCR (Best Cost Route Crossover)
#
def crossover_tsp_bcr_2(distance_matrix, parent_1, parent_2):
    individual = copy.deepcopy(parent_2)
    cut = random.sample(list(range(0, len(parent_1[0]))), int(len(parent_1[0]) / 2))
    cut = [parent_1[0][cut[i]] for i in range(0, len(cut)) if parent_1[0][cut[i]] != parent_2[0][0] and
           parent_1[0][cut[i]] != parent_2[0][-1]]
    d_1 = float('+inf')
    best = []
    for i in range(0, len(cut)):  # check each point in the cut
        A = cut[i]
        parent_2[0].remove(A)
        dist_list = [distance_calc(distance_matrix, [parent_2[0][:n] + [A] + parent_2[0][n:], parent_2[1]]) for n in
                     range(1, len(parent_2[0]))]  # cost list of the movement of A point to all point in the tour
        d_2 = min(dist_list)
        if d_2 <= d_1:
            d_1 = d_2
            n = dist_list.index(d_1)
            best = parent_2[0][:n] + [A] + parent_2[0][n:]  # select the best solution of the A movement
        if best[0] == best[-1]:
            parent_2[0] = best
            parent_2[1] = d_1
            individual = copy.deepcopy(parent_2)
    return individual


def crossover_tsp_bcr(distance_matrix, parent_1, parent_2):
    individual = copy.deepcopy(parent_2)
    cut = random.sample(list(range(0, len(parent_1[0]))), int(len(parent_1) / 2))
    cut = [parent_1[0][cut[i]] for i in range(0, len(cut)) if parent_1[0][cut[i]] != parent_2[0][0]]
    d_1 = float('+inf')
    best = []
    for i in range(0, len(cut)):  # check each point in the cut
        A = cut[i]
        parent_2[0].remove(A)
        dist_list = [distance_calc(distance_matrix, [parent_2[0][:n] + [A] + parent_2[0][n:], parent_2[1]]) for n in
                     range(1, len(parent_2[0]))]  # cost list of the movement of A point to all point in the tour
        d_2 = min(dist_list)
        if d_2 <= d_1:
            d_1 = d_2
            n = dist_list.index(d_1)
            best = parent_2[0][:n] + [A] + parent_2[0][n:]  # select the best solution of the A movement
        if best[0] == best[-1]:
            parent_2[0] = best
            parent_2[1] = d_1
            individual = copy.deepcopy(parent_2)
    return individual


# Function: TSP Crossover - ER (Edge Recombination)
def crossover_tsp_er(distance_matrix, parent_1, parent_2):
    ind_list = [item for item in parent_2[0]]
    ind_list.sort()
    ind_list = list(dict.fromkeys(ind_list))
    edg_list = [[item, []] for item in ind_list]
    for i in range(0, len(edg_list)):
        edges = []
        idx_c = parent_2[0].index(i + 1)
        idx_l = np.clip(idx_c - 1, 0, len(parent_2[0]) - 1)
        idx_r = np.clip(idx_c + 1, 0, len(parent_2[0]) - 1)
        if parent_2[0][idx_l] not in edges:
            edges.append(parent_2[0][idx_l])
        if parent_2[0][idx_r] not in edges:
            edges.append(parent_2[0][idx_r])
        idx_c = parent_1[0].index(i + 1)
        idx_l = np.clip(idx_c - 1, 0, len(parent_1[0]) - 1)
        idx_r = np.clip(idx_c + 1, 0, len(parent_1[0]) - 1)
        if parent_1[0][idx_l] not in edges:
            edges.append(parent_1[0][idx_l])
        if parent_1[0][idx_r] not in edges:
            edges.append(parent_1[0][idx_r])
        for edge in edges:
            edg_list[i][1].append(edge)
    start = parent_1[0][0]
    individual = [[start], 1]
    target = start
    del edg_list[start - 1]
    while len(individual[0]) != len(parent_2[0]) - 1:
        limit = float('+inf')
        candidates = [[[], []] for item in edg_list]
        for i in range(0, len(edg_list)):
            if target in edg_list[i][1]:
                candidates[i][0].append(edg_list[i][0])
                candidates[i][1].append(len(edg_list[i][1]))
                if len(edg_list[i][1]) < limit:
                    limit = len(edg_list[i][1])
                edg_list[i][1].remove(target)
        for i in range(len(candidates) - 1, -1, -1):
            if len(candidates[i][0]) == 0 or candidates[i][1] != [limit]:
                del candidates[i]
        if len(candidates) > 1:
            k = random.sample(list(range(0, len(candidates))), 1)[0]
        else:
            k = 0
        if len(candidates) > 0:
            target = candidates[k][0][0]
            individual[0].append(target)
        else:
            if len(edg_list) > 0:
                target = edg_list[0][0]
                individual[0].append(target)
            else:
                last_edges = [item for item in ind_list if item not in individual[0]]
                for edge in last_edges:
                    individual[0].append(edge)
        for i in range(len(edg_list) - 1, -1, -1):
            if edg_list[i][0] == target:
                del edg_list[i]
                break
        if len(edg_list) == 1:
            individual[0].append(edg_list[0][0])
    individual[0].append(individual[0][0])
    individual[1] = distance_calc(distance_matrix, individual)
    return individual


# Function: Breeding
def breeding_1(distance_matrix, population, fitness, crossover_rate, elite):
    cost = [item[1] for item in population]
    if elite > 0:
        cost, offspring = (list(t) for t in zip(*sorted(zip(cost, population))))
    for i in range(elite, len(offspring), 2):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        parent_1 = copy.deepcopy(population[parent_1])
        parent_2 = copy.deepcopy(population[parent_2])
        rand = int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)
        if rand > crossover_rate:
            rand_1 = int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)
            # if rand_1 > 0.5:
            offspring[i] = crossover_tsp_bcr_1(distance_matrix, parent_1, parent_2)
            offspring[i + 1] = crossover_tsp_bcr_1(distance_matrix, parent_2, parent_1)
        else:
            offspring[i] = parent_1
            offspring[i + 1] = parent_2
        return offspring


# Function: Breeding
def breeding(distance_matrix, population, fitness, crossover_rate, elite):
    cost = [item[1] for item in population]
    if elite > 0:
        cost, offspring = (list(t) for t in zip(*sorted(zip(cost, population))))
    for i in range(elite, len(offspring)):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        parent_1 = copy.deepcopy(population[parent_1])
        parent_2 = copy.deepcopy(population[parent_2])
        rand = int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)
        if rand > crossover_rate:
            rand_1 = int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)
            if rand_1 > 0.5:
                offspring[i] = crossover_tsp_bcr(distance_matrix, parent_1, parent_2)
            else:
                offspring[i] = crossover_tsp_bcr(distance_matrix, parent_2, parent_1)
        else:
            offspring[i] = crossover_tsp_er(distance_matrix, parent_1, parent_2)
    return offspring


# Function: Mutation - Swap with 2-opt Local Search
def mutation_tsp_swap_1(distance_matrix, individual, mutation_search):
    # k = random.sample(list(range(1, len(individual[0]) - 1)), 2)
    if mutation_search < 0:     # random choose the mutation list that must be times 2
        mut_lst = random.sample(list(range(1, len(individual[0]) - 1)), 2)
    else:
        mut_lst = random.sample(list(range(1, len(individual[0]) - 1)), int(mutation_search * 2))
    for k in range(0, len(mut_lst), 2):     # swap by pair in the mut_list
        k1 = mut_lst[k]
        k2 = mut_lst[k + 1]
        A = individual[0][k1]
        B = individual[0][k2]
        individual[0][k1] = B
        individual[0][k2] = A
    individual[1] = distance_calc(distance_matrix, individual)
    individual = local_search_2_opt_1(distance_matrix, individual, mutation_search)
    return individual


# Function: Mutation - Swap with 2-opt Local Search
def mutation_tsp_swap(distance_matrix, individual, mutation_search):
    k = random.sample(list(range(1, len(individual[0]) - 1)), 2)
    k1 = k[0]
    k2 = k[1]
    A = individual[0][k1]
    B = individual[0][k2]
    individual[0][k1] = B
    individual[0][k2] = A
    individual[1] = distance_calc(distance_matrix, individual)
    individual = local_search_2_opt(distance_matrix, individual, mutation_search)
    return individual


# Function: Mutation
def mutation_1(distance_matrix, offspring, mutation_rate, mutation_search, elite):
    for i in range(elite, len(offspring)):
        probability = int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)
        if probability <= mutation_rate:
            offspring[i] = mutation_tsp_swap_1(distance_matrix, offspring[i], mutation_search)
    return offspring  # Function: Mutation


# Function: Mutation
def mutation(distance_matrix, offspring, mutation_rate, mutation_search, elite):
    for i in range(elite, len(offspring)):
        probability = int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)
        if probability <= mutation_rate:
            offspring[i] = mutation_tsp_swap(distance_matrix, offspring[i], mutation_search)
    return offspring


############################################################################

# Function: GA TSP
def genetic_algorithm(distance_matrix, population_size=5, elite=1, crossover_rate=0.5, mutation_rate=0.1,
                      mutation_search=-1, generations=100, verbose=True):
    population = initial_population(population_size, distance_matrix)
    cost = [item[1] for item in population]
    cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
    elite_ind = population[0]
    fitness = fitness_function(cost, population_size)
    count = 0
    count_lst = [count]
    cost_lst = [np.min(cost)]
    while count <= generations:
        if verbose == True:
            print('Generation = ', count, 'Distance = ', round(elite_ind[1], 2))
        offspring = breeding(distance_matrix, population, fitness, crossover_rate, elite)
        offspring = mutation(distance_matrix, offspring, mutation_rate, mutation_search, elite)
        cost = [item[1] for item in offspring]
        cost, population = (list(t) for t in zip(*sorted(zip(cost, offspring))))
        elite_child = population[0]
        fitness = fitness_function(cost, population_size)
        if elite_ind[1] > elite_child[1]:
            elite_ind = elite_child
        count = count + 1
        count_lst.append(count)
        cost_lst.append(round(np.min(cost), 2))
    # route, distance = elite_ind
    best_solution = elite_ind
    evolution_profile = [count_lst, cost_lst]

    # return route, distance, evolution_profile
    # best_solution = [route, distance]
    return best_solution, evolution_profile

############################################################################
