import time
import logging

import numpy as np
import matplotlib.pyplot as plt

from geneal.utils.exceptions import NoFitnessFunction, InvalidInput
from geneal.utils.exceptions_messages import exception_messages
from geneal.utils.helpers import get_elapsed_time
from geneal.utils.logger import configure_logger
from geneal.genetic_algorithms.genetic_algorithm_base import GenAlgSolver


def solve_multi_run(self, no_run):
    self.plot_results = False
    self.mean_fitness_gen = []
    self.max_fitness_gen = []
    self.time_str_gen = []
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    best_fitness = float("-inf")
    best_individual = None
    for i in range(no_run):
        solve_1(self)
        self.mean_fitness_gen.append(self.mean_fitness)
        self.max_fitness_gen.append(self.mean_fitness)
        if best_fitness < self.best_fitness_:
            best_fitness = self.best_fitness_
            best_individual = self.best_individual_
            self.n_best_run = i
        self.time_str_gen.append(self.time_str)
    self.time_avg = np.mean(self.time_str_gen)
    self.best_individual_ = best_individual
    self.best_fitness_ = best_fitness
    for n in range(no_run):
        if n == self.n_best_run:
            line_label_mean = 'Best mean fitness'
            line_label_max = "Best max fitness"
            color_mean = 'blue'
            color_max = 'red'
            line_style = 'solid'
            line_width = 1
        else:
            line_label_mean = 'mean fitness'
            line_label_max = "max fitness"
            color_mean = 'green'
            color_max = 'cyan'
            line_style = 'dotted'
            line_width = 1

        plot_fitness_results_1(self.mean_fitness_gen[n], self.max_fitness_gen[n], self.gen_n,
                               ax, color_mean, color_max, line_style, line_width,
                               line_label_mean, line_label_max)

    text_box = "max running time:" + str(np.max(self.time_str_gen)) + 'sec.' + \
               "\n mean running time: " + str(self.time_avg) + 'sec' + \
               "\n Population size: " + str(self.pop_size) + \
               "\n Number of cities: " + str(self.n_genes) + \
               "\n Selection rate: " + str(self.selection_rate) + \
               "\n Mutation rate: " + str(self.mutation_rate) + \
               "\n Number Generations: " + str(self.generations_) + \
               "\n Best fitness: " + str(-self.best_fitness_) + ' km'
    props = dict(boxstyle='round', facecolor='yellow', alpha=0.9)
    ax.text(0.4, 0.75, text_box, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    plt.title('Convergence Plot of GA search', fontsize=16)
    plt.savefig('results/convergence_plot.pdf', format='pdf')
    plt.show()


def plot_fitness_results_1(mean_fitness, max_fitness, iterations,
                           ax, color_mean, color_max, line_style, line_width,
                           line_label_mean, line_label_max):
    x = np.arange(1, iterations + 1)
    ax.plot(x, -mean_fitness, label=line_label_mean,
            color=color_mean, linestyle=line_style, linewidth=line_width)
    ax.plot(x, -max_fitness, label=line_label_max,
            color=color_max, linestyle=line_style, linewidth=line_width)

    plt.legend(loc='upper right', fontsize=10)
    # plt.show()


def solve_1(self):
    """
    Performs the genetic algorithm optimization according to the parameters
    provided at initialization.

    :return: None
    """

    self.mean_fitness = np.ndarray(shape=(1, 0))
    self.max_fitness = np.ndarray(shape=(1, 0))

    # initialize the population
    population = self.initialize_population()

    fitness = self.calculate_fitness(population)

    fitness, population = self.sort_by_fitness(fitness, population)

    gen_interval = max(round(self.max_gen / 10), 1)

    gen_n = 0
    start_time = time.process_time()
    while True:

        gen_n += 1

        if self.verbose and gen_n % gen_interval == 0:
            logging.info(f"Iteration: {gen_n}")
            logging.info(f"Best fitness: {fitness[0]}")

        self.mean_fitness = np.append(self.mean_fitness, fitness.mean())
        self.max_fitness = np.append(self.max_fitness, fitness[0])

        ma, pa = self.select_parents(fitness)

        ix = np.arange(0, self.pop_size - self.pop_keep - 1, 2)

        xp = np.array(
            list(map(lambda _: self.get_crossover_points(), range(self.n_matings)))
        )

        for i in range(xp.shape[0]):
            # create first offspring
            population[-1 - ix[i], :] = self.create_offspring(
                population[ma[i], :], population[pa[i], :], xp[i], "first"
            )

            # create second offspring
            population[-1 - ix[i] - 1, :] = self.create_offspring(
                population[pa[i], :], population[ma[i], :], xp[i], "second"
            )

        population = self.mutate_population(population, self.n_mutations)

        fitness = np.hstack((fitness[0], self.calculate_fitness(population[1:, :])))

        fitness, population = self.sort_by_fitness(fitness, population)

        if gen_n >= self.max_gen:
            self.gen_n = gen_n
            break

    end_time = time.process_time()
    self.time_str = round(end_time - start_time, 2)
    self.generations_ = gen_n
    self.best_individual_ = population[0, :]
    self.best_fitness_ = fitness[0]
    self.population_ = population
    self.fitness_ = fitness

    if self.plot_results:
        self.plot_fitness_results(self.mean_fitness, self.max_fitness, gen_n)

    if self.show_stats:
        self.print_stats(self.time_str)


# add solve_multi_run method to allow mult running the solve_1
setattr(GenAlgSolver, "solve_multi_run", solve_multi_run)

# modify solve method in the geneal package by adding the attributes of mean_fitness and max_fitness
setattr(GenAlgSolver, "solve_1", solve_1)
