from geneal.applications.tsp.travelling_salesman_problem import TravellingSalesmanProblemSolver
from geneal.applications.tsp.helpers import plot_cities
from examples.world_cities import cities_of, create_graph_of
from solve import *


no_run = 2      # the no of GA search runs
city_id = 2         # select the cities for tour search
state = ''
match city_id:
    case 1:
        country_name = 'WorldCapitals'
    case 2:
        country_name = 'Taiwan'
        geo_scope = 'asia'
    case 3:
        country_name = 'Germany'
        geo_scope = 'europe'
    case 4:
        country_name = 'Japan'
        geo_scope = 'asia'
    case 5:
        country_name = 'China'
        state = 'Jiangsu'
        geo_scope = 'asia'
    case 6:
        country_name = 'United States'
        state = 'New York'
        geo_scope = 'north america'

world_cities_dict = cities_of(country_name, state)
G = create_graph_of(world_cities_dict)

# selection_strategy : 'roulette_wheel', 'random', 'two_by_two', 'tournament'
# mutation_strategy :   '2-opt', 'random_swap', 'random_inversion', 'random_gene_nearest_neighbour',
#                       'worst_gene_random', 'worst_gene_nearest_neighbour', 'select_any_mutation'
tsp_solver = TravellingSalesmanProblemSolver(
    graph=G,
    pop_size=500,  # population size (number of individuals)
    max_gen=200,  # maximum number of generations
    mutation_rate=0.01,  # mutation rate to apply to the population
    selection_rate=0.7,  # percentage of the population to select for mating
    selection_strategy='roulette_wheel',  # strategy to use for selection.
    mutation_strategy='random_swap'  # strategy to use for mutation. see below for more details.
)

# tsp_solver.solve()  # use this for just one run
tsp_solver.solve_multi_run(no_run)      # use this when run solver more than 1 times

plot_cities(world_cities_dict, tsp_solver,
            lon=lambda x: x['lng'], lat=lambda x: x['lat'],
            name=lambda x: x['country'] + ", " + x['city'],
            scope=geo_scope)
