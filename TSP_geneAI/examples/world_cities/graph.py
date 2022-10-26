from turfpy import measurement

# from ._world_cities import tw_cities_dict
from geneal.applications.tsp.helpers import create_graph


def create_graph_of(world_cities_dict):
    G = create_graph(world_cities_dict,
                     measurement.distance,
                     lon=lambda x: x["lng"],
                     lat=lambda x: x["lat"],
                     )
    return G
