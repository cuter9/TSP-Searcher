from turfpy import measurement

from ._tw_cities import tw_cities_dict
from geneal.applications.tsp.helpers import create_graph

G = create_graph(tw_cities_dict,
                 measurement.distance,
                 lon=lambda x: x["lng"],
                 lat=lambda x: x["lat"],
                 )
