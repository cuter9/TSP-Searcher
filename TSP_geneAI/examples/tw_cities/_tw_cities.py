import os

import pandas as pd


try:
    module_path = os.path.abspath(os.path.join("."))
    tw_cities = pd.read_csv(os.path.join(module_path, "data/tw_cities.csv"))
except FileNotFoundError:
    module_path = os.path.abspath(os.path.join(".."))
    tw_cities = pd.read_csv(os.path.join(module_path, "data/tw_cities.csv"))

# s_cities.rename(columns={"LAT": "lat", "LON": "lon"}, inplace=True)

tw_cities = tw_cities[(tw_cities["admin_name"] != "Penghu") &
                      (tw_cities["admin_name"] != "Kinmen") &
                      (tw_cities["admin_name"] != "Lienchiang")]

# tw_cities["lng"] = -tw_cities["lng"]

tw_cities.reset_index(drop=True, inplace=True)
tw_cities.index += 1

tw_cities_dict = tw_cities.to_dict(orient="index")
