import os

import pandas as pd


def cities_of(country_name, state=''):
    try:
        module_path = os.path.abspath(os.path.join("."))
        worldcities = pd.read_csv(os.path.join(module_path, "data/worldcities.csv"))
        world_capitals = pd.read_csv(os.path.join(module_path, "data/world_capitals.csv"))
    except FileNotFoundError:
        module_path = os.path.abspath(os.path.join(".."))
        worldcities = pd.read_csv(os.path.join(module_path, "data/worldcities.csv"))
        world_capitals = pd.read_csv(os.path.join(module_path, "data/world_capitals.csv"))

    if country_name != 'WorldCapitals':
        if not state :
            worldcities = worldcities[(worldcities["country"] == country_name)]
        else:
            worldcities = worldcities[(worldcities["country"] == country_name) &
                                      (worldcities["admin_name"] == state)]

        worldcities = worldcities[(worldcities["admin_name"] != "Penghu") &
                                  (worldcities["admin_name"] != "Kinmen") &
                                  (worldcities["admin_name"] != "Lienchiang")]

        worldcities.reset_index(drop=True, inplace=True)
        worldcities.index += 1

        world_cities_dict = worldcities.to_dict(orient="index")
    else:
        world_capitals.dropna(subset=["CapitalName"], axis=0, inplace=True)
        world_capitals.rename(columns={"CapitalLatitude": "lat",
                                       "CapitalLongitude": "lng",
                                       "CountryName": "country",
                                       "CapitalName": "city"}, inplace=True)

        world_capitals.reset_index(drop=True, inplace=True)
        world_capitals.index += 1

        world_cities_dict = world_capitals.to_dict(orient="index")

    return world_cities_dict
