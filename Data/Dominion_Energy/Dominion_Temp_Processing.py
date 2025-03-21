import pandas as pd
#import shapefile
from json import dumps
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import datetime

# +
#Loading paths config.yml
import yaml

# This script basically just adds the coordinates into the Dominion temp data
with open("../../config_with_new_names.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    
with open("../../coords.yml", "r") as file:
    yaml_coords = yaml.load(file, Loader=yaml.FullLoader)  

# +
#params

path = config["Dominion_Data_Reading_path"]

assert(os.path.exists(path))

output_file="C_and_NB_data_processed.csv"
# -

dom = pd.read_csv(path)
dom.rename(columns={"Station":"Station ID"}, inplace=True)
dom.drop(0, axis=0, inplace=True)
dom.head()

#lat and lon coords from YAML
dom_coords=pd.DataFrame(index=["C", "NB"], columns=["Latitude", "Longitude"])
dom_coords.loc["C"] = np.array(yaml_coords["Dominion_C"])
dom_coords.loc["NB"] = np.array(yaml_coords["Dominion_NB"])
dom_coords.index.name="Station ID"
dom_coords

#Merging coords
dom=dom.merge(dom_coords.reset_index(), how="left", on="Station ID")
dom.head()

#Outputting
dom.to_csv(output_file)


