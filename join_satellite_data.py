import pandas as pd
import numpy as np
# from itertools import product
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.gaussian_process.kernels import RationalQuadratic
# from sklearn.gaussian_process.kernels import WhiteKernel
# from sklearn.gaussian_process.kernels import Matern
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
import os
#Imported specifically to redefine max_iter:
# from sklearn.utils.optimize import _check_optimize_result
# from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator

# Format of this file borrowed from add_coastal_features.py

# +
#Loading paths config.yml
import yaml

# with open("config_with_new_names.yml", "r") as file:
#     config = yaml.load(file, Loader=yaml.FullLoader)

# +
#Paths for loading data
import os

paths={}
outputs={}

#TIFFs

#satellite numpy data
paths[1]="podaac_satellite_data/satellite_data/numpy_files/all_sat_data.npy"
paths[2]="podaac_satellite_data/satellite_data/harmony_metadata_tagged.csv"
# ^Updated to remove dependency on YML for testing

#CSV inputs

# For merging in temperature data:
paths[3]="Data/Aggregate/agg_daily_morning_w_time_coastal_features.csv"

# CSV outputs

# Temperature Data
outputs[1]="Data/Aggregate/agg_daily_morning_w_time_features_sat.csv"

for path in paths.values():
  print(path)
  assert(os.path.exists(path))
# -

cont_orgs=["STS_Tier_II", "EPA_FISM", "USGS_Cont"]

# # Preparing and Testing Data (only run when data is updated)

# ## Converting degrees Kelvin to celcius
array = np.load(paths[1])
print("Array Shape", array.shape)
array = array - 273.15

# Reading in metadata and converting to date
key = pd.read_csv(paths[2], index_col=0, header=0)
key["date"] = pd.to_datetime(key["date"]).dt.date

# Reading in csv and ensuring unique index
agg_data=pd.read_csv(paths[3], header=0, index_col=0)
agg_data.reset_index(inplace=True)
print("datapoints: ", len(agg_data))

# Ensuring all location parameters are the same across time slices
assert(len(key[['min_lat','max_lat','min_lon','max_lon','lat_step','lon_step']].drop_duplicates())==1)

# Getting key geographic parameters
pixelwidth = key['lon_step'].values[0]
pixelheight = key['lat_step'].values[0]

xOrigin = key['min_lon'].values[0]
yOrigin = key['min_lat'].values[0]

# # Checking alignment of mins
# print("mins:")
# print(xOrigin)
# print(min(agg_data["Longitude"]))
#
# print(yOrigin)
# print(min(agg_data["Latitude"]))
#
# # Checking alignment of maxs
# print("maxs:")
# print(xOrigin+pixelwidth*array.shape[2])
# print(min(agg_data["Longitude"]))
#
# print(yOrigin + pixelheight*array.shape[1])
# print(min(agg_data["Latitude"]))

cols = array.shape[2]
rows = array.shape[1]
print(rows, cols)

## Appending coastal distance to data

# Checking on lat/lon ranges to ensure match
print("Latitude Range of sampling:", max(agg_data["Latitude"])-min(agg_data["Latitude"]))
print("Latitude Range of satellite:", key['max_lat'].values[0] - key['min_lat'].values[0])
print(pixelheight*array.shape[1])

print("Longitude Range of sampling:", max(agg_data["Longitude"])-min(agg_data["Longitude"]))
print("Longitude Range of satellite:", key['max_lon'].values[0] - key['min_lon'].values[0])
print(pixelwidth*array.shape[2])

# print(agg_data["Date"])
# print(key["date"])

# Running interpolation in each slice
for index, row in key.iterrows():

  # Getting data to interpolate
  df = agg_data.loc[pd.to_datetime(agg_data["Date"]).dt.date==row["date"]].copy()
  print(row["date"], " samples: ", len(df))

  # Getting array
  timeslice = array[row["array_index"], :, :].copy()

  if len(df)>0:

    df["xPixel"]=((df["Longitude"] - xOrigin) / pixelwidth).astype(int)
    df["yPixel"]=((df["Latitude"] - yOrigin) / pixelheight).astype(int)

    # Adding in satellite temperature to sampling locations
    df["sat_temp_temp"]=df.apply(lambda x: timeslice[x["yPixel"], x["xPixel"]], axis=1)
    df["temp2"] = np.nan

    # ## Interpolating Stations that are not cropped out (nonzero here means not a nan value)
    nonzero_ind=np.argwhere(~np.isnan(timeslice))

    nonzero_values = [timeslice[nonzero_ind[i, 0], nonzero_ind[i, 1]] for i in np.arange(nonzero_ind.shape[0])]
    # print(nonzero_values[0:10])

    interp =  NearestNDInterpolator(nonzero_ind, nonzero_values)
    #
    zeroes = df.loc[df["sat_temp_temp"].isna()]
    # print(zeroes)

    # Interpolation
    df.loc[df["sat_temp_temp"].isna(), "sat_temp_temp"]=interp(zeroes["yPixel"], zeroes["xPixel"])

    # Adding interpolated data from df into agg_data
    agg_data = agg_data.merge(df[["sat_temp_temp", "temp2"]], left_index=True, right_index=True, how='outer')

    # Consolidating revised data
    agg_data.loc[~agg_data["sat_temp_temp"].isna(), "sat_temp"] = agg_data["sat_temp_temp"]
    agg_data.drop(["temp2", "sat_temp_temp"], inplace=True, axis=1)

    # Progress update
    print("Entries missing: ", len(agg_data.loc[agg_data["sat_temp"].isna()]))

# Dropping days outside date range (uninterpolated)
agg_data.dropna(subset="sat_temp", inplace=True)
print("Missing satellite temperature:", len(agg_data.loc[np.isnan(agg_data["sat_temp"])]))

# Creating satellite temperature data as its own temp dataset for joining in
temp_double = agg_data.copy(deep=True)
temp_double["SST"] = 1
temp_double["Temperature (C)"] = temp_double["sat_temp"]
temp_double.drop("sat_temp", axis=1, inplace=True)

# Preparing agg_data
agg_data["SST"] = 0
agg_data.drop("sat_temp", axis=1, inplace=True)

# Recombining
agg_data = pd.concat([agg_data, temp_double])

# Final Output
agg_data.to_csv(outputs[1])

print("datapoints: ", len(agg_data))