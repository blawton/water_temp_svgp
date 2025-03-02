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
# plt.rcParams["figure.figsize"]=(20, 20)

#Imported specifically to redefine max_iter:
# from sklearn.utils.optimize import _check_optimize_result
# from scipy import optimize

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator

# +
# Display params

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

pd.options.display.max_rows=150
pd.options.display.max_columns=150
# -

# Originally, this file was in the space version of Geostatistics Workbook, but now it is a standalone file that can be used for the space only or the space and time data

#Navigating to root of repo
# while(not os.path.basename(os.getcwd()).startswith("lis_gp")):
#     os.chdir("..")
# assert(os.path.basename(os.getcwd()).startswith("lis_gp"))

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

#Coastal Distance for merging with stations
paths[1]="Data_Sources/TIFFs/Masked_Embay_Dist_6_06_2023_Cropped.tif"
# ^Updated to remove dependency on YML for testing

#CSV inputs

# For mergning to temperature data:
paths[2]="Data/Aggregate/ES_means_daily_morning_w_time.csv"
# For merging to eelgrass data:
# paths[2]= "Data/Dominion_Energy/Millstone_Eelgrass_Mapping.csv"

# CSV outputs

# Temperature Data
outputs[1]="Data/Aggregate/agg_daily_morning_w_time_coastal_features.csv"
# Eelgrass Data:
# outputs[1]="Data/Dominion_Energy/Millstone_Eelgrass_Mapping_coastal_features.csv"

for path in paths.values():
  print(path)
  assert(os.path.exists(path))
# -

cont_orgs=["STS_Tier_II", "EPA_FISM", "USGS_Cont"]

# # Preparing and Testing Data (only run when data is updated)

# ## Importing coastal features

#Using the updated file without points removed

gdal.UseExceptions()
embay_dist = gdal.Open(paths[1])
print(embay_dist)

gt=embay_dist.GetGeoTransform()
print("Geotransform", gt)

proj=embay_dist.GetProjection()
print("proj", proj)

band=embay_dist.GetRasterBand(1)
array=band.ReadAsArray()
print("Array Shape", array.shape)

pixelwidth=gt[1]
pixelheight=-gt[5]

xOrigin = gt[0]
yOrigin = gt[3]

cols = embay_dist.RasterXSize
rows = embay_dist.RasterYSize
print(rows, cols)

## Appending coastal distance to data

# Reading in csv
df=pd.read_csv(paths[2])
# print(df.head)

# Ensuring all orgs are covered
print(pd.unique(df["Organization"]))

# Removing stations not in an embayment or extrema in distance from sound
print("Temperature Datapoints:", len(df))
df = df.loc[~df["Station ID"].isin(["1194000",
"1194500",
"1304200",
"1304650",
# ^Not in embayments
"USGS-01127560",
"USGS-01127701"
# ^embay_dist outliers
]), :].copy()
print("Temperature Datapoints in Embayments:", len(df))


#Checking on longitude ranges
print("Longitude Range:", max(df["Longitude"])-min(df["Longitude"]))

df["xPixel"]=((df["Longitude"] - xOrigin) / pixelwidth).astype(int)
df["yPixel"]=(-(df["Latitude"] - yOrigin) / pixelheight).astype(int)

df["embay_dist"]=df.apply(lambda x: array[x["yPixel"], x["xPixel"]], axis=1)

#Getting range of distances in df to ensure it is similar to masked
print("Embay Dist Range:", max(df["embay_dist"])-min(df["embay_dist"]))

# +
# #Getting locations where dist is null or 0
# print(pd.unique(df.loc[df["embay_dist"].isna(), "Station ID"]))

# pd.unique(df.loc[df["embay_dist"]==0, "Station ID"])
# -

# ## Interpolating Stations that are not cropped out
nonzero_ind=np.argwhere(array>0)
# print(nonzero_ind.shape)
# print(nonzero_ind)

nonzero_values = [array[nonzero_ind[i, 0], nonzero_ind[i, 1]] for i in np.arange(nonzero_ind.shape[0])]
# print(nonzero_values[0:10])

interp =  NearestNDInterpolator(nonzero_ind, nonzero_values)

zeroes = df.loc[df["embay_dist"]<=0]
# print(zeroes)
df.loc[df["embay_dist"]<=0, "embay_dist"]=interp(zeroes["yPixel"], zeroes["xPixel"])
# print(df.loc[df["embay_dist"]<=0])

# Setting cropped out stations to 0
array=np.where(array>0, array, 0)

# Cutting out outlier stations
unique_stations = df[["Station ID", "embay_dist"]].copy().groupby("Station ID").max()
outliers = unique_stations.sort_values(by= ["embay_dist"], ascending=False)
print(outliers.head(30))

# Final Output
df.to_csv(outputs[1])

## Various Tests:
#Getting range of distances in df to ensure its similar to masked
print("Post-proccessing Embay Dist Range", max(df["embay_dist"])-min(df["embay_dist"]))

working=df.copy()

#Displaying figure along with all locations
plt.figure()
plt.imshow(array)
plt.scatter(working["xPixel"], working["yPixel"], s=1, c="Red")
plt.show()

#Showing all embayments where there is coverage and all stations missing coverage
coverage_array = np.where(array>0, 1, 0)
plt.figure()
plt.imshow(coverage_array)
working_2=working.loc[working["embay_dist"]<0, ["xPixel", "yPixel", "Station ID"]]
working_2.drop_duplicates(subset=["Station ID"], inplace=True)
plt.scatter(working_2["xPixel"], working_2["yPixel"], s=1, c="Red")
for i in range(len(working_2)):
  plt.text(working_2.iloc[i, 0], working_2.iloc[i,1], working_2.iloc[i, 2])
plt.show()

# +
# #Stations in non-existent embayment (on final model)
# gol=df.loc[df["Station ID"].str.contains("GOL")]
# print(gol)
# df.drop(gol.index, axis=0, inplace=True)
# gol=df.loc[df["Station ID"].str.contains("GOL")]
# print(gol)

# +
# #Fixing East Beach and Barleyfield Cove (These actually should have embay_dist=0)
# df.loc[df["Station ID"]=="East Beach", "embay_dist"] = 0
# df.loc[df["Station ID"]=="Barleyfield Cove", "embay_dist"] = 0
# -

# ## Checking on stations 2019-2021 and outputting array

display_array = np.where(array>0, np.nan, .5)
display_array[0,0]=1

#Reloading Aggregated Temperature Data
working=pd.read_csv(outputs[1])
# working=working.loc[working["Year"].isin(range(2019, 2022))]

# +
#Figure 1a: Non-continuous monitoring stations in LIS
# plt.figure()
#plt.imshow(display_array)

dis=working.loc[~working["Organization"].isin(cont_orgs)]

#Saving data to make Figure 1
dis.to_csv("fig1_discrete.csv")

#plt.scatter(dis["xPixel"], dis["yPixel"], cmap="gray", vmin=0, vmax=1, marker="x", c="red", s=200)
#plt.title("Fig 1a: Sampling Locations", size= "xx-large")
#plt.axis("off")
#plt.savefig("Figures_for_paper/fig1a.png")
#plt.show()

# +
#Figure 1b: only continuous stations in the LIS
# plt.figure()
#plt.imshow(display_array)
cont=working.loc[working["Organization"].isin(cont_orgs)]

#Saving data to make Figure 1
cont.to_csv("fig1_cont.csv")

# plt.scatter(cont["xPixel"], cont["yPixel"], cmap="gray", vmin=0, vmax=1, marker="x", s=200, c="red")
# plt.title("Fig 1b: Continuous Sampling Locations", size= "xx-large")
# plt.axis("off")
# plt.savefig("fig1b.png")
# plt.show()

# Original end of file moved up


