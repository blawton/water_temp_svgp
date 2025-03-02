import pandas as pd
from json import dumps
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# All results in the latest version of this file are taken from the following two queries of the WQP.

# Query for Data
# https://www.waterqualitydata.us/#bBox=-72.592354875000%2C40.970592192000%2C-71.811481513000%2C41.545060950000&siteType=Estuary&startDateLo=01-01-2019&startDateHi=11-02-2024&mimeType=csv&dataProfile=resultPhysChem&providers=NWIS&providers=STORET

# Query for Stations:
# https://www.waterqualitydata.us/#bBox=-72.592354875000%2C40.970592192000%2C-71.811481513000%2C41.545060950000&siteType=Estuary&startDateLo=01-01-2019&startDateHi=11-02-2024&mimeType=csv&providers=NWIS&providers=STORET

pd.options.display.max_rows=150
pd.options.display.max_columns=150

# +
#Loading paths config.yml
import yaml

with open("../../config_with_new_names.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# +
#params

#Naming convention for WQP Data
filename = "resultphyschem"

# #Range of Years
# years = np.arange(2010, 2023)
# ^The above is no longer neccesary given that all years are downloaded in one aggregate file

#Data Source for raw data
path1 = config["WQP_Data_Reading_path1"]

#Data Source for stations
path2 = config["WQP_Data_Reading_path2"]

assert(os.path.exists(path1))
assert(os.path.exists(path2))

#Dependent Variable
depvar="Temperature, water"
depvar_unit="deg C"

output_file="USGS_Discrete.csv"
# -

#Getting filepaths for each year
paths = [path1]

wqpdf=pd.DataFrame()
for path in paths:
    # file= path.rsplit("/")[-1]
    working=pd.read_csv(path)
    wqpdf=pd.concat([wqpdf, working])


# # Merging Non-STS Data

#Reading in stations
wqpstations=pd.read_csv(path2)

#Extracting CTDEEP Stations for use elsewhere
wqpstations.loc[wqpstations["OrganizationIdentifier"]=="CT_DEP01_WQX"].to_csv("../CTDEEP/CTDEEP_Stations.csv")

#Merging Station Data and Underlying Data
wqpdf=wqpdf.merge(wqpstations, how="left", on="MonitoringLocationIdentifier", suffixes=["", "_s"])
wqpdf.head()

#Ensuring completeness of the merge (should be empty df)
print("Unmatched Stations:", len(wqpdf.loc[wqpdf["MonitoringLocationIdentifier"].isna()]))

# # Restricting to Embayment Data

#Reindexing based on newly available parameters
# embaydf=wqpdf.loc[wqpdf["Embay_Pt"]==1].copy(deep=True)
# embaydf.head()
# ^This is ignored in this version of the code because the Embay_Pt field no longer exists,
# replaced with the simple copying below
embaydf=wqpdf.copy()

#Outputting non-STS Data
print(len(embaydf))
embaydf=embaydf.loc[embaydf["OrganizationIdentifier"].isin(['USGS-NY', 'USGS-CT'])].copy()
print(len(embaydf))

print(pd.unique(embaydf["OrganizationIdentifier"]))

#Ensuring all temp is in C (output should be empty)
print("Non metric units:", len(embaydf.loc[(embaydf["CharacteristicName"]==depvar) & (embaydf["ResultMeasure/MeasureUnitCode"]!=depvar_unit)]))

#Restricting to depvar
print(len(embaydf))
embaydf=embaydf.loc[(embaydf["CharacteristicName"]==depvar) & (embaydf["ResultMeasure/MeasureUnitCode"]==depvar_unit)]
print(len(embaydf))

embaydf.to_csv(output_file)
