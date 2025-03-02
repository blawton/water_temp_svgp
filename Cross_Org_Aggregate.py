import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from datetime import datetime
from datetime import time
import dateutil.parser

# +
#Loading paths config.yml
import yaml

with open("config_with_new_names.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
# -

#Move to YAML
cont_orgs=["STS_Tier_II", "EPA_FISM", "USGS_Cont"]
es_lon_min = -72.59
es_lon_max=-71.81
es_lat_min = 40.97
es_lat_max = 41.54
years=[2019, 2020, 2021, 2022, 2023]
min_doy=152
max_doy=274

# # ES Daily Summer Morning

# This section uses un-interpolated continuous data to implement gp interpolation of partial time series. It only aggregates data since 2019 to keep code simple and it uses the continuous time series' datapoint nearest to the median of discrete measurements that year

# +
#params

#Desired output names (order matters), these should match the input of the function aggregate_dataset
output_names = ("Date",
"Station ID",
"Latitude",
"Longitude",
"Temperature (C)",
"minute_of_day")

enforce_input = ["Date",
"Station ID",
"Latitude",
"Longitude",
"Temperature (C)"]

#List of organizations to use
organization_names=["EPA_FISM",
                    "STS_Tier_II",
                    "STS_Tier_I",
                    "USGS_Discrete",
                    "URI",
                    "Dominion",
                    "USGS_Cont"]

output_file="Data/Aggregate/ES_means_daily_morning_w_time.csv"

#Input names for above outputs as dictionary of ordered pairs
#Dep. variable(s) at end for flexibility
#EPA_FISM is used to refer to hobo logger data
#This keys in this list also serve as a list of organizations and should match the paths below
input_names={"EPA_FISM":("Date", "Station ID", "Latitude", "Longitude",  "temp"), 
             "STS_Tier_II": ("Date", "Station ID", "Latitude", "Longitude", "temp"),
             "STS_Tier_I": ("Sample Date", "MonitoringLocationIdentifier", "Latitude", "Longitude", "Bottom Temperature (°C)"),
             "USGS_Discrete": ("ActivityStartDate", "MonitoringLocationIdentifier", "LatitudeMeasure", "LongitudeMeasure", "ResultMeasureValue"),
             "URI": ("Date of Sample", "MonitoringLocationIdentifier", "LAT_DD", "LON_DD", "Concentration"),
             "Dominion": ("Date", "Station", "Latitude", "Longitude", "Bot. Temp."),
             "USGS_Cont": ("datetime", "site_no", "Latitude", "Longitude", "Temperature"),   
         }
assert(list(input_names.keys())==organization_names)

#All these paths are in the project repo and thus don't need to be stored on config.yaml
organization_paths={'EPA_FISM':"Data/hobo_data_all_years/HOBO_Data_Agg.csv",
                    'STS_Tier_II':"Data/STS_Continuous_Data/Processed_STS_Continuous_Data.csv",
                    'STS_Tier_I':"Data/STS_Discrete_Data/STS_Discrete.csv",
                    'USGS_Discrete':"Data/USGS_Discrete_Data/USGS_Discrete.csv",
                    'URI': "Data/URIWW/URIWW.csv",
                    'Dominion': "Data/Dominion_Energy/C_and_NB_data_processed.csv",
                    'USGS_Cont': "Data/USGS_Continuous_Data/USGS_Cont_ES_Processed.csv"
                   }
for path in organization_paths.values(): 
    assert(os.path.exists(path))
assert(list(organization_paths.keys())==organization_names)

#Timecodes
timecodes=['Time on Station, 24-Hour Time (hh:mm)', 'ActivityStartTime/Time']
# +
#Loop of reading, renaming, dropping nas
unaggregated={}

for org in organization_names:
    #Reading
    working = pd.read_csv(organization_paths[org])
    
    #Renaming
    working.rename(columns=dict(zip(input_names[org], output_names)), inplace=True)
    
    #Testing to make sure all columns are present
    for col in (set(output_names) - set(["Time"]) - set(["minute_of_day"])):
        assert(list(working.columns).count(col)==1), org +  " columns count of " + col + " != 1"

    # print(working.columns)

    #Renaming time columns (to keep time for below)
    working.rename(columns={code:"Time" for code in timecodes}, inplace=True)
    
    # Restricting to cols of interest
    if list(working.columns).count("Time")>0:
        working=working[enforce_input+["Time"]]
    else:
        working=working[enforce_input]

    #Dropping nas (time nas get dropped later)
    print(org)
    print("Pre NA drop:", len(working))
    working.dropna(inplace=True)
    print("Post NA drop:", len(working))

    #Restricting to years defined in params
    working["Date"]=pd.to_datetime(working["Date"], format="mixed", dayfirst=False)
    working["Year"]=working["Date"].dt.year
    working=working.loc[working["Year"].isin(years)].copy()

    #Restricting to Eastern Sound
    working=working.loc[(working["Longitude"]>=es_lon_min) & (working["Longitude"]<=es_lon_max)]
    working=working.loc[(working["Latitude"]>=es_lat_min) & (working["Latitude"]<=es_lat_max)]
    
    #Saving dataset
    unaggregated[org]=working
    
    #Plotting time of day of continuous samples (should be uniform dist.)
    plt.hist(working["Date"].dt.hour)
    plt.title(org)
    # plt.show()
# -


unaggregated["STS_Tier_I"].head()

# +
#Getting discrete station time quartiles

#Dict for medians in each year
medians={}

#Id-ing discrete data
discrete_orgs=[key for key in unaggregated.keys() if cont_orgs.count(key)==0]
# print(discrete_orgs)
discrete_data=[unaggregated[key] for key in discrete_orgs]

#Timecodes
timecodes=['Time on Station, 24-Hour Time (hh:mm)', 'ActivityStartTime/Time', ]

#Making sure time was properly renamed, formatting it as date, and plotting
for i, df in enumerate(discrete_data):    
    assert(list(df.columns).count("Time")==1), "No Time for " + discrete_orgs[i]
    
    #Converting dominion time
    if discrete_orgs[i]=="Dominion":
        df.dropna(subset="Time", inplace=True)
        df["Hour"]=df["Time"]/100
        df["Time"]=[time(int(time_num//100), round(time_num % 100)) for time_num in df["Time"]]
    else:
        df["Hour"]=pd.to_datetime(df["Time"]).dt.hour


        
    plt.hist(df["Hour"])
    # plt.show()

discrete_data_agg=pd.concat(discrete_data, axis=0)

#Getting quartiles of aggregate Data
for year in years:
    working=discrete_data_agg.loc[discrete_data_agg["Year"]==year]
    medians[year]=working["Hour"].median()

# print(medians)

# +
#Restricting continuous data to nearest datapoint to median of discrete times

for org in cont_orgs:
    agg=pd.DataFrame()
    temp=unaggregated[org]
    
    for station in pd.unique(temp["Station ID"]):
        working=temp.loc[temp["Station ID"]==station].copy()
        print(station, len(working), "(pre-duplicate-drop)")
        working.drop_duplicates("Date", keep="first", inplace=True)
        print(station, len(working), "(post-duplicate-drop)")
        working.set_index("Date", inplace=True)
        
        for date in pd.unique(working.index.date):
            #Getting year
            year=date.year
    
            #Creating day-specific frames
            target_time = datetime.combine(datetime(date.year, date.month, date.day), time(int(medians[year]), 0))
    
            #Locating nearest hour
            idx=working.index.get_indexer([target_time], method="nearest")
            agg=pd.concat([agg, working.reset_index(names="Date").iloc[idx]])
            # print(len(agg))
            
    unaggregated[org]=agg
    
    plt.hist(unaggregated[org]["Date"].dt.hour)
    # plt.show()
# -

#Ensuring numeric dtypes, dropping nas, and adding day variable
processed={}
for org, df in unaggregated.items():
    working=df.copy()
    # print(working.head())
    for col in enforce_input:
        if col != "Date" and col != "Station ID":
            working[col]=pd.to_numeric(working[col], errors="coerce")

    #Adding day col and limiting to days in maximum range of data
    working["Day"]=working["Date"].dt.dayofyear
    working=working.loc[(working["Day"]>=min_doy) & (working["Day"]<=max_doy)].copy()
    # print(working.head())

    # Adding in time if not present
    if "Time" not in list(working.columns):
        working["minute_of_day"] = working["Date"].dt.hour*60 + working["Date"].dt.minute
    else:
        try:
            working["minute_of_day"] = [time_obj.hour*60 + time_obj.minute for time_obj in working["Time"]]
        except:
            working["minute_of_day"] = [dateutil.parser.parse(time_obj).hour*60 + dateutil.parser.parse(time_obj).minute for time_obj in working["Time"]]
    processed[org]=working


processed["URI"].head()

#Aggregating and tagging
aggregated = pd.DataFrame()
for org, df in processed.items():
    df["Organization"]=org
    try:
        print(df[["Time", "minute_of_day"]].sort_values(by=["Time", "minute_of_day"], ascending=False).head(20))
    except:
        print(df[["Date", "minute_of_day"]].sort_values(by= ["Date", "minute_of_day"], ascending=False).head(20))
    aggregated=pd.concat([aggregated, df])

print(len(aggregated))

# Limiting to variables in output_names
aggregated=aggregated[list(output_names) + ["Organization"]].copy()

# Saving
aggregated.reset_index(inplace=True, drop=True)
aggregated.to_csv(output_file)
