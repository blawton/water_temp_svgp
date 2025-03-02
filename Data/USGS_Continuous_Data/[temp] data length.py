import pandas as pd
file=pd.read_csv("USGS_Cont_ES_Pre_Processing_11_17_2024.csv", header=0)
print(file.columns)
print(len(file))
file["year"]=pd.to_datetime(file["datetime"]).dt.year
print(file.groupby(["year", "station_nm"]).count())