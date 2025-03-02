import pandas as pd
import numpy as np

file=pd.read_csv("../USGS_Discrete_Data/wqp_data_pull_2024_11_21.csv", header=0)
print(file.columns)
print(len(file))
print(pd.unique(file.loc[file["CharacteristicName"].str.match(".*temp.*", case=False), "CharacteristicName"]))
print(file.loc[file["CharacteristicName"]=='Temperature, water', ["ResultMeasureValue", "OrganizationIdentifier", "ActivityIdentifier"]])
print(pd.unique(file.loc[file['CharacteristicName']=='Temperature, water', "OrganizationIdentifier"]))