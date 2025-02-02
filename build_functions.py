import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold as groupkfold

test_input = 'agg_daily_morning_coastal_features.csv'

#Helper function for randomly choosing folds
def get_splits(X, y, groups, folds):

  train_folds={}
  test_folds={}

  fold_object=groupkfold(folds, shuffle=True)
  fold_object.split(X=X, y=y, groups=groups)

  for i, (train_index, test_index) in enumerate(fold_object.split(X, y, groups)):
    train_folds[i]=train_index
    test_folds[i]=test_index
    # print("Fold", i, "Composition")
    # print("Train")
    # print(df.iloc[train_index, :].drop_duplicates(["Station ID"]).groupby("Organization").count().iloc[:, 0])
    # print("Test")
    # print(df.iloc[test_index, :].drop_duplicates(["Station ID"]).groupby("Organization").count().iloc[:, 0])

  return(train_folds, test_folds)

# Helper function to get all necessary data for each fold from original dataframe
def get_folds(df, year, folds, dep_var, indep_var, group_vars):

    df_year=df.copy()

    X=np.asarray(df_year[indep_var])
    y=np.asarray(df_year[dep_var])
    groups=np.asarray(df_year[group_vars])
    data_types=np.asarray(df_year["Continuous"])

    train_folds, test_folds = get_splits(X, y, groups, folds)
    # print(test_folds)
    fold_data = []
    for fold in np.arange(folds):

        #Isolating fold
        Xtrain=X[train_folds[fold], :]
        Xtest=X[test_folds[fold], :]

        ytrain=y[train_folds[fold]]
        ytest=y[test_folds[fold]]

        train_types=data_types[train_folds[fold]]
        test_types=data_types[test_folds[fold]]

        test_groups=groups[test_folds[fold]]

        fold_data.append((Xtrain, Xtest, ytrain, ytest,
            train_types, test_types,
            year, fold))

    return(fold_data)

#

#Testing method (with 50 random samples of data)

##Creating test df
df = pd.read_csv(test_input)

#Assigning continuous vs. discrete station
df["Continuous"]= False
df.loc[df["Organization"].isin(["EPA_FISM", "STS_Tier_II", "USGS_Cont"]), "Continuous"]=True

##Adding in Year Var
df["Date"]=pd.to_datetime(df["Date"])
df["Year"]=df["Date"].dt.year
years=[2019, 2020, 2021, 2022, 2023]

##Adding in Overall_Day in addition to Day (new in v3.1)
df["Overall_Day"]= 365*(df["Year"]-min(years)) + df["Day"]

##Fixing index and dropping Unnamed: 0
df.drop("Unnamed: 0", axis=1, inplace=True)
df.reset_index(inplace=True, drop=True)

##Final step
lon_min=-72.59
lon_max=-71.81
df=df.loc[(df["Longitude"]>lon_min) & (df["Longitude"]<lon_max)].copy()

indep_var=["Day", "Overall_Day", "Latitude", "Longitude", "embay_dist"]
dep_var=["Temperature (C)"]
group_vars = ["Station ID"]
folds = 5

#Testing splits
print(get_splits(df[indep_var].sample(20),
                 df[dep_var].sample(20),
                 df[group_vars].sample(20),
                 folds))

#Testing get_folds with data from all years, with year=0
get_folds(df, 0, folds, dep_var, indep_var, group_vars)
