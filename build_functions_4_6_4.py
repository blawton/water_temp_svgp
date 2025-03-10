import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold as groupkfold

test_input = 'Data/Aggregate/agg_daily_morning_coastal_features.csv'

#Standardize Data
def standardize(X, y, exclude_dims):
  Xmean=np.mean(X, axis=0)
  Xstd=np.std(X, axis=0)
  ymean=np.mean(y)
  ystd=np.std(y)

#  Hardcoding the cyclical time dimension to have no scaling applied
  if (len(X.shape)>1 and exclude_dims is not None):
    for dim in exclude_dims:
        Xmean[dim] = 0
        Xstd[dim] = 1

  X_s=(X-Xmean)/Xstd
  y_s=(y-ymean)/ystd

  return(X_s, Xstd, y_s, ymean, ystd)

# Defining inducing locations helper function
def get_inducing_variable(M, dep_var, indep_var, loc_vars, data):
    # Testing new method for inducing variables
    all_data=data.copy()

    # Getting unique locations
    locs = all_data.drop_duplicates(subset=loc_vars, inplace=False)[loc_vars]
    num_locs = len(locs)
    leftover_locs = M % num_locs

    # Getting non-location variables
    probs = (all_data[dep_var].values - all_data[dep_var].mean().values)/all_data[dep_var].var()[0]
    probs = np.array([max(prob[0], .5) for prob in probs])
    # Putting a floor on probability for data close to mean as having .5 standard deviations from the mean so it still gets sampled
    norm_probs = probs/np.sum(probs)
    # print(norm_probs)

    time_vars=[col for col in indep_var if col not in loc_vars]
    times=pd.DataFrame(all_data[time_vars].sample(M, weights = norm_probs), columns=time_vars)
    times.reset_index(inplace=True, drop=True)
    # print(times.head())

    repeated_locs = np.repeat(locs,(M // len(locs)), axis=0)
    extra_locs = locs.sample(leftover_locs, replace=False)
    # print((repeated_locs.shape), extra_locs.shape)

    all_locs = pd.DataFrame(np.concatenate([repeated_locs, extra_locs], axis=0), columns=loc_vars)
    all_locs.reset_index(inplace=True, drop=True)
    # print(all_locs.head())

    assert all_locs.shape[0] == M
    assert times.shape[0] == M

    inducing_locations = all_locs
    inducing_locations[time_vars]=times
    # print(inducing_locations.head())

    return (inducing_locations)

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
    # print(df.iloc[train_index, :].groupby("Station ID").count().iloc[:, 0])
    # print("Test")
    # print(df.iloc[test_index, :].groupby("Station ID").count().iloc[:, 0])

  return(train_folds, test_folds)

# Helper function to get all necessary data for each fold from original dataframe
def get_folds(df, year, folds, dep_var, indep_var, group_vars):

    df_year=df.copy()

    X=np.asarray(df_year[indep_var])
    y=np.asarray(df_year[dep_var])

    groups=np.array(df_year[group_vars].apply(lambda x: "_".join(x.astype(str)), axis=1))

    # groups=df_year["Station ID"]
    # print(groups)

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
group_vars = ["Station ID", "Year"]
folds = 5

df[indep_var]=df[indep_var].astype(float)
df.dropna(subset=dep_var+indep_var, inplace=True)

#1 Standardizing data
print(standardize(np.arange(1,10), np.arange(1,10)*10, exclude_dims=None))

#2 Testing inducing variable
print(get_inducing_variable(M=1000, dep_var=dep_var, indep_var=indep_var, loc_vars=["Latitude", "Longitude", "embay_dist"], data=df))

#3 Testing splits
print(get_splits(df[indep_var].sample(20),
                 df[dep_var].sample(20),
                 df[group_vars].astype(str).sample(20),
                 folds))

#4 Testing get_folds with data from all years, with year=0
get_folds(df, 0, folds, dep_var, indep_var, group_vars)


