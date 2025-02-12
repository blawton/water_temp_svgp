import timeit
import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gpflow
from gpflow.utilities import print_summary
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold as groupkfold
from build_functions import get_splits, get_folds
#

# Additional requirements for SVGP
import time
import itertools
from typing import Tuple, Optional, Iterable
import pathlib

from gpflow.config import default_float
# from gpflow.ci_utils import ci_niter
from gpflow.utilities import to_default_float

#

#These should stay the same
time_dim=[0]
overall_time_dim=[1]
minute_dim = [5]
dep_var=["Temperature (C)"]
group_vars = ["Station ID"]
years=[2019, 2020, 2021, 2022, 2023]
start_year = 2019
folds=5

#These are for SVGP:
# Number of inducing locations
M=1000
#Batch_size for each epoch
batch_size = 32
# minibatch size for estimating ELBO
# minibatch_size=100
epochs_per_fold=40
logging_epoch_freq=10
# This next condition enforces 1 graph/fold post training
graphing_epoch_freq=10
# maximum iterations
maxiter=10000

#These are dependent on specific version of the model
indep_var=["Day", "Overall_Day", "Latitude", "Longitude", "embay_dist", "minute_of_day"]
area="ES"
method="SVGP"
spatial_pca= False
notes="Like 4.1 but now using dataset without outliers and shuffled folds."
model_name="4.2"
within_year_vars = np.setdiff1d(np.arange(len(indep_var)), np.array(overall_time_dim + minute_dim))
# Characteristic Map
characteristic_names = ['Kernel', 'Optimizer', 'Fold', 'Year', 'Independent Vars', 'Area', 'Method', 'Spatial PCA', 'Model Name', 'Notes', 'Inducing Locations', 'Epochs/Fold', 'Inducing Variables']

#Date
date = "12_26_2024"

#Filename
filename="agg_daily_morning_w_time_coastal_features_v2.csv"

#

# path=r'/content/' + filename
# Changed to below for Pycharm
path = filename
df=pd.read_csv(path, index_col=0)

#Assigning continuous vs. discrete station
df["Continuous"]= False
df.loc[df["Organization"].isin(["EPA_FISM", "STS_Tier_II", "USGS_Cont"]), "Continuous"]=True

#Adding in Year and Day Vars
df["Date"]=pd.to_datetime(df["Date"])
df["Day"]=df["Date"].dt.day_of_year
df["Year"]=df["Date"].dt.year

# Adding in Overall_Day in addition to Day (new in v3.1)
df["Overall_Day"]= 365*(df["Year"]-min(years)) + df["Day"]


#Fixing index, dropping Unnamed: 0, and ensuring all dtypes are float
df.drop("Unnamed: 0", axis=1, inplace=True)
df.reset_index(inplace=True, drop=True)
df[indep_var]=df[indep_var].astype(float)

#

lon_min=-72.59
lon_max=-71.81

print(len(df))
df=df.loc[(df["Longitude"]>lon_min) & (df["Longitude"]<lon_max)].copy()
print(len(df))

#

#Getting kernel
def get_kernels():

  '''Changed so that kernels are independent between combinations, to minimize
  error propagation'''

## Kernels using within year vars

  # matern32_1 = gpflow.kernels.Matern32(lengthscales=[1]*len(within_year_vars),
  #                                    active_dims=within_year_vars)
  # matern32_2 = gpflow.kernels.Matern32(lengthscales=[1]*len(within_year_vars),
  #                                    active_dims=within_year_vars)
  # matern32_3 = gpflow.kernels.Matern32(lengthscales=[1]*len(within_year_vars),
  #                                    active_dims=within_year_vars)
  matern32_4 = gpflow.kernels.Matern32(lengthscales=[1]*len(within_year_vars),
                                     active_dims=within_year_vars)

  # rbf_1= gpflow.kernels.RBF(lengthscales=[1]*len(within_year_vars),
  #                         active_dims=within_year_vars)
  rbf_2= gpflow.kernels.RBF(lengthscales=[1]*len(within_year_vars),
                          active_dims=within_year_vars)

## Kernels using overall day within time period
  rbf_time_1= gpflow.kernels.RBF(lengthscales=[1]*len(overall_time_dim),
                        active_dims=overall_time_dim)
  rbf_time_2= gpflow.kernels.RBF(lengthscales=[1]*len(overall_time_dim),
                        active_dims=overall_time_dim)
  rbf_time_3= gpflow.kernels.RBF(lengthscales=[1]*len(overall_time_dim),
                        active_dims=overall_time_dim)
  rbf_time_4= gpflow.kernels.RBF(lengthscales=[1]*len(overall_time_dim),
                        active_dims=overall_time_dim)

## Kernels using cyclical day of year
  sin_1 = gpflow.kernels.Cosine(lengthscales=[365], active_dims=time_dim)

  ## Kernels using cyclical time of day
  sin_2 = gpflow.kernels.Cosine(lengthscales=[1440], active_dims=minute_dim)

  #Testing set (all have an rbf for time within year), note: sin must be last!
  kernels={
      # "rbf": rbf_1
      # "matern32_rbf_intra": matern32_1 + rbf_time_1
      # "matern32_rbf_intra_sin": matern32_2 + rbf_time_2 + sin_1
      # "rbf_matern32_plus": rbf_1 + matern32_3 + rbf_time_3
      "rbf_rbf_matern32_sin": rbf_2 + matern32_4 + rbf_time_4 + sin_1 + sin_2
      # "rbf_rq": rbf_3 + rq_1,
      # "matern32_rq": matern32_3 +rq_2
  }
  return(kernels)

#Test method
test_output=get_kernels()

#

# Get optimizers (made into a dummy function for now)
def get_opt():
  optimizers={
    "lfbgs":gpflow.optimizers.Scipy(),
  }
  return(optimizers)

#Test method
get_opt()

#

#Standardize Data
def standardize(X, y):
  Xmean=np.mean(X, axis=0)
  Xstd=np.std(X, axis=0)
  ymean=np.mean(y)

#  Hardcoding the cyclical time dimension to have no scaling applied
  if len(X.shape)>1:

    Xmean[time_dim] = 0
    Xstd[time_dim] = 1

    Xmean[minute_dim] = 0
    Xstd[minute_dim] = 1

  X_s=(X-Xmean)/Xstd
  y_s=y-ymean

  return(X_s, y_s, ymean)

#Test method
print(standardize(np.arange(1,10), np.arange(1,10)*10))

#

# Test sample for getting ELBO estimate (temporary)

train_sample=df.sample(1000)
Xtrain_d = train_sample[indep_var].values
ytrain_d = train_sample[dep_var].values

#Building model in each fold
def CV_fold(Xtrain, Xtest, ytrain, ytest,
            train_types, test_types,
            year, fold):

  agg_models={}
  agg_params=pd.DataFrame()
  agg_rmse=pd.DataFrame()
  running_rmse = pd.DataFrame()
  optimizers=get_opt()

  for o_name, optimizer in optimizers.items():

    #Kernels should be reset before usage of each optimizer
    kernels=get_kernels()
    for k_name, kernel in kernels.items():
      #Normalizing (seperately for train and test vars)
      Xtrain_s, ytrain_s, ytrain_mean =standardize(Xtrain, ytrain)
      Xtest_s, ytest_s, ytest_mean =standardize(Xtest, ytest)
      start_time=datetime.datetime.now()

      #Tensor for ELBO calculation
      data = (Xtrain_s, ytrain_s)
      tensor_data = tuple(map(tf.convert_to_tensor, data))

  # From here on out, the SVGP is implemented

      # Shuffling data to get inducing points (later this will be replaced with intentional selection)
      shuffled = Xtrain_s.copy()
      np.random.shuffle(shuffled)

      # Initialize inducing locations to M random inputs in the dataset
      Z = shuffled[:M, :]
      N = tf.constant(len(Xtrain_s))

      # Define Model
      m = gpflow.models.SVGP(kernel=kernel,
                              likelihood = gpflow.likelihoods.Gaussian(),
                              inducing_variable=Z,
                              num_data=N)

      # Setting whether or not parameters are trainable

      ## Cyclical Sin
      if k_name[-3:]=="sin":
        gpflow.utilities.set_trainable(m.kernel.kernels[-1].lengthscales, False)
        gpflow.utilities.set_trainable(m.kernel.kernels[-2].lengthscales, False)
      #   gpflow.utilities.set_trainable(m.kernel.kernels[-1].variance, False)

      ## Inducing variables
      gpflow.utilities.set_trainable(m.inducing_variable, False)

      # print_summary(m)

      # Params needed for initialization of SVGP

      num_train_data = len(Xtrain_s)
      train_dataset = tf.data.Dataset.from_tensor_slices((Xtrain_s, ytrain_s))
      prefetch_size = tf.data.experimental.AUTOTUNE
      shuffle_buffer_size = num_train_data // 2
      num_batches_per_epoch = num_train_data // batch_size

      original_train_dataset = train_dataset
      train_dataset = (
          train_dataset.repeat()
          .prefetch(prefetch_size)
          .shuffle(buffer_size=shuffle_buffer_size)
          .batch(batch_size)
      )
      optimizer = tf.keras.optimizers.Adam()

      @tf.function
      def optimization_step(model: gpflow.models.SVGP, batch: Tuple[tf.Tensor, tf.Tensor]):
          with tf.GradientTape(watch_accessed_variables=False) as tape:
              tape.watch(model.trainable_variables)
              loss = model.training_loss(batch)
          grads = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(grads, model.trainable_variables))
          return loss

      def simple_training_loop(model: gpflow.models.SVGP, epochs: 1,
                         logging_epoch_freq: int = 10,
                         iter_training_data: Iterable[tuple] = []):
        tf_optimization_step = tf.function(optimization_step)

        batches = iter_training_data
        for epoch in range(epochs):
            for _ in range(num_batches_per_epoch):
                tf_optimization_step(model, next(batches))

            epoch_id = epoch + 1
            if epoch_id % logging_epoch_freq == 0:
                tf.print(f"Epoch {epoch_id}: ELBO (train) {model.elbo(tensor_data)}")
                # tf.print(epoch)

      # Training and monitoring loop
      monitor_cycles = max((epochs_per_fold // graphing_epoch_freq), 1)
      running_epoch = 0
      for i in range(monitor_cycles):

        ## Actually running iteration loop
        simple_training_loop(m, epochs=graphing_epoch_freq,
                            logging_epoch_freq=logging_epoch_freq,
                            iter_training_data=iter(train_dataset)
                            )

        running_epoch+=graphing_epoch_freq

        ## Predicting for monitoring
        pred_mean, pred_var = m.predict_y(Xtest_s)

        # ## Graphing
        # fig, ax = plt.subplots()
        # ax.set_title(k_name)
        # ax.scatter(Xtest_s[:, overall_time_dim], pred_mean, c = 'Red', alpha=.15)
        # ax.scatter(Xtest_s[:, overall_time_dim], ytest_s, c='Blue', alpha=.15)
        # plt.savefig(model_name + "_" + "_".join([k_name,
        #                                        o_name,
        #                                        str(fold),
        #                                        str(year),
        #                                        ]) + ".png")

        ## Running rmse
        working = pd.DataFrame(columns=["Code", "RMSE", "Wghted RMSE", "Cont RMSE",
                                        "Discrete RMSE", "y_std", "Epoch"], index=[0])

        working.loc[0, "Code"] = model_name + "_" + "_".join([k_name,
                                               o_name,
                                               str(fold),
                                               str(year),
                                               ])

        working.loc[0, "Epoch"] = running_epoch
        working.loc[0, "RMSE"] = np.sqrt(np.mean((pred_mean - ytest_s)[:, 0] ** 2))

        # For location-weighted RMSE, we assume stratified sample of stations,
        # 1st group is continuous stations, 2nd group is discrete statons
        # Actual population frequency should be 50-50, see derivation in notes on
        # creating an unbiased predictor
        if np.sum(test_types) > 0 and np.sum(np.logical_not(test_types)) > 0:
            # Weighted RMSE
            working.loc[0, "Wghted RMSE"] = np.sqrt(
                .5 * np.nanmean(np.where(test_types, (pred_mean - ytest_s)[:, 0], np.nan) ** 2)
                + .5 * np.nanmean(np.where(test_types == False, (pred_mean - ytest_s)[:, 0], np.nan) ** 2)
            )
        if np.sum(test_types) > 0:
            # For RMSE of continuous stations only
            working.loc[0, "Cont RMSE"] = np.sqrt(
                np.nanmean(np.where(test_types, (pred_mean - ytest_s)[:, 0], np.nan) ** 2)
            )
        if np.sum(np.logical_not(test_types)) > 0:
            # For RMSE of discrete stations only
            working.loc[0, "Discrete RMSE"] = np.sqrt(
                np.nanmean(np.where(test_types == False, (pred_mean - ytest_s)[:, 0], np.nan) ** 2)
            )

        # Std dev of y
        working.loc[0, "y_std"] = np.std(ytest)

        # # Training Time
        # working["Training Time"] = training_time

        running_rmse = pd.concat([running_rmse, working])

      #Storing model for later recreation
      param_dict=gpflow.utilities.parameter_dict(m)
      model_code= model_name + "_" + "_".join([k_name,
                                               o_name,
                                               str(fold),
                                               str(year),
                                               ])
      agg_models[model_code]=param_dict

      #Finishing timing
      end_time=datetime.datetime.now()
      training_time = end_time-start_time

      #Defining characteristics of this loop (order here must match characteristic_names var)
      characteristics = [k_name, o_name, fold, year, "," .join(indep_var), area, method,  spatial_pca,  model_name,  notes,  M,  epochs_per_fold, Z]
      assert(len(characteristics) == len(characteristic_names))

      #Storing Params
      working=pd.DataFrame({k: pd.Series(v.numpy()) for k, v
                                     in [(i, j) for i, j in param_dict.items() if i[:7] == '.kernel']
                            }).transpose()

      #Setting characteristics and saving
      working[characteristic_names] = pd.DataFrame([characteristics], index=working.index)
      agg_params=pd.concat([agg_params, working])

      #Predicting
      pred_mean, pred_var = m.predict_y(Xtest_s)
      #pred_mean=pred_mean+ytrain_mean

      #Storing Predictions
      working=pd.DataFrame(columns=["RMSE", "Wghted RMSE", "Cont RMSE",
                                    "Discrete RMSE", "y_std"], index=[0])

      working.loc[0, "RMSE"]=np.sqrt(np.mean((pred_mean-ytest_s)[:, 0]**2))

      #For location-weighted RMSE, we assume stratified sample of stations,
      #1st group is continuous stations, 2nd group is discrete statons
      #Actual population frequency should be 50-50, see derivation in notes on
      #creating an unbiased predictor
      if np.sum(test_types)>0 and np.sum(np.logical_not(test_types))>0:
        #Weighted RMSE
        working.loc[0, "Wghted RMSE"]= np.sqrt(
            .5*np.nanmean(np.where(test_types, (pred_mean-ytest_s)[:, 0], np.nan)**2)
            + .5*np.nanmean(np.where(test_types==False, (pred_mean-ytest_s)[:, 0], np.nan)**2)
        )
      if np.sum(test_types)>0:
        #For RMSE of continuous stations only
        working.loc[0, "Cont RMSE"]=np.sqrt(
          np.nanmean(np.where(test_types, (pred_mean-ytest_s)[:,0], np.nan)**2)
        )
      if np.sum(np.logical_not(test_types))>0:
        #For RMSE of discrete stations only
        working.loc[0, "Discrete RMSE"]=np.sqrt(
        np.nanmean(np.where(test_types==False, (pred_mean-ytest_s)[:, 0], np.nan)**2)
        )

      #Std dev of y
      working.loc[0, "y_std"]=np.std(ytest)

      #Training Time
      working["Training Time"] = training_time

      #For live monitoring
      print(running_rmse)

      #Setting characteristics and saving
      working[characteristic_names] = pd.DataFrame([characteristics], index=working.index)
      agg_rmse=pd.concat([agg_rmse, working])

  return(agg_models, agg_params, agg_rmse, running_rmse)

#

# # Testing one fold
#
# train_sample=df.sample(500)
# test_sample=df.sample(100)
#
# Xtrain_d = train_sample[indep_var].values
# Xtest_d = test_sample[indep_var].values
# ytrain_d = train_sample[dep_var].values
# ytest_d = test_sample[dep_var].values
#
# # test_types = test_sample["Continuous"].values
# ##Instead setting all continuous types to track error message
# test_types=np.array([True]*100)
# train_types = train_sample["Continuous"].values
#
# now1 = datetime.datetime.now()
#
# for i in range (0, 1):
#     CV_fold(Xtrain_d, Xtest_d, ytrain_d, ytest_d,
#                   train_types, test_types,
#                   test_groups, 2019, 1)[2]
#
# now2 = datetime.datetime.now()
# print(now2-now1)

#

# # Running model in all folds (with small test data to start -statement below picks random samples)
# df = df.sample(50)

## Dummy (will be removed later)
year = 0

## Aggregated storage
agg_models = {}
agg_params = pd.DataFrame()
agg_rmse = pd.DataFrame()
agg_running_rmse = pd.DataFrame()

## Getting the folds for training
fold_data = get_folds(df, year, folds, dep_var, indep_var, group_vars)

## Optional storing of folds for reference
# save_output = pd.DataFrame()
# for i in range(folds):
#     working = fold_data[i]
#     working_Xtrain = working[0]
#     working_ytrain = working[2]
#
#     working_df = pd.DataFrame(working_Xtrain, columns = [indep_var])
#     working_df['ytrain'] = working_ytrain
#     working_df["fold"] = i
#     save_output = pd.concat([save_output, working_df])
#
# print("Test of save output:")
# print(save_output)
# save_output.to_csv("train_folds_through_3.10.csv")

## Running model
assert(folds == len(fold_data))

for i in range(folds):

    models, params, rmse, running_rmse = CV_fold(*fold_data[i])

    agg_models.update(models)
    agg_params = pd.concat([agg_params, params])
    agg_rmse = pd.concat([agg_rmse, rmse])
    agg_running_rmse = pd.concat([agg_running_rmse, running_rmse])

    # Progress
    print("Fold", i, "complete")

# Saving all results

model_path="models_" + model_name + "_" + date +".csv"
param_path="params_" + model_name +  "_" + date + ".csv"
rmse_path="rmse_" + model_name + "_" + date +".csv"
running_rmse_path="running_rmse_" + model_name + "_" + date +".csv"

## Saving model performance and details
agg_models=pd.DataFrame(agg_models)
agg_models.to_csv(model_path)
agg_params.to_csv(param_path)
agg_rmse.to_csv(rmse_path)
agg_running_rmse.to_csv(running_rmse_path)

## Getting inducing variables parameter
assert(sum(list(agg_models.index.str[:18]==".inducing_variable"))==1)
ind_index = list(agg_models.index.str[:18]==".inducing_variable").index(True)
print(ind_index)


## Saving inducing variables specifically
manual_output = agg_models.values[:][ind_index].copy()
index = agg_models.columns.values

for i in range(len(index)):
  working = manual_output[i].numpy()
  # print(manual_output[i])
  output = pd.DataFrame(working, columns = indep_var)

  index_parts = index[i].split("_")[1:-3] + index[i].split("_")[-2:]
  name = "_".join(index_parts)

  output.to_csv(model_name + "_inducing_variables_" + name + ".csv")

## Printing quick summary of results
print(agg_rmse.loc[agg_rmse["Kernel"]=="matern32_rbf_sin_intra"].groupby(["Year", "Fold"])[["RMSE", "Training Time"]].mean())
print(agg_running_rmse)
