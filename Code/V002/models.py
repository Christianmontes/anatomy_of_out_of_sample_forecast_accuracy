import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression
import autokeras as ak
import tensorflow as tf
import torch
from tensorflow.keras import layers
#from autosklearn.regression import AutoSklearnRegressor

import joblib
import os
import inspect


def xgboost_plain(df_train, df_test, n_fold, n_threads, search=False):

    if search:
        function_name = inspect.getframeinfo(inspect.currentframe()).function
        cv_cache_file = "./_cv_cache/%s_%s_%i.bin" % (function_name,
                                                      joblib.hash(df_train.to_numpy().astype(np.float)), n_fold)
        if os.path.exists(cv_cache_file):
            gs_model = pickle.load(open(cv_cache_file, "rb"))
        else:
            model = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=1)
            grid = {
                "n_estimators": [100, 1000, 10000],
                "max_depth": range(2, 10),
                "learning_rate": np.linspace(0.001, 0.3, 10)
            }
            gs_model = GridSearchCV(estimator=model, param_grid=grid, cv=n_fold, verbose=2, n_jobs=n_threads)
            gs_model.fit(df_train.drop("Y", axis=1), df_train["Y"])
            os.makedirs("_cv_cache", exist_ok=True)
            pickle.dump(gs_model, open(cv_cache_file, "wb"))
        print(gs_model.best_params_)
    else:
        gs_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=10000, n_jobs=n_threads,
                                    max_depth=1, subsample=1, learning_rate=0.001)
        gs_model.fit(df_train.drop("Y", axis=1), df_train["Y"])

    y_hat = gs_model.predict(df_test.drop("Y", axis=1))
    return y_hat

    print(pd.concat((df_test["Y"], pd.Series(y_hat, index=df_test.index)), axis=1).corr())
    print(np.mean(np.sqrt((df_test["Y"] - y_hat) ** 2)))
    print(np.mean(np.sqrt((df_test["Y"] - df_train["Y"].mean()) ** 2)))


def xgboost_rolling(df_train, df_test, weight_recent_obs=False):

    out = []
    for i in tqdm(range(df_test.shape[0])):

        df_train_i = pd.concat((df_train, df_test.iloc[:i]))
        df_test_i = df_test.iloc[[i]]

        tau = 5000
        sample_weights = np.exp(-(df_train_i.index.max() - df_train_i.index).days.to_numpy() / tau)

        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=10000, n_jobs=1)
        grid = {
            'max_depth': range(1, 8),
            'learning_rate': np.linspace(0.001, 0.3, 5)
        }
        gs_model_i = GridSearchCV(estimator=model, param_grid=grid, cv=10, verbose=0, n_jobs=64-1)
        extra_args = {"sample_weight": sample_weights} if weight_recent_obs else {}
        gs_model_i.fit(df_train_i.drop("Y", axis=1), df_train_i["Y"], **extra_args)
        pred_i = gs_model_i.predict(df_test_i.drop("Y", axis=1))

        weighted_mean_i = (df_train_i["Y"] * (sample_weights / sample_weights.sum())).sum()

        out.append((pred_i, gs_model_i, df_train_i["Y"].mean(), weighted_mean_i, df_test_i["Y"]))

    pickle.dump(out, open("xgboost_rolling_out_.bin", "wb"))
    print(np.mean(np.sqrt((df_test["Y"] - np.array([x[0][0] for x in out])) ** 2)))
    print(np.corrcoef((df_test["Y"].to_numpy(), [x[0][0] for x in out])))
    exit(-1)


def random_forest_plain(df_train, df_test, n_fold, n_threads, search=False):

    if search:
        function_name = inspect.getframeinfo(inspect.currentframe()).function
        cv_cache_file = "./_cv_cache/%s_%s_%i.bin" % (function_name,
                                                      joblib.hash(df_train.to_numpy().astype(np.float)), n_fold)
        if os.path.exists(cv_cache_file):
            gs_model = pickle.load(open(cv_cache_file, "rb"))
        else:
            model = RandomForestRegressor(n_jobs=1)
            grid = {
                "n_estimators": [100, 1000],
                "max_depth": range(1, 10),
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
            gs_model = GridSearchCV(estimator=model, param_grid=grid, cv=n_fold, verbose=2, n_jobs=n_threads)
            gs_model.fit(df_train.drop("Y", axis=1), df_train["Y"])
            os.makedirs("_cv_cache", exist_ok=True)
            pickle.dump(gs_model, open(cv_cache_file, "wb"))
        print(gs_model.best_params_)
    else:
        gs_model = RandomForestRegressor(n_estimators=1000, max_depth=3, n_jobs=n_threads)
        gs_model.fit(df_train.drop("Y", axis=1), df_train["Y"])

    y_hat = gs_model.predict(df_test.drop("Y", axis=1))
    return y_hat

    print(pd.concat((df_test["Y"], pd.Series(y_hat, index=df_test.index)), axis=1).corr())
    print(np.mean(np.sqrt((df_test["Y"] - y_hat) ** 2)))
    print(np.mean(np.sqrt((df_test["Y"] - df_train["Y"].mean()) ** 2)))


def linear_models(df_train, df_test):

    models = [("ridge", RidgeCV(alphas=np.linspace(0.1, 10, 100), fit_intercept=True)),
              ("lasso", LassoCV(n_alphas=100, fit_intercept=True)),
              ("elnet", ElasticNetCV(n_alphas=100, fit_intercept=True))]

    for key, model in models:
        model.fit(df_train.drop("y", axis=1), df_train["y"])
        yield pd.Series(model.predict(df_test.drop("y", axis=1)), index=df_test.index).rename("%s_y_hat" % key)


def nn_auto(df_train, df_test):

    """def generate_data(num_instances=100, shape=(32, 32, 3), dtype="np"):
        np.random.seed(1)
        data = np.random.rand(*((num_instances,) + shape))
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        if dtype == "np":
            return data
        if dtype == "dataset":
            return tf.data.Dataset.from_tensor_slices(data)

    def generate_data_with_categorical(
            num_instances=100, num_numerical=10, num_categorical=3, num_classes=5, dtype="np"
    ):
        categorical_data = np.random.randint(
            num_classes, size=(num_instances, num_categorical)
        )
        numerical_data = np.random.rand(num_instances, num_numerical)
        data = np.concatenate((numerical_data, categorical_data), axis=1)
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        if dtype == "np":
            return data
        if dtype == "dataset":
            return tf.data.Dataset.from_tensor_slices(data)

    lookback = 2
    predict_from = 1
    predict_until = 10
    train_x = generate_data_with_categorical(num_instances=100)
    train_y = generate_data(num_instances=80, shape=(1,))
    clf = ak.TimeseriesForecaster(
        lookback=lookbacsk,
        predict_from=predict_from,
        predict_until=predict_until,
        max_trials=2,
        seed=1,
    )
    clf.fit(train_x, train_y, epochs=1, validation_data=(train_x, train_y))
    keras_model = clf.export_model()
    clf.evaluate(train_x, train_y)"""

    #model_aml = ak.TimeseriesForecaster(max_trials=5000, lookback=5,
    #                                    column_types=dict([(x, "numerical") for x in df_train.columns if x != "Y"]))

    model_aml = ak.StructuredDataRegressor(overwrite=True, max_trials=5000, # tuner="random",
                                           column_types=dict([(x, "numerical") for x in df_train.columns if x != "Y"]))

    model_aml.fit(df_train.drop("Y", axis=1), df_train["Y"], epochs=5000,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=20)])
    model = model_aml.export_model()

    tf.keras.models.save_model(model, "_nn_auto", overwrite=True)

    model.summary()
    y_hat = model.predict(df_test.drop("Y", axis=1)).reshape(-1)

    print(pd.concat((df_test["Y"], pd.Series(y_hat, index=df_test.index)), axis=1).corr())
    print(np.mean(np.sqrt((df_test["Y"] - y_hat) ** 2)))
    print(np.mean(np.sqrt((df_test["Y"] - df_train["Y"].mean()) ** 2)))

    return y_hat


def auto_reg(df_train, df_test):
    model = AutoSklearnRegressor().fit(df_train.drop("Y", axis=1), df_train["Y"])

    y_hat = model.predict(df_test.drop("Y", axis=1))
    print(pd.concat((df_test["Y"], pd.Series(y_hat, index=df_test.index)), axis=1).corr())
    print(np.mean(np.sqrt((df_test["Y"] - y_hat) ** 2)))
    print(np.mean(np.sqrt((df_test["Y"] - df_train["Y"].mean()) ** 2)))
    return y_hat


def nn_plain(df_train, df_test):
    normalizer = layers.experimental.preprocessing.Normalization()
    normalizer.adapt(df_train.drop("Y", axis=1).to_numpy())
    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(6, activation="relu"),
        layers.Dense(6, activation="relu"),
        layers.Dense(1, activation="linear")
    ])
    #model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.001), loss="mean_squared_error")
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")
    fit_history = model.fit(df_train.drop("Y", axis=1), df_train["Y"], epochs=2500, batch_size=16, shuffle=True)
    y_hat = model.predict(df_test.drop("Y", axis=1)).reshape(-1)
    return y_hat

    plt.plot(fit_history.history["loss"]); plt.show()
    print(pd.concat((df_test["Y"], pd.Series(y_hat, index=df_test.index)), axis=1).corr())
    print(np.mean(np.sqrt((df_test["Y"] - y_hat) ** 2)))
    print(np.mean(np.sqrt((df_test["Y"] - df_train["Y"].mean()) ** 2)))

    return y_hat


def disp_performance(model, df_train, df_test):
    model.fit(df_train.drop("Y", axis=1), df_train["Y"])
    y_hat = model.predict(df_test.drop("Y", axis=1))
    print(pd.concat((df_test["Y"], pd.Series(y_hat, index=df_test.index)), axis=1).corr())
    print(np.mean(np.sqrt((df_test["Y"] - y_hat) ** 2)))
    print(np.mean(np.sqrt((df_test["Y"] - df_train["Y"].mean()) ** 2)))
    print("_" * 20)