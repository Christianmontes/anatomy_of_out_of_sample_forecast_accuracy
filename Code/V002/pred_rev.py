import glob
import itertools
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import collections
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from sklearn.decomposition import PCA
from data_cacher import DataCacher
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed, parallel_backend
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer, MinMaxScaler
from statsmodels.tsa.ar_model import ar_select_order
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, LeaveOneOut
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import ConvergenceWarning


def deep_update(source, overrides):
    # https://stackoverflow.com/a/30655448:
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


def iterative_estimation(df_all, y_all, h, start, config):

    rs = np.random.RandomState(seed=config["pred"]["random_seed"])

    def _df(train, test, y_key):
        from deepforest import CascadeForestRegressor
        model = CascadeForestRegressor(n_jobs=1)
        model.fit(train.drop(y_key, axis=1).to_numpy(), train[y_key].to_numpy())

        y_hat = pd.Series(model.predict(
            test.drop(y_key, axis=1).to_numpy()).flatten(), index=test.index).rename("df_y_hat")

        preds = [y_hat]
        models = [model]

        return preds, models

    def _xgb_cv(train, test, y_key, model_config):

        model = XGBRegressor(n_estimators=model_config["n_estimators"], learning_rate=model_config["eta"],
                             n_jobs=1, random_state=rs)

        cv_strategy = TimeSeriesSplit(n_splits=32, gap=h, test_size=3)

        model = GridSearchCV(estimator=model, param_grid=model_config["grid"], cv=cv_strategy, verbose=0, n_jobs=1,
                             scoring="neg_mean_squared_error", refit=False)

        model.fit(train.drop(y_key, axis=1), train[y_key], verbose=False)

        # to stabilize result can take median (instead of mean) of the n-folds or put more weight on recent folds:
        cv_scores = np.stack([model.cv_results_["split%i_test_score" % x] for x in range(cv_strategy.n_splits)])
        best_model_median = model.cv_results_["params"][np.argmax(np.median(cv_scores, axis=0))]
        best_model_mean = model.best_params_  # sklearn default, same as params[argmax(mean(cv_scores))]
        # weights = np.exp(-np.array([cv_strategy.n_splits - x for x in range(cv_strategy.n_splits)]) / 7.5)
        # best_model_exp = model.cv_results_["params"][np.argmax(np.average(cv_scores, axis=0, weights=weights))]
        weights = np.linspace(0.2, 1, cv_strategy.n_splits)
        best_model_lin = model.cv_results_["params"][np.argmax(np.average(cv_scores, axis=0, weights=weights))]

        mean_model = XGBRegressor(
            n_estimators=model_config["n_estimators"], learning_rate=model_config["eta"],
            random_state=rs, n_jobs=1, **best_model_mean
        ).fit(
            train.drop(y_key, axis=1), train[y_key], verbose=False
        )

        y_hat_mean = pd.Series(mean_model.predict(
            test.drop(y_key, axis=1)), index=test.index).rename("xgb_cv_y_hat")

        preds = [y_hat_mean]
        models = [mean_model]

        return preds, models

    def _nn_keras(train, test, y_key, depth_type, model_config):

        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.optimizers import Adam

        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.keras.backend.set_floatx("float64")

        k = train.shape[1] - 1
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train.drop(y_key, axis=1))
        layer_sizes = [max(1, int(np.round(k ** x))) for x in
                       model_config["%s_layer_exponents" % depth_type]]

        alpha = .0001

        def mlp_l2_norm(x):
            return alpha * 0.5 * tf.reduce_sum(tf.square(x))

        # from tensorflow.keras import regularizers
        # regularizer = regularizers.L2(alpha)
        regularizer = mlp_l2_norm

        models, preds = [], []
        for _ in tqdm(range(model_config["ensemble_n"])):
            model = Sequential()
            model.add(Dense(
                layer_sizes[0], input_shape=(k,), activation="relu", kernel_initializer="glorot_uniform",
                bias_initializer="glorot_uniform", kernel_regularizer=regularizer))
            for layer_size in layer_sizes[1:]:
                model.add(Dense(
                    layer_size, activation="relu", kernel_initializer="glorot_uniform",
                    bias_initializer="glorot_uniform", kernel_regularizer=regularizer))
            model.add(Dense(
                1, activation="linear", kernel_initializer="glorot_uniform",
                bias_initializer="glorot_uniform", kernel_regularizer=regularizer))
            model.compile(loss="mean_squared_error", optimizer=Adam(epsilon=1e-7, learning_rate=.001))
            model.fit(scaler.transform(train.drop(y_key, axis=1)), train[y_key], epochs=1000, batch_size=32, verbose=0)
            pred = model(scaler.transform(test.drop(y_key, axis=1))).numpy().flatten()
            models.append(model), preds.append(pred)

        model = models[np.argsort(np.hstack(preds))[model_config["ensemble_n"]//2]]
        pred = model(scaler.transform(test.drop(y_key, axis=1))).numpy().flatten()
        pred = pd.Series(pred, index=test.index).rename("nn_%s_y_hat" % depth_type)

        return [pred], [None]

    def _rf_cv(train, test, y_key, model_config):

        model = RandomForestRegressor(n_estimators=model_config["n_estimators"], n_jobs=1)

        # cv_strategy = KFold(20)
        # cv_strategy = TimeSeriesSplit(n_splits=12, gap=h, test_size=1)
        cv_strategy = TimeSeriesSplit(n_splits=32, gap=h, test_size=3)

        model = GridSearchCV(estimator=model, param_grid=model_config["grid"], cv=cv_strategy, verbose=0, n_jobs=1,
                             scoring="neg_mean_squared_error", refit=False)
        model.fit(train.drop(y_key, axis=1), train[y_key])

        # to stabilize result can take median (instead of mean) of the n-folds or put more weight on recent folds:
        cv_scores = np.stack([model.cv_results_["split%i_test_score" % x] for x in range(cv_strategy.n_splits)])
        best_model_median = model.cv_results_["params"][np.argmax(np.median(cv_scores, axis=0))]
        best_model_mean = model.best_params_  # sklearn default, same as params[argmax(mean(cv_scores))]
        # weights = np.exp(-np.array([cv_strategy.n_splits - x for x in range(cv_strategy.n_splits)]) / 7.5)
        # best_model_exp = model.cv_results_["params"][np.argmax(np.average(cv_scores, axis=0, weights=weights))]
        weights = np.linspace(0.2, 1, cv_strategy.n_splits)
        best_model_lin = model.cv_results_["params"][np.argmax(np.average(cv_scores, axis=0, weights=weights))]

        mean_model = RandomForestRegressor(
            n_jobs=1, n_estimators=model_config["n_estimators"], random_state=rs, **best_model_mean).fit(
            train.drop(y_key, axis=1), train[y_key])

        print("rf_cv", best_model_mean)

        y_hat_mean = pd.Series(mean_model.predict(
            test.drop(y_key, axis=1)), index=test.index).rename("rf_cv_y_hat")

        preds = [y_hat_mean]
        models = [mean_model]

        return preds, models

    def _rf(train, test, y_key, model_config):

        model = RandomForestRegressor(n_estimators=model_config["n_estimators"], n_jobs=1)
        model.fit(train.drop(y_key, axis=1), train[y_key])

        y_hat = pd.Series(model.predict(test.drop(y_key, axis=1)), index=test.index).rename("rf_y_hat")

        preds = [y_hat]
        models = [model]

        return preds, models

    def _nn(train, test, y_key, depth_type, model_config):

        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train.drop(y_key, axis=1))
        layer_sizes = [max(1, int(np.round((train.shape[1] - 1) ** x))) for x in
                       model_config["%s_layer_exponents" % depth_type]]

        preds = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            if model_config["cv"]:
                model = MLPRegressor(
                    hidden_layer_sizes=layer_sizes, batch_size=model_config["batch_size"],
                    learning_rate="constant", learning_rate_init=model_config["learning_rate"],
                    activation=model_config["activation"], solver="adam",
                    max_iter=model_config["epochs"],
                    n_iter_no_change=model_config["epochs"], verbose=False, random_state=rs
                )
                cv_strategy = TimeSeriesSplit(n_splits=12, gap=h, test_size=1)
                model = GridSearchCV(
                    estimator=model, param_grid=model_config["grid"], cv=cv_strategy, verbose=0,
                    n_jobs=1, scoring="neg_mean_squared_error", refit=False
                )
                model.fit(scaler.transform(train.drop(y_key, axis=1)), train[y_key])
                best_model_mean = model.best_params_  # sklearn default, same as params[argmax(mean(cv_scores))]

            models = []
            for _ in range(model_config["ensemble_n"]):
                if model_config["cv"]:
                    model = MLPRegressor(hidden_layer_sizes=layer_sizes, batch_size=model_config["batch_size"],
                                         learning_rate="constant", learning_rate_init=model_config["learning_rate"],
                                         activation=model_config["activation"], solver="adam",
                                         max_iter=model_config["epochs"], n_iter_no_change=model_config["epochs"],
                                         verbose=False, random_state=rs, **best_model_mean)
                else:
                    model = MLPRegressor(hidden_layer_sizes=layer_sizes, batch_size=model_config["batch_size"],
                                         learning_rate="constant", learning_rate_init=model_config["learning_rate"],
                                         activation=model_config["activation"], solver="adam",
                                         max_iter=model_config["epochs"], n_iter_no_change=model_config["epochs"],
                                         verbose=False, random_state=rs)  # alpha=default

                model.fit(scaler.transform(train.drop(y_key, axis=1)), train[y_key])
                preds.append(model.predict(scaler.transform(test.drop(y_key, axis=1))))

                models.append(model)

        model = models[np.argsort(np.hstack(preds))[model_config["ensemble_n"]//2]]
        y_hat_med = pd.Series(np.median(preds), index=test.index).rename("nn_%s_med_y_hat" % depth_type)

        return [y_hat_med], [model]

    def _ar(train, test, model_config):
        model = ar_select_order(train, maxlag=model_config["max_lags"], ic=model_config["ic"],
                                old_names=False).model.fit()
        pred = pd.Series(model.predict(
            start=train.index.max() + pd.DateOffset(months=1),
            end=train.index.max() + pd.DateOffset(months=h)).mean(), index=test.index).rename("ar_y_hat")
        return [pred], [model]

    def _ee(train, test, y_key):
        models = dict([(x_key, LinearRegression(n_jobs=1, fit_intercept=False).fit(
            train[[x_key]]-train[x_key].mean(), train[y_key])) for x_key in train.columns if x_key != y_key])
        pred = pd.Series([np.mean([
            models[x_key].predict(test[[x_key]]-train[x_key].mean())
            for x_key in train.columns if x_key != y_key])], index=test.index).rename("ee_y_hat")
        return [pred], [models]

    def _univar_lin_comb(train, test, y_key):
        models = dict([(x_key, LinearRegression(n_jobs=1, fit_intercept=True).fit(
            train[[x_key]], train[y_key])) for x_key in train.columns if x_key != y_key])
        pred = pd.Series([np.mean([
            models[x_key].predict(test[[x_key]])
            for x_key in train.columns if x_key != y_key])], index=test.index).rename("univar_lin_comb_y_hat")
        """models = dict([(x_key, LinearRegression(n_jobs=1, fit_intercept=True).fit(
            train[[x_key]] - train[[x_key]].mean(), train[y_key])) for x_key in train.columns if x_key != y_key])
        pred = pd.Series([np.mean([
            models[x_key].predict(test[[x_key]] - train[[x_key]].mean())
            for x_key in train.columns if x_key != y_key])], index=test.index).rename("univar_lin_comb_y_hat")"""
        return [pred], [models]

    def _combination_enet(train, test, y_key):

        gap = h
        holdout_size = 120
        train_sub, holdout = train.iloc[:-holdout_size-gap], train.iloc[-holdout_size:]

        models = dict([(x_key, LinearRegression(n_jobs=1, fit_intercept=True).fit(
            train_sub[[x_key]], train_sub[y_key])) for x_key in train_sub.columns if x_key != y_key])

        y_hats_holdout = np.stack([models[x].predict(holdout[[x]]) for x in models.keys()]).T

        def _aicc(x_norm, y, alpha, l1_ratio):
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True).fit(x_norm, y)
            obs = y.shape[0]
            sse = np.sum((y - model.predict(x_norm)) ** 2)
            df = np.count_nonzero(model.coef_)
            return obs * np.log(sse / obs) + 2 * df * np.log(obs) / (np.log(obs) - df)

        l1_ratios = [.1, .5, .7, .9, .95, .99, 1]
        alphas = np.logspace(np.log10(0.001), np.log10(10), 100)
        params = list(itertools.product(alphas, l1_ratios))

        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
            holdout_x_mean = y_hats_holdout.mean(axis=0)
            holdout_x_std = y_hats_holdout.std(axis=0)
            holdout_x_norm = (y_hats_holdout - holdout_x_mean) / holdout_x_std
            ics = [
                _aicc(y_hats_holdout, holdout[y_key], alpha, l1_ratio)
                for alpha, l1_ratio in params
            ]
            alpha, l1_ratio = params[np.argmin(ics)]
            enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True).fit(holdout_x_norm, holdout[y_key])

        cols = [x for x, coef in list(zip(models.keys(), enet.coef_)) if coef > 0]

        if len(cols) == 0:
            models = {}
            pred = pd.Series(np.mean(train[y_key]), index=test.index).rename("cenet_y_hat")
        else:
            models = dict([(x_key, LinearRegression(n_jobs=1, fit_intercept=True).fit(
                train[[x_key]], train[y_key])) for x_key in cols])
            pred = pd.Series([np.mean([
                models[x_key].predict(test[[x_key]])
                for x_key in cols])], index=test.index).rename("cenet_y_hat")

        return [pred], [models]

    def _enet_univar_lin_comb(train, test, y_key, model_config):

        x_mean, x_std = train.drop(y_key, axis=1).mean().to_numpy(), train.drop(y_key, axis=1).std().to_numpy()
        ts_cv = TimeSeriesSplit(n_splits=model_config["cv_n_splits"], gap=h, test_size=model_config["cv_test_size"])

        model = ElasticNetCV(
            alphas=model_config["alphas"], l1_ratio=model_config["l1_ratios"], fit_intercept=True, cv=ts_cv,
            n_jobs=1, random_state=rs, max_iter=100000)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit((train.drop(y_key, axis=1) - x_mean) / x_std, train[y_key])

        cols = train.drop(y_key, axis=1).columns[~np.isclose(model.coef_, 0)]

        if cols.shape[0] == 0:
            models = {}
            pred = pd.Series(np.mean(train[y_key]), index=test.index).rename("enet_univar_lin_comb_y_hat")
        else:
            models = dict([(x_key, LinearRegression(n_jobs=1, fit_intercept=True).fit(
                train[[x_key]], train[y_key])) for x_key in cols])
            pred = pd.Series([np.mean([
                models[x_key].predict(test[[x_key]])
                for x_key in cols])], index=test.index).rename("enet_univar_lin_comb_y_hat")

        return [pred], [models]

    def _pca_ylag(train, test, y_key, model_config):
        y_lag_keys = [x for x in train.columns if x.startswith("y_t-")]
        x_mean = train.drop([y_key] + y_lag_keys, axis=1).mean().to_numpy()
        x_std = train.drop([y_key] + y_lag_keys, axis=1).std().to_numpy()
        x_norm = (train.drop([y_key] + y_lag_keys, axis=1) - x_mean) / x_std
        pca_model = PCA(n_components="mle").fit(x_norm)

        n_components_max = model_config["n_components_max"]

        models = {}
        factor_idx = np.argsort(pca_model.explained_variance_ratio_)[::-1]  # ensure factors are sorted
        for i in range(1, n_components_max+1):
            for j in range(0, len(y_lag_keys)+1):
                train_concat = np.hstack((train[y_lag_keys[:j]], pca_model.transform(x_norm)[:, factor_idx[:i]]))
                model = LinearRegression(fit_intercept=True).fit(train_concat, train[[y_key]])
                adj_r2 = 1 - (1 - model.score(train_concat, train[y_key])) * (train.shape[0] - 1) / (train.shape[0] - (i+j) - 1)
                models[(i, j)] = (adj_r2, model)

        sel_model = list(models.keys())[np.argmax([models[x][0] for x in list(models.keys())])]
        model = models[sel_model][1]
        x_test_norm = (test.drop([y_key] + y_lag_keys, axis=1) - x_mean) / x_std
        test_concat = np.hstack((test[y_lag_keys[:sel_model[1]]], pca_model.transform(x_test_norm)[:, factor_idx[:sel_model[0]]]))
        pred = model.predict(test_concat).flatten()
        pred = pd.Series(pred, index=test.index).rename("pca_ylag_y_hat")

        return [pred], [[pca_model, x_mean, x_std], [model, sel_model, y_lag_keys[:sel_model[1]]]]

    def _pca(train, test, y_key, model_config):

        x_mean, x_std = train.drop(y_key, axis=1).mean().to_numpy(), train.drop(y_key, axis=1).std().to_numpy()
        x_norm = (train.drop(y_key, axis=1) - x_mean) / x_std
        pca_model = PCA().fit(x_norm)
        x_transformed = pca_model.transform(x_norm)

        n_components_max = model_config["n_components_max"]

        factor_idx = np.argsort(pca_model.explained_variance_ratio_)[::-1]  # ensure factors are sorted
        models = [LinearRegression(fit_intercept=True).fit(
            x_transformed[:, factor_idx[:i+1]], train[y_key]) for i in range(n_components_max)]
        # bics = [sm.OLS(train[y_key], sm.add_constant(
        #    x_transformed[:, factor_idx[:i+1]])).fit().bic for i in range(n_components_max)]
        adj_r2 = [
            1 - (1 - models[i].score(x_transformed[:, factor_idx[:i+1]], train[y_key])) *
            (train.shape[0] - 1) / (train.shape[0] - (i+1) - 1)
            for i in range(n_components_max)]
        model_idx = np.argmax(adj_r2)
        # model_idx = np.argmin(bics)
        model = models[model_idx]
        pred = model.predict(pca_model.transform(
            (test.drop(y_key, axis=1) - x_mean) / x_std)[:, factor_idx[:model_idx+1]])
        pred = pd.Series(pred, index=test.index).rename("pca_y_hat")

        return [pred], [[pca_model, x_mean, x_std], model]

    def _pca_dropping(train, test, y_key, model_config):

        for _ in range(10000):
            x_mean, x_std = train.drop(y_key, axis=1).mean().to_numpy(), train.drop(y_key, axis=1).std().to_numpy()
            x_norm = (train.drop(y_key, axis=1) - x_mean) / x_std
            pca_model = PCA().fit(x_norm)
            x_transformed = pca_model.transform(x_norm)

            n_components_max = model_config["n_components_max"]

            factor_idx = np.argsort(pca_model.explained_variance_ratio_)[::-1]  # ensure factors are sorted
            models = [LinearRegression(fit_intercept=True).fit(
                x_transformed[:, factor_idx[:i + 1]], train[y_key]) for i in range(n_components_max)]
            # bics = [sm.OLS(train[y_key], sm.add_constant(
            #    x_transformed[:, factor_idx[:i+1]])).fit().bic for i in range(n_components_max)]
            adj_r2 = [
                1 - (1 - models[i].score(x_transformed[:, factor_idx[:i + 1]], train[y_key])) *
                (train.shape[0] - 1) / (train.shape[0] - (i + 1) - 1)
                for i in range(n_components_max)]
            model_idx = np.argmax(adj_r2)
            # model_idx = np.argmin(bics)
            model = models[model_idx]
            pred = model.predict(pca_model.transform(
                (test.drop(y_key, axis=1) - x_mean) / x_std)[:, factor_idx[:model_idx + 1]])
            pred = pd.Series(pred, index=test.index).rename("pca_y_hat")

        return [pred], [[pca_model, x_mean, x_std], model]

        while True:
            x_mean, x_std = train.drop(y_key, axis=1).mean().to_numpy(), train.drop(y_key, axis=1).std().to_numpy()
            x_norm = (train.drop(y_key, axis=1) - x_mean) / x_std
            pca_model = PCA().fit(x_norm)
            x_transformed = pca_model.transform(x_norm)

            n_components_max = model_config["n_components_max"]

            factor_idx = np.argsort(pca_model.explained_variance_ratio_)[::-1]  # ensure factors are sorted
            models = [LinearRegression(fit_intercept=True).fit(
                x_transformed[:, factor_idx[:i+1]], train[y_key]) for i in range(n_components_max)]
            # bics = [sm.OLS(train[y_key], sm.add_constant(
            #    x_transformed[:, factor_idx[:i+1]])).fit().bic for i in range(n_components_max)]
            adj_r2 = [
                1 - (1 - models[i].score(x_transformed[:, factor_idx[:i+1]], train[y_key])) *
                (train.shape[0] - 1) / (train.shape[0] - (i+1) - 1)
                for i in range(n_components_max)]
            model_idx = np.argmax(adj_r2)
            # model_idx = np.argmin(bics)
            model = models[model_idx]
            pred = model.predict(pca_model.transform(
                (test.drop(y_key, axis=1) - x_mean) / x_std)[:, factor_idx[:model_idx+1]])
            pred = pd.Series(pred, index=test.index).rename("pca_y_hat")

            n_components = model.coef_.shape[0]
            explain_transformed = pca_model.transform((test.drop(y_key, axis=1) - x_mean) / x_std)[:, :n_components]
            shap_values_pca = np.stack([
                ((test.drop(y_key, axis=1).to_numpy() - x_mean) / x_std) *
                pca_model.components_[i, :] / test.drop(y_key, axis=1).shape[1]
                for i in range(n_components)
            ]).T
            shap_values_ols = explain_transformed * model.coef_
            shap_factors = shap_values_pca.sum(axis=0) / shap_values_ols
            shap_values = (shap_values_pca / shap_factors).sum(axis=-1).T
            shap_values = pd.DataFrame(shap_values, columns=test.drop(y_key, axis=1).columns, index=test.index)
            base_value = model.intercept_

            pbvi = pd.concat([(((shap_values.drop(x, axis=1).sum(axis=1) + base_value) - test[y_key]) ** 2 - (pred - test[y_key]) ** 2).rename(x) for x in shap_values.columns], axis=1)
            if (pbvi < 0).any(axis=1).any():
                to_del = pbvi.idxmin(axis=1)[0]
                train, test = train.drop(to_del, axis=1), test.drop(to_del, axis=1)
                # print(to_del)
                if train.shape[1]-1 == 1:
                    pred = pd.Series(np.mean(train[y_key]), index=test.index).rename("pca_y_hat")
                    return [pred], [None]
            else:
                break

        return [pred], [[pca_model, x_mean, x_std], model]

    def _ols(train, test, y_key):
        model = LinearRegression().fit(train.drop(y_key, axis=1), train[[y_key]])
        pred = pd.Series(model.predict(test.drop(y_key, axis=1)).flatten(), index=test.index).rename("ols_y_hat")
        return [pred], [model]

    def _pls(train, test, y_key):

        n_components_max = 1

        models = [PLSRegression(n_components=i+1).fit(
            train.drop(y_key, axis=1), train[y_key]) for i in range(n_components_max)]
        adj_r2 = [
            1 - (1 - models[i].score(train.drop(y_key, axis=1), train[y_key])) *
            (train.shape[0] - 1) / (train.shape[0] - (i+1) - 1)
            for i in range(n_components_max)]
        model_idx = np.argmax(adj_r2)
        model = models[model_idx]
        pred = pd.Series(model.predict(test.drop(y_key, axis=1)).flatten(), index=test.index).rename("pls_y_hat")

        return [pred], [model]

    def _ln(name, train, test, y_key, model_config):
        x_mean, x_std = train.drop(y_key, axis=1).mean().to_numpy(), train.drop(y_key, axis=1).std().to_numpy()
        ts_cv = TimeSeriesSplit(n_splits=model_config["cv_n_splits"], gap=h, test_size=model_config["cv_test_size"])
        models = {
            "ridge":
                RidgeCV(
                    alphas=model_config["alphas"], fit_intercept=True, cv=ts_cv,
                    scoring="neg_mean_squared_error"),
            "lasso":
                LassoCV(
                    alphas=model_config["alphas"], fit_intercept=True, cv=ts_cv,
                    n_jobs=1, random_state=rs, max_iter=100000),
            "enet":
                ElasticNetCV(
                    alphas=model_config["alphas"], l1_ratio=model_config["l1_ratios"], fit_intercept=True, cv=ts_cv,
                    n_jobs=1, random_state=rs, max_iter=100000),
        }
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            models[name].fit((train.drop(y_key, axis=1) - x_mean) / x_std, train[y_key])
        pred = pd.Series(
            models[name].predict((test.drop(y_key, axis=1) - x_mean) / x_std),
            index=test.index).rename("%s_y_hat" % name)

        return [pred], [[x_mean, x_std], models[name]]

    def _run(payload):

        train, train_ar, test, rolling_start, run_config = payload

        pred_key = "y_h%i" % h

        algos = {
            "df":
                lambda x: _df(
                    train.iloc[x:], test, pred_key
                ),
            "xgb_cv":
                lambda x: _xgb_cv(
                    train.iloc[x:], test, pred_key, run_config["algo_configs"]["xgb_cv"]
                ),
            "rf_cv":
                lambda x: _rf_cv(
                    train.iloc[x:], test, pred_key, run_config["algo_configs"]["rf_cv"]
                ),
            "nn_shallow":
                lambda x: _nn(
                    train.iloc[x:], test, pred_key, "shallow", run_config["algo_configs"]["nn"]
                ),
            "nn_deep":
                lambda x: _nn(
                    train.iloc[x:], test, pred_key, "deep", run_config["algo_configs"]["nn"]
                ),
            "ar":
                lambda x: _ar(
                    train_ar.iloc[x:], test, run_config["algo_configs"]["ar"]
                ),
            "ridge":
                lambda x: _ln(
                    "ridge", train.iloc[x:], test, pred_key, run_config["algo_configs"]["ridge_lasso_enet"]
                ),
            "lasso":
                lambda x: _ln(
                    "lasso", train.iloc[x:], test, pred_key, run_config["algo_configs"]["ridge_lasso_enet"]
                ),
            "enet":
                lambda x: _ln(
                    "enet", train.iloc[x:], test, pred_key, run_config["algo_configs"]["ridge_lasso_enet"]
                ),
            "univar_lin_comb":
                lambda x: _univar_lin_comb(
                    train.iloc[x:], test, pred_key
                ),
            "ee":
                lambda x: _ee(
                    train.iloc[x:], test, pred_key
                ),
            "enet_univar_lin_comb":
                lambda x: _enet_univar_lin_comb(
                    train.iloc[x:], test, pred_key, run_config["algo_configs"]["ridge_lasso_enet"]
                ),
            "pls":
                lambda x: _pls(
                    train.iloc[x:], test, pred_key
                ),
            "pca":
                lambda x: _pca(
                    train.iloc[x:], test, pred_key, run_config["algo_configs"]["pca"]
                ),
            "pca_ylag":
                lambda x: _pca_ylag(
                    train.iloc[x:], test, pred_key, run_config["algo_configs"]["pca"]
                ),
            "ols":
                lambda x: _ols(
                    train.iloc[x:], test, pred_key
                ),
            "cenet":
                lambda x: _combination_enet(
                    train.iloc[x:], test, pred_key
                ),
            "rf":
                lambda x: _rf(
                    train.iloc[x:], test, pred_key, run_config["algo_configs"]["rf"]
                ),
        }

        model_data = {
            "pred_date": test.index[0] + pd.DateOffset(months=h),
            "train": train.iloc[rolling_start:].copy() if config["pred"]["window_type"] == "rolling" else train.copy(),
            "test": test.copy(),
            "y_hats": {}, "models": {}
        }

        for model_name in config["pred"]["active_algos"]:
            y_hat, model = algos[model_name](x=rolling_start if config["pred"]["window_type"] == "rolling" else 0)
            model_data["y_hats"][model_name], model_data["models"][model_name] = y_hat, model

        return model_data

    df_test = df_all.copy()
    df_test["y"] = df_test["y"].rolling(h).mean().shift(-h)
    df_test = df_test.rename({"y": "y_h%i" % h}, axis=1).dropna()

    init_window = np.where(df_all.index >= start)[0].min()

    rolling_offset, payloads = 0, []
    for t in range(init_window - h + 1, df_test.shape[0] + 1):
        df_train = df_all.iloc[:t].copy()
        df_train_ar = y_all.loc[:df_train.index.max()].copy()  # distinct df for ar; iterated forecast always h=1
        df_eval = df_test.loc[[df_train.index.max()]].copy()  # contains y h obs in the future

        df_train["y"] = df_train["y"].rolling(h).mean().shift(-h)
        df_train = df_train.rename({"y": "y_h%i" % h}, axis=1).dropna()  # last h obs for direct forecasts not observed

        payloads.append(delayed(_run)(payload=(df_train, df_train_ar, df_eval, rolling_offset, config)))
        rolling_offset += 1

    with parallel_backend("loky", inner_max_num_threads=1):  # force one thread (e.g., OMP_NUM_THREADS=1)
        model_dict = Parallel(n_jobs=config["n_threads"], verbose=True)(payloads)

    if 1 == 2:
        for algo in config["pred"]["active_algos"]:
            y_hat = pd.concat([model_dict[x]["y_hats"][algo][0] for x in range(len(model_dict))])
            y_hat_mean = pd.Series([model_dict[x]["train"]["y_h%i" % h].mean() for x in range(len(model_dict))], index=y_hat.index)
            y = pd.concat([model_dict[x]["test"]["y_h%i" % h] for x in range(len(model_dict))])
            y.index += pd.DateOffset(months=h)
            y_hat.index += pd.DateOffset(months=h)
            y_hat_mean.index += pd.DateOffset(months=h)
            if 1 == 2:
                start_date, end_date = "1966-12-01", "2020-12-01"
            else:
                start_date, end_date = "2000-12-01", "2022-01-01"
            y = y.loc[start_date:end_date]
            y_hat_mean = y_hat_mean.loc[start_date:end_date]
            y_hat = y_hat.loc[start_date:end_date]
            print(h, algo, 1 - ((y-y_hat)**2).sum() / ((y-y_hat_mean)**2).sum(), np.sqrt(((y-y_hat)**2).mean()))

    return model_dict


def run_pred():

    if 1 == 2:
        override_config = {
            "n_threads": 60,
            "pred": {
                "start": "01-01-1966",
                "y_names": ["rx"],
                "x_names": ["nrtz"],
                "window_type": "expanding",
                "active_algos": ["pca_"],
                "horizons": [1, 3, 6, 12]
            },
            "algo_configs": {
                "nn": {
                    "ensemble_n": 100,
                }
            }
        }
    else:
        override_config = {
            "n_threads": 32,
            "pred": {
                "start": "01-01-1990",
                "x_names": ["fredmd", "soc", "fredmd_ma3", "soc_ma3", "y_lags"],
                "y_names": ["cpiaucsl"],
                "window_type": "rolling",
                # "active_algos": ["xgb_cv", "rf_cv"],  # "nn_deep", "nn_shallow"],
                "active_algos": ["xgb_cv", "nn_deep", "nn_shallow", "rf_cv", "enet", "pca", "ar"],
                # "active_algos": ["rf_cv"],
                "horizons": [12, 6, 3, 1]
            },
            "algo_configs": {
                "pca": {
                    "n_components_max": 10
                },
                "nn": {
                    "ensemble_n": 200-1,
                    # "ensemble_n": 20-1,
                }
            }
        }

    default_config = {
        "n_threads": os.cpu_count() - 2,
        "pred": {
            "start": "01-01-2000",
            "active_algos": ["xgb_cv", "rf_cv", "nn_shallow", "nn_deep", "ar", "ridge", "lasso", "enet"],
            "x_names": ["fredmd", "ism", "soc", "y_lags"],
            "y_names": ["cpiaucsl", "ce16ov", "indpro"],
            "fred_pca": False,
            "horizons": [1, 3, 6, 12],
            "window_type": "rolling",
            "random_seed": 43210,
            "run_name": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        },
        "algo_configs": {
            "xgb_cv": {
                "n_estimators": 500,
                "eta": 0.3,
                "grid": {
                    "max_depth": [1, 2, 5],
                    "subsample": [0.6, 0.8, 1],
                    "colsample_bytree": [0.6, 0.8, 1]
                }
            },
            "rf": {
                "n_estimators": 500
            },
            "rf_cv": {
                "n_estimators": 500,
                "grid": {
                    # "max_features": [None, 2 / 3, 1 / 3, "sqrt", "log2"]
                    "max_features": np.linspace(0.05, 1, 10).tolist()
                }
            },
            "nn": {
                "batch_size": 32,
                "learning_rate": 0.01,
                "epochs": 1000,
                "activation": "relu",
                "shallow_layer_exponents": [1/2],
                "deep_layer_exponents": [3/4, 2/4, 1/4],
                "ensemble_n": 5,
                "cv": False,  # cv is very time consuming
                "grid": {"alpha": np.logspace(np.log10(0.001), np.log10(10), 10).tolist()}
            },
            "ar": {
                "max_lags": 12,
                "ic": "bic"
            },
            "ridge_lasso_enet": {
                "alphas": np.logspace(np.log10(0.001), np.log10(10), 100).tolist(),
                "l1_ratios": [.1, .5, .7, .9, .95, .99, 1],
                "cv_n_splits": 32,
                "cv_test_size": 3
            },
            "pca": {
                "n_components_max": 4
            }
        }
    }

    config = deep_update(default_config, override_config)

    Path("/FAST_STORE/IML_MODELS/%s" % config["pred"]["run_name"]).mkdir(parents=True, exist_ok=True)
    Path("./_runs/%s" % config["pred"]["run_name"]).mkdir(parents=True, exist_ok=True)
    Path("./_runs/%s/config.json" % config["pred"]["run_name"]).write_text(json.dumps(config, indent=4))

    cache = DataCacher().data

    def _impute_mean(col):
        _col = col.copy()
        for na_index in np.where(_col.isnull())[0]:
            _col.iloc[na_index] = _col.iloc[:na_index].mean()  # conditional mean
        return _col

    def _match_yx(y, x):
        _df = pd.concat((y.dropna(), *(x if type(x) is tuple or type(x) is list else (x,))), axis=1, join="outer")
        return _df["y"], _df[[x for x in _df.columns if x != "y"]]

    for y_name in tqdm(config["pred"]["y_names"]):

        for horizon in tqdm(config["pred"]["horizons"]):

            fred_md_raw = cache.get_fred_md_raw()
            fred_md_transformed = cache.get_fred_md()
            # ism = cache.get_ism()
            soc = cache.get_soc()
            nrtz = cache.get_nrtz()

            # these series are dropped as they were redefined in 2020 and are thus invalid:
            fred_md_raw.drop(["m1sl", "m2sl", "m2real"], axis=1, inplace=True)
            fred_md_transformed.drop(["m1sl", "m2sl", "m2real"], axis=1, inplace=True)

            if y_name == "rx":
                y_t = nrtz[y_name].rename("y")
            else:
                y_t = np.log(fred_md_raw[y_name]).diff(1).rename("y")

            x_t = {
                "fredmd": fred_md_transformed.drop(y_name, axis=1, errors="ignore").rename(
                    lambda x: "fred_%s" % x, axis=1),
                "soc": soc.ffill(),  # sometimes quarterly (ffill)
                # "ism": ism,
                # "nrtz": nrtz.drop(y_name, axis=1, errors="ignore").rename(lambda x: "nrtz_%s" % x, axis=1),
            }

            for x_key in list(x_t.keys()):
                x_t["%s_ma3" % x_key] = x_t[x_key].rolling(3, min_periods=3, closed="both").mean().rename(
                    lambda x: "%s_ma3" % x, axis=1)

            x_t = [x_t[x] for x in config["pred"]["x_names"] if x != "y_lags"]
            y_t, x_t = _match_yx(y=y_t, x=x_t)

            if np.isin(["fredmd", "soc"], config["pred"]["x_names"]).any():  # "ism"
                first = "1959-02-01"  # remove first obs (max obs lost with modified fred transform)
                last = "2025-12-01"
                x_t = x_t.loc[first:last].apply(lambda col: _impute_mean(col), axis=0)  # mean impute missing values
                y_t = y_t.loc[first:last].dropna()

            # add n lagged (and contemporaneous) y, drop n first obs:
            if "y_lags" in config["pred"]["x_names"]:
                x_t = pd.concat((pd.concat([y_t.shift(x).rename("y_t-%i" % (x+1)) for x in range(12)], axis=1).dropna(),
                                 x_t), axis=1, join="inner")

            # remove columns in x with any missing values not imputed; importantly after removing n first obs:
            x_t = x_t.loc[:, ~x_t.isnull().any(axis=0)]

            # todo: debug
            if 1 == 2:
                # all bad ones (univar_lin_comb):
                to_drop = {
                    1: ["bm", "ep", "dfr", "dfy", "ma0312", "vol0109", "mom09", "mom12"],
                    3: ["bm", "ep", "dfy", "ma0309", "ma0209", "ma0109", "ma0112", "dfr", "ma0212", "mom12", "ma0312", "mom09"],
                    6: ["bm", "ep", "dfy", "mom12", "vol0309", "vol0209", "vol0212", "mom09", "ma0212", "ma0312", "ma0109", "ma0112", "ma0209", "ma0309", "dfr"],
                    12: ["bm", "mom12", "mom09", "ma0212", "dfy", "ma0312", "ma0112", "vol0312", "vol0212", "ma0309", "ma0109", "ma0209", "dfr"]
                }
                # all bad ones (univar_lin_comb - special method):
                to_drop = {
                    1: ["ma0309", "de", "ma0112", "infl", "ntis", "bm", "ep", "ma0209", "vol0312", "dfr", "vol0209", "ma0109", "vol0212", "dfy", "ma0312", "vol0309", "vol0109", "mom09", "mom12"],
                    3: ["bm", "ntis", "ep", "ltr", "dfy", "vol0309", "vol0312", "vol0209", "ma0309", "vol0212", "ma0209", "ma0109", "vol0109", "ma0112", "dfr", "ma0212", "dfr", "ma0212", "mom12", "ma0312", "mom09"],
                    6: ["bm", "ep", "ntis", "dfy", "vol0112", "mom12", "vol0109", "vol0312", "vol0309", "vol0209", "vol0212", "mom09", "ma0212", "ma0312", "ma0109", "ma0112", "ma0209", "ma0309", "dfr"],
                    12: ["dy", "dp", "bm", "ep", "mom12", "ntis", "de", "infl", "rvol", "ltr", "mom09", "ma0212", "dfy", "ma0312", "ma0112", "vol0112", "vol0309", "vol0109", "vol0209", "vol0312", "vol0212", "ma0309", "ma0109", "ma0209", "dfr"]
                }
                # all bad ones (pca):
                to_drop = {
                    1: ["dy", "dp", "dfr", "de", "infl", "bm", "ma0309", "ep", "ma0112", "ma0209", "mom09", "ma0212", "ma0312", "mom12", "ma0109", "vol0109", "ntis"],
                    3: ["dp", "dy", "infl", "bm", "mom09", "mom12", "vol0112", "ep", "ma0109", "ma0312", "ma0112", "ma0209", "ma0212", "ma0309", "dfr", "ntis"],
                    6: ["dy", "dp", "infl", "mom12", "bm", "ep", "mom09", "ma0312", "ma0212", "ma0112", "ma0109", "ma0209", "ma0309", "dfr", "ntis"],
                    12: ["dy", "dp", "mom12", "bm", "mom09", "ma0212", "ma0312", "ma0112", "ma0209", "ma0109", "ma0309", "dfr", "ntis"]
                }
                to_drop = {
                    1: ["ltr"],
                    3: ["tbl"],
                    6: ["tbl"],
                    12: ["tms"]
                }
                x_t = x_t.drop(["nrtz_%s" % x for x in to_drop[horizon]], axis=1)  # drop bad ones
                # x_t = x_t.drop([x for x in x_t.columns if x.replace("nrtz_", "") not in to_drop[horizon]], axis=1)  # drop good ones

            df = pd.concat((y_t, x_t), axis=1, join="inner")  # inner join y with x
            df.index = pd.DatetimeIndex(df.index, freq="MS")  # specify frequency of index (for ar)

            models = iterative_estimation(df, y_t, h=horizon, start=config["pred"]["start"], config=config)
            pickle.dump(models, open(
                "/FAST_STORE/IML_MODELS/%s/%s_h%i.bin" % (config["pred"]["run_name"], y_name, horizon), "wb"))


def run_eval():

    files = glob.glob("../../Results/Updated Models/*.bin")
    y_hat_writers, y_true_writers = {}, {}

    for f in tqdm(files):

        data = pickle.load(open(f, "rb"))
        data = pd.concat([pd.Series(x).rename(x["index"]) for x in data], axis=1).T.sort_index()
        y_name, h = f.split("/")[-1].split("_")[0], int(f.split("/")[-1].split("_h")[-1].split(".bin")[0])

        if y_name not in y_true_writers:
            y_true_writers[y_name] = pd.ExcelWriter(
                "../../Results/Updated Model Output/True/%s_y.xlsx" % y_name)
            y_hat_writers[y_name] = pd.ExcelWriter(
                "../../Results/Updated Model Output/Predicted/%s_y_hat.xlsx" % y_name)

        y_true = data["eval_data"].apply(lambda x: x["y_h%i" % h].iloc[0]).rename("y_true")
        y_true.index.name = "date"
        y_true.to_excel(y_true_writers[y_name], sheet_name="h%i" % h)

        model_names = data.iloc[0]["rolling"].keys()
        for model_name in tqdm(model_names):
            y_hat = []
            for date in tqdm(data.index):
                model = data.loc[date]["rolling"][model_name]
                model = model[0] if type(model) is list and len(model) == 1 else model
                train_data = data.loc[date]["rolling_data"].drop("y_h%i" % h, axis=1)
                eval_data = data.loc[date]["eval_data"].drop("y_h%i" % h, axis=1)
                if "tsa.ar_model" in str(type(model)):
                    eval_row = model.predict(start=model.data.dates.max() + pd.DateOffset(months=1),
                                             end=model.data.dates.max() + pd.DateOffset(months=h)).mean()
                elif type(model) is list and "MLPRegressor" in str(type(model[0])):
                    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_data)
                    eval_row = np.median(np.hstack([x.predict(scaler.transform(eval_data)) for x in model]))
                else:
                    eval_row = model.predict(eval_data)
                y_hat.append(pd.Series(eval_row, index=[date]))
            y_hat = pd.concat(y_hat).rename("%s_h%i_y_hat" % (model_name, h))
            y_hat.index.name = "date"
            y_hat.to_excel(y_hat_writers[y_name], sheet_name="%s_h%i" % (model_name, h))

    [x.close() for x in y_hat_writers.values()], [x.close() for x in y_true_writers.values()]
    exit(-1)


def export(run_name):
    for f in tqdm(glob.glob("../../Results/%s/*.bin" % run_name)):
        data = pickle.load(open(f, "rb"))
        out = []
        for row in data:
            out.append(pd.concat((
                pd.Series([row["eval_data"], row["rolling_data"]], index=["data_eval", "data_rolling"]),
                pd.Series(dict([("model_%s" % x, row["rolling"][x] if x.startswith("nn") else row["rolling"][x][0]) for x in row["rolling"].keys()]))
            )).rename(row["index"]))
        out = pd.concat(out, axis=1).T
        pickle.dump(out, open("../../Results/Updated Models/%s" % f.split("/")[-1].replace("models_", ""), "wb"))


def export_nns(run_key):

    # hs = [1, 3, 6, 12]
    hs = [12,]
    for h in hs:
        ds = []
        d = pickle.load(open("/FAST_STORE/IML_MODELS/%s/cpiaucsl_h%i.bin" % (run_key, h), "rb"))
        y_key = "y_h%i" % h
        for row in d:
            test = MinMaxScaler(feature_range=(-1, 1)) \
                .fit(row["train"].drop(y_key, axis=1)) \
                .transform(row["test"].drop(y_key, axis=1))
            deep_preds = np.hstack([x.predict(test) for x in row["models"]["nn_deep"]])
            shallow_preds = np.hstack([x.predict(test) for x in row["models"]["nn_shallow"]])
            deep_model = row["models"]["nn_deep"][np.argsort(deep_preds)[len(row["models"]["nn_deep"]) // 2]]
            shallow_model = row["models"]["nn_shallow"][np.argsort(shallow_preds)[len(row["models"]["nn_shallow"]) // 2]]
            row["models"]["nn_deep"], row["models"]["nn_shallow"] = [deep_model], [shallow_model]
            ds.append(row)
        pickle.dump(ds, open("/FAST_STORE/IML_MODELS/%s/cpiaucsl_h%i_compact.bin" % (run_key, h), "wb"))
    exit(-1)


def read_multiple():

    a = "20230504_090454"
    b = "20230510_082421"
    c = "20241005_091334"
    # b = "20230131_161507"
    # c = "20230204_163839"

    hs = [1, 3, 6, 12]  # 12, 3, 1

    res = []
    for h in tqdm(hs):
        y = None
        y_hats = []

        for f in [c, ]:
            if os.path.exists("/FAST_STORE/IML_MODELS/%s/cpiaucsl_h%i.bin" % (f, h)):
                d = pickle.load(open("/FAST_STORE/IML_MODELS/%s/cpiaucsl_h%i.bin" % (f, h), "rb"))
                if y is None:
                    y = pd.DataFrame(pd.concat([x["test"]["y_h%i" % h] for x in d]))
                y_hats.append(pd.DataFrame(pd.concat([pd.concat([x[0] for x in z["y_hats"].values()], axis=1) for z in d])))

        y_hats = pd.concat([y] + y_hats, axis=1)
        y_hats.index += pd.DateOffset(months=h)

        print(y_hats.shape)

        y_hats["nn_comb"] = y_hats[["nn_deep_med_y_hat", "nn_shallow_med_y_hat"]].mean(axis=1)
        y_hats["nonlin_comb"] = y_hats[["rf_cv_y_hat", "nn_deep_med_y_hat", "nn_shallow_med_y_hat", "xgb_cv_y_hat"]].mean(axis=1)
        y_hats["lin_comb"] = y_hats[["enet_y_hat", "pca_y_hat"]].mean(axis=1)
        y_hats["all_comb"] = y_hats[["nonlin_comb", "lin_comb"]].mean(axis=1)

        import scipy

        def _cw_test(y, y_hat, y_hat_bench, lags):
            ldiff = (y - y_hat_bench) ** 2 - (y - y_hat) ** 2 + (y_hat_bench - y_hat) ** 2
            reg = sm.OLS(ldiff, np.ones(ldiff.shape[0])).fit(
                cov_type="HAC", cov_kwds={"maxlags": lags, "kernel": "bartlett"})
            p = 1 - scipy.stats.norm.cdf(reg.tvalues[0])
            return p

        def _dm_test(y, y_hat, y_hat_bench, lags):
            ldiff = (y - y_hat_bench) ** 2 - (y - y_hat) ** 2
            reg = sm.OLS(ldiff, np.ones(ldiff.shape[0])).fit(
                cov_type="HAC", cov_kwds={"maxlags": lags, "kernel": "bartlett"})
            p = 1 - scipy.stats.norm.cdf(reg.tvalues[0])
            return p

        rmse = np.sqrt((y_hats.drop("y_h%i" % h, axis=1).subtract(y_hats["y_h%i" % h], axis=0)**2).mean(axis=0))

        rmse_ratio = rmse / rmse["ar_y_hat"]

        cw_p = pd.Series({
            x: _cw_test(y_hats["y_h%i" % h], y_hats[x], y_hats["ar_y_hat"], lags=h)
            for x in y_hats.columns if x not in ["y_h%i" % h, "ar_y_hat"]
        })

        dm_p = pd.Series({
            x: _dm_test(y_hats["y_h%i" % h], y_hats[x], y_hats["ar_y_hat"], lags=h)
            for x in y_hats.columns if x not in ["y_h%i" % h, "ar_y_hat"]
        })

        res.append(pd.concat((
            rmse.rename("rmse"), rmse_ratio.rename("rmse_ratio"), cw_p.rename("cw_pval"), dm_p.rename("dm_pval")
        ), axis=1))

    pd.concat(res, keys=hs).to_excel("../../Results/Updated CPI 3/perf_raw.xlsx")
    pd.concat(res, keys=hs).to_pickle("../../Results/Updated CPI 3/perf_raw.pickle")

    breakpoint()


def split(f):
    o = f.replace(".bin", "")
    d = pickle.load(open(f, "rb"))
    os.makedirs(o, exist_ok=True)
    for i, row in enumerate(d):
        pickle.dump(row, open("%s/%i.bin" % (o, i), "wb"))


if __name__ == "__main__":
    # run_pred()
    # read_multiple()
    split("/FAST_STORE/IML_MODELS/20241005_091334/cpiaucsl_h1.bin")
    split("/FAST_STORE/IML_MODELS/20241005_091334/cpiaucsl_h3.bin")
    split("/FAST_STORE/IML_MODELS/20241005_091334/cpiaucsl_h6.bin")
    split("/FAST_STORE/IML_MODELS/20241005_091334/cpiaucsl_h12.bin")

    exit(-1)

    x = pickle.load(open("/media/sander/SSD/IML_MODELS/cpiaucsl_rf_h1.bin", "rb"))

    from sklearn import tree
    import graphviz
    import re

    dot_data = tree.export_graphviz(
        x[-1]["models"]["rf_cv"][0].estimators_[0],
        out_file=None,
        feature_names=x[0]["train"].columns[1:],
        # feature_names=np.arange(x[0]["train"].columns.shape[0]-1),
        filled=True,
        impurity=False,
        leaves_parallel=False,
        max_depth=3
    )
    cleaned_dot_data = re.sub(r'samples = \d+\\n', '', dot_data)
    # cleaned_dot_data = re.sub(r'value = ', '', cleaned_dot_data)

    graph = graphviz.Source(cleaned_dot_data, format="png")
    graph.render("decision_tree_graphivz")

    # x[0]["models"]["rf_cv"][0].estimators_[0]

    # read_multiple()
    exit(-1)
    # export("XX")
    # run_eval()
    # run_pred()

    # export_nns("20230202_114612")
    # read_multiple()

    split("/FAST_STORE/IML_MODELS/20230504_090454/cpiaucsl_h6.bin")
    split("/FAST_STORE/IML_MODELS/20230504_090454/cpiaucsl_h12.bin")

    exit(-1)

    # export_nns("20230111_172741")
    breakpoint()

    read_multiple()
    run_pred()
    # compile_results_excel("20210910_203308")

    # export("20230109_165607")
    # run_eval()
    # compile_dm("20210318_101742_final")
    # compile_multihorizon_losses("20210318_101742_final")
    # run_pred()
