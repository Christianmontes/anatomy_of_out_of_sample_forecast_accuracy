import pandas as pd
import numpy as np
import shap
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from shap.maskers import Independent
import os
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend

# Root of the fitted-model archives (the contents of Models/20241005_091334.7z).
# Originally /FAST_STORE/IML_MODELS on the estimation box; the default matches the
# repo layout relative to Code/V002. Override with the IML_MODELS_DIR environment variable.
MODELS_DIR = os.environ.get("IML_MODELS_DIR", "../../Models")


def ishapley(model_key, models, background, explain, y_key):

    if model_key == "pca":

        ((pca_model, x_mean, x_std), model) = models
        n_components = model.coef_.shape[0]
        explain_transformed = pca_model.transform((explain.drop(y_key, axis=1) - x_mean) / x_std)[:, :n_components]
        shap_values_pca = np.stack([
            ((explain.drop(y_key, axis=1).to_numpy() - x_mean) / x_std) *
            pca_model.components_[i, :] / explain.drop(y_key, axis=1).shape[1]
            for i in range(n_components)
        ]).T
        shap_values_ols = explain_transformed * model.coef_
        shap_factors = shap_values_pca.sum(axis=0) / shap_values_ols
        shap_values = (shap_values_pca / shap_factors).sum(axis=-1).T
        shap_values = pd.DataFrame(shap_values, columns=explain.drop(y_key, axis=1).columns, index=explain.index)
        base_value = model.intercept_

    elif model_key == "enet" or model_key == "ridge" or model_key == "lasso":

        ((x_mean, x_std), models) = models
        background_norm = (background.drop(y_key, axis=1) - x_mean) / x_std
        explain_norm = (explain.drop(y_key, axis=1) - x_mean) / x_std
        base_value = (background_norm.to_numpy().mean(axis=0) * models.coef_).sum() + models.intercept_
        shap_values = (explain_norm - background_norm.mean()) * models.coef_

    elif model_key == "nn_deep" or model_key == "nn_shallow":

        model = models[0]
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(background.drop(y_key, axis=1))
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.keras.backend.set_floatx("float64")
        depth_type = model_key.replace("nn_", "")
        k = background.shape[1] - 1
        model_config = {"shallow_layer_exponents": [1 / 2], "deep_layer_exponents": [3 / 4, 2 / 4, 1 / 4]}
        layer_sizes = [
            max(1, int(np.round(k ** x)))
            for x in model_config["%s_layer_exponents" % depth_type]
        ]
        model_clone = Sequential()
        model_clone.add(Dense(layer_sizes[0], input_shape=(k,), activation="relu"))
        for layer_size in layer_sizes[1:]:
            model_clone.add(Dense(layer_size, activation="relu"))
        model_clone.add(Dense(1, activation="linear"))
        model_clone.compile(loss="mean_squared_error", optimizer=Adam())
        weights = []
        for i in range(model.n_layers_ - 1):
            weights.append(model.coefs_[i]), weights.append(model.intercepts_[i])
        model_clone.set_weights(weights)
        explainer = shap.DeepExplainer(
            (model_clone.layers[0].input, model_clone.layers[-1].output),
            scaler.transform(background.drop(y_key, axis=1))
        )
        shap_values = explainer.shap_values(scaler.transform(explain.drop(y_key, axis=1)))[0]
        shap_values = pd.DataFrame(shap_values, index=explain.index, columns=explain.drop(y_key, axis=1).columns)
        base_value = explainer.expected_value.numpy()

    elif model_key == "rf_cv" or model_key == "xgb_cv":

        model = models[0]
        explainer = shap.TreeExplainer(model, data=Independent(
            background.drop(y_key, axis=1), max_samples=background.shape[0]))
        shap_values = explainer.shap_values(explain.drop(y_key, axis=1), approximate=False, check_additivity=False)
        shap_values = pd.DataFrame(shap_values, index=explain.index, columns=explain.drop(y_key, axis=1).columns)
        base_value = explainer.expected_value

    shap_values = pd.concat((
        pd.Series(np.repeat(base_value, shap_values.shape[0]), index=shap_values.index).rename("base_value"),
        shap_values
    ), axis=1)

    return shap_values


def estimate(h):
    ps = {
        "nn_deep": os.path.join(MODELS_DIR, "20241005_091334", "cpiaucsl_h%i" % h),
        "nn_shallow": os.path.join(MODELS_DIR, "20241005_091334", "cpiaucsl_h%i" % h),
        "rf_cv": os.path.join(MODELS_DIR, "20241005_091334", "cpiaucsl_h%i" % h),
        "xgb_cv": os.path.join(MODELS_DIR, "20241005_091334", "cpiaucsl_h%i" % h),
        "enet": os.path.join(MODELS_DIR, "20241005_091334", "cpiaucsl_h%i" % h),
        "pca": os.path.join(MODELS_DIR, "20241005_091334", "cpiaucsl_h%i" % h),
    }

    def _estimate(_f, _model_key, _h):
        d = pickle.load(open(_f, "rb"))
        ishap = ishapley(_model_key, d["models"][_model_key], d["train"], d["train"], "y_h%i" % _h)
        return _model_key, d["test"].index[0], ishap

    payloads = []
    for model_key in ps.keys():
        for period in range(416):
            f = "%s/%i.bin" % (ps[model_key], period)
            payloads.append(delayed(_estimate)(f, model_key, h))

    with parallel_backend("loky", inner_max_num_threads=1):
        raw_res = Parallel(n_jobs=32, verbose=True)(payloads)

    res = {x: {} for x in ps.keys()}
    for model_key, period, data in raw_res:
        res[model_key][period] = data
    res = pd.DataFrame(res)

    res.to_pickle("../../Results/Updated CPI 3/ishapley_h%i_upd.bin" % h)
    # breakpoint()


if __name__ == "__main__":
    estimate(h=12)
    estimate(h=6)
    estimate(h=3)
    estimate(h=1)
