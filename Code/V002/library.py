# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer, MinMaxScaler
from statsmodels.tsa.ar_model import ar_select_order
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression 
from scipy.stats import rankdata
from xlsxwriter.utility import xl_col_to_name
from sklearn.exceptions import ConvergenceWarning
import statsmodels.formula.api as smf
from sklearn import linear_model
from scipy.stats import norm, chi2
from scipy.stats import t
"""import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.regularizers import l1
from tensorflow.keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD"""
from numpy.random import seed
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


from sklearn.exceptions import ConvergenceWarning
ConvergenceWarning('ignore')

def _rf(train, test, y_key, h, model_config, jobs = 24):
    rs = np.random.RandomState(seed = 666)

    model = RandomForestRegressor(n_estimators=model_config["n_estimators"], n_jobs = jobs)

    # cv_strategy = KFold(20)
    cv_strategy = TimeSeriesSplit(n_splits=24, gap=h, test_size=1)

    model = GridSearchCV(estimator=model, param_grid=model_config["grid"], cv=cv_strategy, verbose=0, n_jobs = jobs, scoring="neg_mean_squared_error", refit=False)
    model.fit(train.drop(y_key, axis=1), train[y_key])

    best_model_mean = model.best_params_  # sklearn default, same as params[argmax(mean(cv_scores))]


    mean_model = RandomForestRegressor(n_jobs = jobs, n_estimators=model_config["n_estimators"], random_state=rs,
                                           **best_model_mean).fit(train.drop(y_key, axis=1), train[y_key])
    y_hat_mean = pd.Series(mean_model.predict(
            test.drop(y_key, axis=1)), index=test.index).rename("rf_tune_mean_y_hat")

    preds = y_hat_mean

    return preds


def _xgb(train, test, y_key, h, model_config, jobs = 24):
    rs = np.random.RandomState(seed = 666)

    model = XGBRegressor(n_estimators=model_config["n_estimators"], learning_rate=model_config["eta"],
                             n_jobs=jobs, random_state=rs)

    cv_strategy = TimeSeriesSplit(n_splits=12, gap=h, test_size=1)

    model = GridSearchCV(estimator=model, param_grid=model_config["grid"], cv=cv_strategy, verbose=0, n_jobs=jobs,
                             scoring="neg_mean_squared_error", refit=False)

    model.fit(train.drop(y_key, axis=1), train[y_key], verbose=False)
    best_model_mean = model.best_params_  # sklearn default, same as params[argmax(mean(cv_scores))]


    mean_model = XGBRegressor(n_estimators=model_config["n_estimators"], learning_rate=model_config["eta"],
                                  random_state=rs, n_jobs=jobs, **best_model_mean).fit(
            train.drop(y_key, axis=1), train[y_key], verbose=False)
    y_hat_mean = pd.Series(mean_model.predict(
            test.drop(y_key, axis=1)), index=test.index).rename("xgb_tune_mean_y_hat")

    preds = y_hat_mean


    return preds



def _nn(train, test, y_key,h , depth_type, model_config, jobs = 24):
    rs = np.random.RandomState(seed =1)

    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train.drop(y_key, axis=1))

    layer_sizes = [max(1, int(np.round((train.shape[1] - 1) ** x))) for x in
                        model_config["%s_layer_exponents" % depth_type]]

    preds = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if model_config["cv"]:
            model = MLPRegressor(hidden_layer_sizes=layer_sizes, batch_size=model_config["batch_size"],
                                      learning_rate="constant", learning_rate_init=model_config["learning_rate"],
                                      activation=model_config["activation"], solver="adam",
                                      max_iter=model_config["epochs"],
                                      n_iter_no_change=model_config["epochs"], verbose=False, random_state=rs)
            cv_strategy = TimeSeriesSplit(n_splits=12, gap=h, test_size=1)
            model = GridSearchCV(estimator=model, param_grid=model_config["grid"], cv=cv_strategy, verbose=0,
                                      n_jobs=jobs, scoring="neg_mean_squared_error", refit=False)
            model.fit(scaler.transform(train.drop(y_key, axis=1)), train[y_key])
            best_model_mean = model.best_params_  # sklearn default, same as params[argmax(mean(cv_scores))]
        print(model.best_params_)
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
                                          verbose=False, random_state=rs, alpha = 0.01)  
            model.fit(scaler.transform(train.drop(y_key, axis=1)), train[y_key])
            preds.append(model.predict(scaler.transform(test.drop(y_key, axis=1))))
            models.append(model)

    y_hat_med = pd.Series(np.median(preds), index=test.index).rename("nn_%s_med_y_hat" % depth_type)
    

    return y_hat_med





def _ln(name, train, test, y_key, h, model_config, jobs = 24):
       
    ytrain = train[y_key]    
    Xtrain = (train.drop(y_key, axis=1) - train.drop(y_key, axis=1).mean()) / train.drop(y_key, axis=1).std()
    Xpred = (test.drop(y_key, axis=1) - train.drop(y_key, axis=1).mean()) / train.drop(y_key, axis=1).std()
    
    rs = np.random.RandomState(seed = 666)
    ts_cv = TimeSeriesSplit(n_splits = 32, gap=(h), test_size=3)
    models = {
        "Ridge": RidgeCV(alphas=model_config["alphas"], fit_intercept=True, cv=ts_cv, scoring="neg_mean_squared_error"),
        "Lasso": LassoCV(alphas=model_config["alphas"], fit_intercept=True, n_jobs=jobs, cv=ts_cv, random_state=rs),
        "ENet": ElasticNetCV(alphas=model_config["alphas"], l1_ratio=model_config["l1_ratios"], fit_intercept=True, n_jobs=jobs, cv=ts_cv, random_state=rs),
    }
    models[name].fit(Xtrain, ytrain)
    pred = pd.Series(models[name].predict(Xpred), index=test.index).rename("%s_y_hat" % name)
    
        
    return pred



def _enetselection(train, test, y_key, h, jobs):
    
    
    train_std = (train - train.mean()) / train.std()
    
    model_config ={"alphas": np.linspace(0.01, 10, 100).tolist(), "l1_ratios": [0.1, 0.5, 0.9]}
    name = 'ENet'
  
        
    rs = np.random.RandomState(seed = 666)
    ts_cv = TimeSeriesSplit(n_splits = 32, gap = (h), test_size = 3)
    models = {
        "Ridge": RidgeCV(alphas=model_config["alphas"], fit_intercept=True, cv=ts_cv, scoring="neg_mean_squared_error"),
        "Lasso": LassoCV(alphas=model_config["alphas"], fit_intercept=True, n_jobs=jobs, cv=ts_cv, random_state=rs),
        "ENet": ElasticNetCV(alphas=model_config["alphas"], l1_ratio=model_config["l1_ratios"], fit_intercept=True, n_jobs=jobs, cv=ts_cv, random_state=rs),
    }
    models[name].fit(train_std.drop(y_key, axis=1), train_std[y_key])

    print(np.count_nonzero(models[name].coef_))
            
    idx = np.nonzero(models[name].coef_)[0]

    idx = np.append(np.array([0]), idx+1)      
    
    train = train.iloc[:,idx]
    
    test = test.iloc[:,idx]
    
    return [train, test]


def _targetedpredictors(train, test, y_key, noPred):
#   This function conducts variable selection via Elastic Net, targeting the number of noPred predictors. 
#   The funciton returns an already normalized matrix. 

    # The parameter l1_ratio corresponds to alpha in the glmnet R package while alpha corresponds to the lambda parameter in glmnet
    # Specifically, l1_ratio = 1 is the lasso penalty.

    
    train_std = (train - train.mean()) / train.std()
    
    # Search by bi-section method
    minAlpha = 0.0001
    maxAlpha = 100000
    
    noPredEval = 0
    alphaEval  = (maxAlpha - minAlpha)/2
    dist       = alphaEval
    while noPredEval != noPred:
        ENet      = ElasticNet(alpha = alphaEval, l1_ratio = 0.5, fit_intercept = True, tol = 1e-5, selection = 'random', max_iter=5000)
        ENet.fit(X = train_std.drop(y_key, axis=1), y = train_std[y_key])
        noPredEval = np.count_nonzero(ENet.coef_)
        # Bi-section step 
        alphaEval = alphaEval + np.sign(noPredEval - noPred)*(dist/2)
        dist = dist/2
        
    # Get indx of the predictors from the last run (which is the targeted ones)
       
    idx = np.nonzero(ENet.coef_)[0]

    idx = np.append(np.array([0]), idx+1)      
    
    train = train.iloc[:,idx]
    
    test = test.iloc[:,idx]
    
    return [train, test]   



def _enetpls(train, test, y_key, h, model_config, jobs):

    
    [train, test] = _enetselection(train, test, y_key, h, jobs)

    pls = PLSRegression(n_components = model_config["components"])
    
    if len(train.columns) < 2:
        yhat = np.array(train[y_key].mean())
    else:
         
        pls.fit(train.drop(y_key, axis=1), train[y_key])
      
        yhat = pls.predict(test.drop(y_key, axis=1))
       
    return yhat



def _targetedpls(train, test, y_key, h, model_config, noPred):

    
    [train, test] = _targetedpredictors(train, test, y_key, noPred)

    pls = PLSRegression(n_components = model_config["components"])
    
    if len(train.columns) < 2:
        yhat = np.array(train[y_key].mean())
    else:
         
        pls.fit(train.drop(y_key, axis=1), train[y_key])
      
        yhat = pls.predict(test.drop(y_key, axis=1))
       
    return yhat


def _enetpca(train, test, y_key, h, model_config, jobs):
    
    [train, test] = _enetselection(train, test, y_key, h, jobs)
    
    if len(train.columns) < 2:
        yhat = np.array(train[y_key].mean())
    else:
    
        ytrain = train[y_key]    
        Xtrain = (train.drop(y_key, axis=1) - train.drop(y_key, axis=1).mean()) / train.drop(y_key, axis=1).std()
        Xpred = (test.drop(y_key, axis=1) - train.drop(y_key, axis=1).mean()) / train.drop(y_key, axis=1).std()

        pca = PCA(n_components = model_config["components"])

        pcafit = pca.fit(Xtrain)
    
        Xtrain_pca = pcafit.transform(Xtrain)
    
        Xpred_pca = pcafit.transform(Xpred)
       
        reg = LinearRegression().fit(Xtrain_pca, ytrain)
    
        yhat = reg.predict(Xpred_pca)
    
    return yhat



def _pca(train, test, y_key, h, model_config):
    ytrain = train[y_key]    
    Xtrain = (train.drop(y_key, axis=1) - train.drop(y_key, axis=1).mean()) / train.drop(y_key, axis=1).std()
    Xpred = (test.drop(y_key, axis=1) - train.drop(y_key, axis=1).mean()) / train.drop(y_key, axis=1).std()

    pca = PCA(n_components = model_config["components"])

    pcafit = pca.fit(Xtrain)
    
    Xtrain_pca = pcafit.transform(Xtrain)
    
    Xpred_pca = pcafit.transform(Xpred)
       
    reg = LinearRegression().fit(Xtrain_pca, ytrain)
    
    yhat = reg.predict(Xpred_pca)
    
    return yhat
    

def _combination(train, test, y_key, h):
    
     
    ytrain = train[y_key]
    Xtrain = train.drop(y_key, axis=1)
    Xpred = test.drop(y_key, axis=1)

    forecasts = np.zeros(len(Xtrain.columns))     

    for i, var in enumerate(Xtrain):
        reg = LinearRegression().fit(Xtrain[var].values.reshape(-1, 1), ytrain)
        f = reg.predict(Xpred[var].values.reshape(-1, 1))
        forecasts[i] = f
        
    yhat = np.mean(forecasts)

    return yhat


def _enetcombination(train, test, y_key, h, jobs):
    
    [train, test] = _enetselection(train, test, y_key, h, jobs)
    
    if len(train.columns) < 2:
        yhat = np.array(train[y_key].mean())
    else:
    
        ytrain = train[y_key]
        Xtrain = train.drop(y_key, axis=1)
        Xpred = test.drop(y_key, axis=1)

        forecasts = np.zeros(len(Xtrain.columns))     

        for i, var in enumerate(Xtrain):
            reg = LinearRegression().fit(Xtrain[var].values.reshape(-1, 1), ytrain)
            f = reg.predict(Xpred[var].values.reshape(-1, 1))
            forecasts[i] = f
        
        yhat = np.mean(forecasts)

    return yhat



def _targetedcombination(train, test, y_key, h, noPred):
    
    [train, test] = _targetedpredictors(train, test, y_key, noPred)
    
    if len(train.columns) < 2:
        yhat = np.array(train[y_key].mean())
    else:
    
        ytrain = train[y_key]
        Xtrain = train.drop(y_key, axis=1)
        Xpred = test.drop(y_key, axis=1)

        forecasts = np.zeros(len(Xtrain.columns))     

        for i, var in enumerate(Xtrain):
            reg = LinearRegression().fit(Xtrain[var].values.reshape(-1, 1), ytrain)
            f = reg.predict(Xpred[var].values.reshape(-1, 1))
            forecasts[i] = f
        
        yhat = np.mean(forecasts)

    return yhat



def _enetrf(train, test, y_key, h, model_config, jobs = 10):
    
    [train, test] = _enetselection(train, test, y_key, h, jobs)
    
    
    if len(train.columns) < 2:
        yhat = np.array(train[y_key].mean())
    else:
    
        rs = np.random.RandomState(seed = 666)

        model = RandomForestRegressor(n_estimators=model_config["n_estimators"], n_jobs = jobs)

        # cv_strategy = KFold(20)
        cv_strategy = TimeSeriesSplit(n_splits = 24, gap=h, test_size=1)

        model = GridSearchCV(estimator=model, param_grid=model_config["grid"], cv=cv_strategy, verbose=0, n_jobs = jobs, scoring="neg_mean_squared_error", refit=False)
        model.fit(train.drop(y_key, axis=1), train[y_key])

        best_model_mean = model.best_params_  # sklearn default, same as params[argmax(mean(cv_scores))]


        mean_model = RandomForestRegressor(n_jobs = jobs, n_estimators=model_config["n_estimators"], random_state=rs,
                                           **best_model_mean).fit(train.drop(y_key, axis=1), train[y_key])
        y_hat_mean = pd.Series(mean_model.predict(
                test.drop(y_key, axis=1)), index=test.index).rename("rf_tune_mean_y_hat")

        yhat = y_hat_mean

    return yhat





def _pls(train, test, y_key, h, model_config):

    
    pls = PLSRegression(n_components = model_config["components"])
    
    pls.fit(train.drop(y_key, axis=1), train[y_key])
      
    yhat = pls.predict(test.drop(y_key, axis=1))
    
    
    return yhat



def _meanfore(train, y_key):
    
    
    yhat = np.array(train[y_key].mean())
    
    return yhat



def _R2OoS(errorsmodel,errorsbenchmark):
    # Campbell Thomson Out of Sample R2. Errors benchmark should be the errors of the historical mean model
    R2 = 1 - np.sum(np.array(errorsmodel.dropna())**2) / np.sum(np.array(errorsbenchmark.dropna())**2)
    return R2


def _RMSE(errorsmodel,errorsbenchmark):
    # RMSE ratio
    RMSE = np.sqrt(np.mean(np.array(errorsmodel.dropna())**2)) / np.sqrt(np.mean(np.array(errorsbenchmark.dropna())**2))
    return RMSE

def _CSSED(errorsmodel,errorsbenchmark):
    # CSSED 
    cumsum = np.nancumsum(np.array(errorsbenchmark**2) - np.array(errorsmodel**2))
    CSSED = pd.DataFrame(cumsum,index=errorsmodel.index.values)
    return CSSED


def _DieboldMarianoTest(errorsmodel,errorsbenchmark, h=1):
    # Diebold-Mariano test with Newey-West std errors (Bartlett kernel and h-1 lags)

    lossDiff   = np.array(errorsbenchmark.dropna())**2 - np.array(errorsmodel.dropna())**2
    maxLag     = h#math.floor(np.power(len(lossDiff),1/4))

    df         = pd.DataFrame({'y':lossDiff})
    reg        = smf.ols('y~1', data=df).fit(cov_type='HAC',cov_kwds={'maxlags':np.max(maxLag),'kernel':'bartlett'})

    # Output
    params     = reg.params
    testStat   = reg.tvalues
    pValue     = 1-norm.cdf(testStat)# one-sided test
    
    return pValue    
    #return [testStat, pValue, params]


def _ClarkWestTest(errorsmodel,errorsbenchmark):
    # Clark-West test with Newey-West std errors (Bartlett kernel and h-1 lags)

    # get correction: y_bench - y_true - (y_model - y_true) = y_bench - y_model
    extraTerm  = np.array(errorsbenchmark.dropna()) - np.array(errorsmodel.dropna())
    lossDiff   = np.array(errorsbenchmark.dropna())**2 - (np.array(errorsmodel.dropna())**2 -  extraTerm**2)
    maxLag     = 12#math.floor(np.power(len(lossDiff),1/4))



    df         = pd.DataFrame({'y':lossDiff})
    reg        = smf.ols('y~1', data=df).fit(cov_type='HAC',cov_kwds={'maxlags':np.max(maxLag),'kernel':'bartlett'})

    # Output
    params     = reg.params
    testStat   = reg.tvalues
    pValue     = 1-norm.cdf(testStat)# one-sided test
        
    #return [testStat, pValue, params]

    return pValue
    
