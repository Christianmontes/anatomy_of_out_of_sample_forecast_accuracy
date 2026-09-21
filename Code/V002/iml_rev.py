import os
import pickle
import numpy as np
import pandas as pd
from anatomy import *
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import string
import matplotlib.ticker as mtick
from typing import Union

# Root of the fitted-model archives (the contents of Models/20241005_091334.7z).
# Originally /FAST_STORE/IML_MODELS on the estimation box; the default matches the
# repo layout relative to Code/V002, same convention as the ../../Results paths.
# Override with the IML_MODELS_DIR environment variable if the models live elsewhere.
MODELS_DIR = os.environ.get("IML_MODELS_DIR", "../../Models")


def rankdiffs2_w(h, a, n_mc_sims=1000000, override=False):
    if not override and os.path.exists("../../Results/Updated CPI 3/MAS/rankdiffs2_w_h%i_mc%i_a%.2f.pickle" % (h, n_mc_sims, a)):
        return pickle.load(open("../../Results/Updated CPI 3/MAS/rankdiffs2_w_h%i_mc%i_a%.2f.pickle" % (h, n_mc_sims, a), "rb"))

    (pbsv_rmse, _), (oshapley, ishapley), oshapley_vi, ishapley_vi = pickle.load(
        open("../../Results/Updated CPI 3/shapleys_h%i_upd.bin" % h, "rb")
    )

    def expected_msdr_w(k, w, r):
        import math
        def _s(n):
            return n * (n + 1) * (2 * n + 1) / 6
        def _c(n, b):
            return math.factorial(n) / (math.factorial(b) * math.factorial(n - b))
        return 1/k * (sum(w*(r**2)) + sum([_c(k, a) * (0.5 ** k) * (_s(a) + _s(k - a)) for a in range(0, k + 1)]))

    def rnd_diffs2_w(alpha, k, r, w, num_trials=n_mc_sims):
        A = np.arange(1, k + 1)
        mean_squared_diffs = np.empty(num_trials)
        b = np.random.binomial(k, alpha, num_trials)
        perms_a = np.argsort(np.random.uniform(-1, 1, (num_trials, k)), axis=1)
        perms_b = np.argsort(np.random.uniform(-1, 1, (num_trials, k)), axis=1)
        for i in range(num_trials):
            num_pos = b[i]
            num_neg = k - num_pos
            B = np.hstack((A[:num_pos], -A[:num_neg]))[perms_b[i]]
            mean_squared_diffs[i] = np.mean(w[perms_a[i]] * (r[perms_a[i]] - B) ** 2)
        return mean_squared_diffs

    def signed_ranking(X):
        from scipy.stats import rankdata
        R = np.zeros_like(X)
        R[X < 0] = rankdata(-X[X < 0])
        R[X >= 0] = -rankdata(X[X >= 0])
        return R

    unused_enet = ishapley_vi.loc["enet"][np.isclose(ishapley_vi.loc["enet"], 0)].index

    ishapley_w = ishapley_vi.drop("base_contribution", axis=1).divide(
        ishapley_vi.drop("base_contribution", axis=1).mean(axis=1), axis=0)

    ishapley_enet_w = ishapley_vi.loc["enet"].drop("base_contribution").drop(unused_enet)
    ishapley_enet_w /= ishapley_enet_w.mean()

    ishapley_r = ishapley_vi.drop("base_contribution", axis=1).rank(axis=1, ascending=True)
    pbsv_rmse_r = pd.concat([pd.Series(signed_ranking(pbsv_rmse.loc[x].drop("base_contribution")), index=pbsv_rmse.drop("base_contribution", axis=1).columns).rename(x) for x in pbsv_rmse.index], axis=1).T

    ishapley_enet_r = ishapley_vi.loc["enet"].drop("base_contribution").drop(unused_enet).rank(ascending=True)
    pbsv_rmse_enet_r = pd.Series(signed_ranking(pbsv_rmse.loc["enet"].drop("base_contribution").drop(unused_enet)), index=pbsv_rmse.loc["enet"].drop("base_contribution").drop(unused_enet).index)

    p_enet = pbsv_rmse_enet_r.shape[0]
    rnd_msdr_enet = expected_msdr_w(p_enet, r=ishapley_enet_r.to_numpy(), w=ishapley_enet_w.to_numpy())
    rnd_msdr_h0_enet = rnd_diffs2_w(alpha=a, k=p_enet, r=ishapley_enet_r.to_numpy(), w=ishapley_enet_w.to_numpy())
    rnd_mas_h0_enet = 1 - rnd_msdr_h0_enet / rnd_msdr_enet

    p = pbsv_rmse_r.shape[1]
    rnd_msdr = {x: expected_msdr_w(p, r=ishapley_r.loc[x].to_numpy(), w=ishapley_w.loc[x].to_numpy()) for x in ishapley_r.index}
    rnd_msdr_h0 = {x: rnd_diffs2_w(alpha=a, k=p, r=ishapley_r.loc[x].to_numpy(), w=ishapley_w.loc[x].to_numpy()) for x in tqdm(ishapley_r.index)}
    rnd_mas_h0 = {x: 1 - rnd_msdr_h0[x] / rnd_msdr[x] for x in ishapley_r.index}

    ishapley_vs_pbsv_mas_enet = (ishapley_enet_w * (ishapley_enet_r - pbsv_rmse_enet_r) ** 2).mean()
    ishapley_vs_pbsv_mas_rnd_norm_enet = 1 - ishapley_vs_pbsv_mas_enet / rnd_msdr_enet
    ishapley_vs_pbsv_mas_rnd_norm_enet_p = (rnd_mas_h0_enet >= ishapley_vs_pbsv_mas_rnd_norm_enet).mean()

    ishapley_vs_pbsv_mas = (ishapley_w * (ishapley_r - pbsv_rmse_r) ** 2).mean(axis=1).dropna()
    ishapley_vs_pbsv_mas_rnd_norm = (1 - ishapley_vs_pbsv_mas / pd.Series(rnd_msdr)).dropna()
    ishapley_vs_pbsv_mas_rnd_norm_p = pd.Series({x: (rnd_mas_h0[x] >= ishapley_vs_pbsv_mas_rnd_norm[x]).mean() for x in ishapley_vs_pbsv_mas.index})

    ishapley_vs_pbsv_mas_rnd_norm["enet"] = ishapley_vs_pbsv_mas_rnd_norm_enet
    ishapley_vs_pbsv_mas_rnd_norm_p["enet"] = ishapley_vs_pbsv_mas_rnd_norm_enet_p

    res_dict = {
        "ishapley_vs_pbsv_mas_rnd_norm": ishapley_vs_pbsv_mas_rnd_norm,
        "ishapley_vs_pbsv_mas_rnd_norm_p": ishapley_vs_pbsv_mas_rnd_norm_p,
    }

    pickle.dump(res_dict, open("../../Results/Updated CPI 3/MAS/rankdiffs2_w_h%i_mc%i_a%.2f.pickle" % (h, n_mc_sims, a), "wb"))
    return res_dict

def rankdiffs2(h, a, n_mc_sims=1000000, override=False):
    if not override and os.path.exists("_rankdiffs2_h%i_mc%i_a%.2f.pickle" % (h, n_mc_sims, a)):
        return pickle.load(open("_rankdiffs2_h%i_mc%i_a%.2f.pickle" % (h, n_mc_sims, a), "rb"))

    (pbsv_rmse, _), (oshapley, ishapley), oshapley_vi, ishapley_vi = pickle.load(
        open("../../Results/Updated CPI 2/shapleys_h%i_upd.bin" % h, "rb")
    )

    def expected_msdr(k):
        import math
        def _s(n):
            return n * (n + 1) * (2 * n + 1) / 6
        def _c(n, b):
            return math.factorial(n) / (math.factorial(b) * math.factorial(n - b))
        return 1/k * (_s(k) + sum([_c(k, a) * (0.5 ** k) * (_s(a) + _s(k - a)) for a in range(0, k + 1)]))

    def expected_msdr_w(k, w, r):
        import math
        def _s(n):
            return n * (n + 1) * (2 * n + 1) / 6
        def _c(n, b):
            return math.factorial(n) / (math.factorial(b) * math.factorial(n - b))
        return 1/k * (sum(w*(r**2)) + sum([_c(k, a) * (0.5 ** k) * (_s(a) + _s(k - a)) for a in range(0, k + 1)]))

    def rnd_diffs3(alpha, k, num_trials=n_mc_sims):
        # this is at alpha=0.5 NOT equivalent to sampling from Unif(-1, 1) and signed-ranking it
        # because the number of positive ranks is not sampled but fixed, so this is a bad null
        A = np.arange(1, k + 1)
        mean_squared_diffs = np.empty(num_trials)
        for i in range(num_trials):
            num_pos = int(k*alpha)
            num_neg = k - num_pos
            B = np.hstack((A[:num_pos], -A[:num_neg]))[np.random.permutation(k)]
            mean_squared_diffs[i] = np.mean((A - B) ** 2)
        return (mean_squared_diffs)

    def rnd_diffs2_05(k, num_trials=n_mc_sims):
        # alpha=0.5 due to uniform dist
        mean_squared_diffs = []
        A = np.arange(1, p + 1)
        for _ in tqdm(range(n_mc_sims)):
            mean_squared_diffs.append(((A - signed_ranking(np.random.uniform(-1, 1, p))) ** 2).mean())
        return np.array(mean_squared_diffs)

    def rnd_diffs2(alpha, k, num_trials=n_mc_sims):
        # this is at alpha=0.5 equivalent to sampling from Unif(-1, 1) and signed-ranking it
        A = np.arange(1, k + 1)
        mean_squared_diffs = np.empty(num_trials)
        b = np.random.binomial(k, alpha, num_trials)
        perms_a = np.argsort(np.random.uniform(-1, 1, (num_trials, k)), axis=1)
        perms_b = np.argsort(np.random.uniform(-1, 1, (num_trials, k)), axis=1)
        for i in range(num_trials):
            num_pos = b[i]
            num_neg = k - num_pos
            B = np.hstack((A[:num_pos], -A[:num_neg]))[perms_b[i]]
            mean_squared_diffs[i] = np.mean((A[perms_a[i]] - B) ** 2)
        return mean_squared_diffs

    def rnd_diffs2_w(alpha, k, r, w, num_trials=n_mc_sims):
        # this is at alpha=0.5 equivalent to sampling from Unif(-1, 1) and signed-ranking it
        A = np.arange(1, k + 1)
        mean_squared_diffs = np.empty(num_trials)
        b = np.random.binomial(k, alpha, num_trials)
        perms_a = np.argsort(np.random.uniform(-1, 1, (num_trials, k)), axis=1)
        perms_b = np.argsort(np.random.uniform(-1, 1, (num_trials, k)), axis=1)
        for i in range(num_trials):
            num_pos = b[i]
            num_neg = k - num_pos
            B = np.hstack((A[:num_pos], -A[:num_neg]))[perms_b[i]]
            mean_squared_diffs[i] = np.mean(w[perms_a[i]] * (r[perms_a[i]] - B) ** 2)
        return mean_squared_diffs

    def rnd_diffs(alpha, k, num_trials=n_mc_sims):
        A = np.arange(1, k + 1)
        mean_squared_diffs = np.empty(num_trials)
        for i in range(num_trials):
            num_pos = np.random.binomial(k, alpha)
            num_neg = k - num_pos
            pos_inds = np.random.permutation(k)[:num_pos]
            neg_inds = np.random.permutation(k)[:num_neg]
            B = np.hstack((A[pos_inds], -A[neg_inds]))[np.random.permutation(k)]
            mean_squared_diffs[i] = np.mean((A - B) ** 2)
        return mean_squared_diffs

    def signed_ranking(X):
        from scipy.stats import rankdata
        R = np.zeros_like(X)
        R[X < 0] = rankdata(-X[X < 0])
        R[X >= 0] = -rankdata(X[X >= 0])
        return R

    unused_enet = ishapley_vi.loc["enet"][np.isclose(ishapley_vi.loc["enet"], 0)].index

    ishapley_w = ishapley_vi.drop("base_contribution", axis=1).divide(
        ishapley_vi.drop("base_contribution", axis=1).mean(axis=1), axis=0)

    oshapley_w = oshapley_vi.drop("base_contribution", axis=1).divide(
        oshapley_vi.drop("base_contribution", axis=1).mean(axis=1), axis=0)

    ishapley_enet_w = ishapley_vi.loc["enet"].drop("base_contribution").drop(unused_enet)
    ishapley_enet_w /= ishapley_enet_w.mean()

    oshapley_enet_w = oshapley_vi.loc["enet"].drop("base_contribution").drop(unused_enet)
    oshapley_enet_w /= oshapley_enet_w.mean()

    ishapley_r = ishapley_vi.drop("base_contribution", axis=1).rank(axis=1, ascending=True)
    oshapley_r = oshapley_vi.drop("base_contribution", axis=1).rank(axis=1, ascending=True)
    pbsv_rmse_r = pd.concat([pd.Series(signed_ranking(pbsv_rmse.loc[x].drop("base_contribution")), index=pbsv_rmse.drop("base_contribution", axis=1).columns).rename(x) for x in pbsv_rmse.index], axis=1).T

    ishapley_enet_r = ishapley_vi.loc["enet"].drop("base_contribution").drop(unused_enet).rank(ascending=True)
    oshapley_enet_r = oshapley_vi.loc["enet"].drop("base_contribution").drop(unused_enet).rank(ascending=True)
    pbsv_rmse_enet_r = pd.Series(signed_ranking(pbsv_rmse.loc["enet"].drop("base_contribution").drop(unused_enet)), index=pbsv_rmse.loc["enet"].drop("base_contribution").drop(unused_enet).index)

    p = pbsv_rmse_r.shape[1]

    # rnd_msdr = (2 / 3 * p * (p + 1) * (2 * p + 1)) / p / 2

    rnd_msdr = expected_msdr_w(k=4, r=np.array([1, 4, 2, 3]), w=np.array([0.5, 1.5, 1, 1]))
    # xx = rnd_diffs2_w(alpha=0.5, k=4, r=np.array([1, 4, 2, 3]), w=np.array([0.5, 1.5, 1, 1]))
    # rnd_msdr_h0_w = [rnd_diffs2_w(alpha=0.5, k=p, r=ishapley_r.loc["xgb_cv"].to_numpy(), w=ishapley_w.loc["xgb_cv"].to_numpy()) for _ in tqdm(range(10))]
    breakpoint()

    rnd_msdr = expected_msdr(p)
    rnd_msdr_h0 = 1 - rnd_diffs2(alpha=a, k=p) / rnd_msdr
    # rnd_msdr_h0 = rnd_diffs2_w(alpha=a, k=p, w=ishapley_w.loc["nn_comb"].to_numpy())
    breakpoint()

    p_enet = pbsv_rmse_enet_r.shape[0]
    # rnd_msdr_enet = (2 / 3 * p_enet * (p_enet + 1) * (2 * p_enet + 1)) / p_enet / 2
    rnd_msdr_enet = expected_msdr(p_enet)
    rnd_msdr_h0_enet = 1 - rnd_diffs2(alpha=a, k=p_enet) / rnd_msdr_enet

    def two_sided_pval(mas, mas_h0):
        if type(mas) == pd.Series:
            return pd.Series([np.mean(np.abs(mas_h0) >= np.abs(x)) for x in mas], index=mas.index)
        else:
            return np.mean(np.abs(mas_h0) >= np.abs(mas))

    oshapley_vs_pbsv_mas_enet = ((oshapley_enet_r - pbsv_rmse_enet_r) ** 2).mean()
    oshapley_vs_pbsv_mas_rnd_norm_enet = 1 - oshapley_vs_pbsv_mas_enet / rnd_msdr_enet
    oshapley_vs_pbsv_mas_rnd_norm_enet_p = two_sided_pval(oshapley_vs_pbsv_mas_rnd_norm_enet, rnd_msdr_h0_enet)

    ishapley_vs_pbsv_mas_enet = ((ishapley_enet_r - pbsv_rmse_enet_r) ** 2).mean()
    ishapley_vs_pbsv_mas_rnd_norm_enet = 1 - ishapley_vs_pbsv_mas_enet / rnd_msdr_enet
    ishapley_vs_pbsv_mas_rnd_norm_enet_p = two_sided_pval(ishapley_vs_pbsv_mas_rnd_norm_enet, rnd_msdr_h0_enet)

    oshapley_vs_pbsv_mas = ((oshapley_r - pbsv_rmse_r) ** 2).mean(axis=1).dropna()
    oshapley_vs_pbsv_mas_rnd_norm = 1 - oshapley_vs_pbsv_mas / rnd_msdr
    oshapley_vs_pbsv_mas_rnd_norm_p = pd.Series(two_sided_pval(oshapley_vs_pbsv_mas_rnd_norm, rnd_msdr_h0), index=oshapley_vs_pbsv_mas.index)

    ishapley_vs_pbsv_mas = ((ishapley_r - pbsv_rmse_r) ** 2).mean(axis=1).dropna()
    ishapley_vs_pbsv_mas_rnd_norm = 1 - ishapley_vs_pbsv_mas / rnd_msdr
    ishapley_vs_pbsv_mas_rnd_norm_p = pd.Series(two_sided_pval(ishapley_vs_pbsv_mas_rnd_norm, rnd_msdr_h0), index=ishapley_vs_pbsv_mas.index)

    oshapley_vs_pbsv_mas_rnd_norm["enet"] = oshapley_vs_pbsv_mas_rnd_norm_enet
    ishapley_vs_pbsv_mas_rnd_norm["enet"] = ishapley_vs_pbsv_mas_rnd_norm_enet

    oshapley_vs_pbsv_mas_rnd_norm_p["enet"] = oshapley_vs_pbsv_mas_rnd_norm_enet_p
    ishapley_vs_pbsv_mas_rnd_norm_p["enet"] = ishapley_vs_pbsv_mas_rnd_norm_enet_p

    res_dict = {
        "oshapley_vs_pbsv_mas_rnd_norm": oshapley_vs_pbsv_mas_rnd_norm,
        "ishapley_vs_pbsv_mas_rnd_norm": ishapley_vs_pbsv_mas_rnd_norm,
        "oshapley_vs_pbsv_mas_rnd_norm_p": oshapley_vs_pbsv_mas_rnd_norm_p,
        "ishapley_vs_pbsv_mas_rnd_norm_p": ishapley_vs_pbsv_mas_rnd_norm_p,
    }

    pickle.dump(res_dict, open("_rankdiffs2_h%i_mc%i_a%.2f.pickle" % (h, n_mc_sims, a), "wb"))
    return res_dict


def rankdiffs(h, n_mc_sims=1000000, override=False):
    if not override and os.path.exists("_rankdiffs_h%i_mc%i.pickle" % (h, n_mc_sims)):
        return pickle.load(open("_rankdiffs_h%i_mc%i.pickle" % (h, n_mc_sims), "rb"))

    (pbsv_rmse, _), (oshapley, ishapley), oshapley_vi, ishapley_vi = pickle.load(
        open("../../Results/Updated CPI 2/shapleys_h%i_upd.bin" % h, "rb")
    )

    unused_enet = ishapley_vi.loc["enet"][np.isclose(ishapley_vi.loc["enet"], 0)].index

    ishapley_w = ishapley_vi.drop("base_contribution", axis=1).divide(ishapley_vi.drop("base_contribution", axis=1).mean(axis=1), axis=0)

    oshapley_w = oshapley_vi.drop("base_contribution", axis=1).divide(oshapley_vi.drop("base_contribution", axis=1).mean(axis=1), axis=0)

    ishapley_enet_w = ishapley_vi.loc["enet"].drop("base_contribution").drop(unused_enet)
    ishapley_enet_w /= ishapley_enet_w.mean()

    oshapley_enet_w = oshapley_vi.loc["enet"].drop("base_contribution").drop(unused_enet)
    oshapley_enet_w /= oshapley_enet_w.mean()

    ishapley_r = ishapley_vi.drop("base_contribution", axis=1).rank(axis=1, ascending=False)
    oshapley_r = oshapley_vi.drop("base_contribution", axis=1).rank(axis=1, ascending=False)
    pbsv_rmse_r = pbsv_rmse.drop("base_contribution", axis=1).rank(axis=1, ascending=True)

    ishapley_enet_r = ishapley_vi.loc["enet"].drop("base_contribution").drop(unused_enet).rank(ascending=False)
    oshapley_enet_r = oshapley_vi.loc["enet"].drop("base_contribution").drop(unused_enet).rank(ascending=False)
    pbsv_rmse_enet_r = pbsv_rmse.loc["enet"].drop("base_contribution").drop(unused_enet).rank(ascending=True)

    if 1 == 2:
        n_c = 75

        from sklearn.cluster import KMeans

        cluster_df = pd.DataFrame({
            k: KMeans(n_clusters=n_c).fit_predict(
                ishapley_vi.drop("base_contribution", axis=1).loc[k].to_numpy().reshape(-1, 1)
            )
            for k in pbsv_rmse.index
        }, index=ishapley_vi.drop("base_contribution", axis=1).columns)

        ishapley_c = pd.concat([
            pd.concat([
                ishapley.xs(k, level=1)[cluster_df.index[cluster_df[k] == c]].sum(axis=1).rename(c)
                for c in range(n_c)
            ], axis=1)
            for k in pbsv_rmse.index
        ], keys=pbsv_rmse.index)

        ishapley_vi = ishapley.abs().groupby(level=1).mean()

        oshapley_c = pd.concat([
            pd.concat([
                oshapley.loc[k][cluster_df.index[cluster_df[k] == c]].sum(axis=1).rename(c)
                for c in range(n_c)
            ], axis=1)
            for k in pbsv_rmse.index
        ], keys=pbsv_rmse.index)

        oshapley_c_vi = oshapley_c.abs().groupby(level=0).mean()
        oshapley_c_vi_r = oshapley_c_vi.rank(ascending=False, axis=1)

        pbsv_rmse_c = pd.concat([
            pd.Series([
                pbsv_rmse.loc[k].loc[cluster_df.index[cluster_df[k] == c]].sum()
                for c in range(n_c)
            ], index=range(n_c)).rename(k)
            for k in pbsv_rmse.index
        ], axis=1).T

        pbsv_rmse_c_r = pbsv_rmse_c.rank(ascending=True, axis=1)

    p = pbsv_rmse_r.shape[1]
    max_msdr = (p**2-1) / 3
    # rnd_msdr = (p**2-1) / 6

    p_enet = pbsv_rmse_enet_r.shape[0]
    max_msdr_enet = (p_enet**2-1) / 3
    # rnd_msdr_enet = (p_enet**2-1) / 6

    # oshapley_vs_pbsv_msdr_max_norm_enet = ((oshapley_enet_r-pbsv_rmse_enet_r)**2).mean() / max_msdr_enet
    # ishapley_vs_pbsv_msdr_max_norm_enet = ((ishapley_enet_r-pbsv_rmse_enet_r)**2).mean() / max_msdr_enet
    # ishapley_vs_oshapley_msdr_max_norm_enet = ((ishapley_enet_r-oshapley_enet_r)**2).mean() / max_msdr_enet

    def rnd_ranks(_p):
        return np.random.uniform(size=(n_mc_sims, _p)).argsort(axis=1) + 1

    def find_quantile(x, list_x):
        list_x_sorted = np.sort(list_x)
        position = np.searchsorted(list_x_sorted, x)
        quantile = position / len(list_x)
        return quantile

    _rnd_msdr_enet_i = ((np.tile(ishapley_enet_r.to_numpy(), (n_mc_sims, 1)) - rnd_ranks(p_enet)) ** 2 * np.tile(ishapley_enet_w, (n_mc_sims, 1))).mean(axis=1)
    _rnd_msdr_enet_i_pval = find_quantile(((ishapley_enet_r-pbsv_rmse_enet_r)**2*ishapley_enet_w).mean(), _rnd_msdr_enet_i)
    _rnd_msdr_enet_io_pval = find_quantile(((ishapley_enet_r-oshapley_enet_r)**2*ishapley_enet_w).mean(), _rnd_msdr_enet_i)
    _rnd_msdr_enet_i = _rnd_msdr_enet_i.mean()

    _rnd_msdr_enet_i_uw = ((np.tile(ishapley_enet_r.to_numpy(), (n_mc_sims, 1)) - rnd_ranks(p_enet)) ** 2).mean(axis=1)
    _rnd_msdr_enet_i_pval_uw = find_quantile(((ishapley_enet_r-pbsv_rmse_enet_r)**2).mean(), _rnd_msdr_enet_i_uw)
    _rnd_msdr_enet_io_pval_uw = find_quantile(((ishapley_enet_r-oshapley_enet_r)**2).mean(), _rnd_msdr_enet_i_uw)
    _rnd_msdr_enet_i_uw = _rnd_msdr_enet_i_uw.mean()

    _rnd_msdr_enet_o = ((np.tile(oshapley_enet_r.to_numpy(), (n_mc_sims, 1)) - rnd_ranks(p_enet)) ** 2 * np.tile(oshapley_enet_w, (n_mc_sims, 1))).mean(axis=1)
    _rnd_msdr_enet_o_pval = find_quantile(((oshapley_enet_r-pbsv_rmse_enet_r)**2*oshapley_enet_w).mean(), _rnd_msdr_enet_o)
    _rnd_msdr_enet_o = _rnd_msdr_enet_o.mean()

    _rnd_msdr_enet_o_uw = ((np.tile(oshapley_enet_r.to_numpy(), (n_mc_sims, 1)) - rnd_ranks(p_enet)) ** 2).mean(axis=1)
    _rnd_msdr_enet_o_pval_uw = find_quantile(((oshapley_enet_r-pbsv_rmse_enet_r)**2).mean(), _rnd_msdr_enet_o_uw)
    _rnd_msdr_enet_o_uw = _rnd_msdr_enet_o_uw.mean()

    oshapley_vs_pbsv_msdr_rnd_norm_enet_w = ((oshapley_enet_r-pbsv_rmse_enet_r)**2*oshapley_enet_w).mean() / _rnd_msdr_enet_o
    ishapley_vs_pbsv_msdr_rnd_norm_enet_w = ((ishapley_enet_r-pbsv_rmse_enet_r)**2*ishapley_enet_w).mean() / _rnd_msdr_enet_i
    ishapley_vs_oshapley_msdr_rnd_norm_enet_w = ((ishapley_enet_r-oshapley_enet_r)**2*ishapley_enet_w).mean() / _rnd_msdr_enet_i

    oshapley_vs_pbsv_msdr_rnd_norm_enet = ((oshapley_enet_r-pbsv_rmse_enet_r)**2).mean() / _rnd_msdr_enet_o_uw
    ishapley_vs_pbsv_msdr_rnd_norm_enet = ((ishapley_enet_r-pbsv_rmse_enet_r)**2).mean() / _rnd_msdr_enet_i_uw
    ishapley_vs_oshapley_msdr_rnd_norm_enet = ((ishapley_enet_r-oshapley_enet_r)**2).mean() / _rnd_msdr_enet_i_uw

    # oshapley_vs_pbsv_msdr_max_norm = ((oshapley_r-pbsv_rmse_r)**2).mean(axis=1) / max_msdr
    # ishapley_vs_pbsv_msdr_max_norm = ((ishapley_r-pbsv_rmse_r)**2).mean(axis=1) / max_msdr
    # ishapley_vs_oshapley_msdr_max_norm = ((ishapley_r-oshapley_r) ** 2).mean(axis=1) / max_msdr

    _rnd_msdr_i, _rnd_msdr_i_pval, _rnd_msdr_io_pval = {}, {}, {}
    for k in oshapley_r.index:
        _rnd_msdr_i_k = ((np.tile(ishapley_r.loc[k].to_numpy(), (n_mc_sims, 1)) - rnd_ranks(p)) ** 2 * np.tile(ishapley_w.loc[k], (n_mc_sims, 1))).mean(axis=1)
        _rnd_msdr_i_pval[k] = find_quantile(((ishapley_r.loc[k] - pbsv_rmse_r.loc[k]) ** 2 * ishapley_w.loc[k]).mean(), _rnd_msdr_i_k)
        _rnd_msdr_io_pval[k] = find_quantile(((ishapley_r.loc[k] - oshapley_r.loc[k]) ** 2 * ishapley_w.loc[k]).mean(), _rnd_msdr_i_k)
        _rnd_msdr_i[k] = _rnd_msdr_i_k.mean()
    _rnd_msdr_i, _rnd_msdr_i_pval, _rnd_msdr_io_pval = pd.Series(_rnd_msdr_i), pd.Series(_rnd_msdr_i_pval), pd.Series(_rnd_msdr_io_pval)

    _rnd_msdr_i_uw, _rnd_msdr_i_pval_uw, _rnd_msdr_io_pval_uw = {}, {}, {}
    for k in oshapley_r.index:
        _rnd_msdr_i_k_uw = ((np.tile(ishapley_r.loc[k].to_numpy(), (n_mc_sims, 1)) - rnd_ranks(p)) ** 2).mean(axis=1)
        _rnd_msdr_i_pval_uw[k] = find_quantile(((ishapley_r.loc[k] - pbsv_rmse_r.loc[k]) ** 2).mean(), _rnd_msdr_i_k_uw)
        _rnd_msdr_io_pval_uw[k] = find_quantile(((ishapley_r.loc[k] - oshapley_r.loc[k]) ** 2).mean(), _rnd_msdr_i_k_uw)
        _rnd_msdr_i_uw[k] = _rnd_msdr_i_k_uw.mean()
    _rnd_msdr_i_uw, _rnd_msdr_i_pval_uw, _rnd_msdr_io_pval_uw = pd.Series(_rnd_msdr_i_uw), pd.Series(_rnd_msdr_i_pval_uw), pd.Series(_rnd_msdr_io_pval_uw)
    
    _rnd_msdr_o, _rnd_msdr_o_pval = {}, {}
    for k in oshapley_r.index:
        _rnd_msdr_o_k = ((np.tile(oshapley_r.loc[k].to_numpy(), (n_mc_sims, 1)) - rnd_ranks(p)) ** 2 * np.tile(oshapley_w.loc[k], (n_mc_sims, 1))).mean(axis=1)
        _rnd_msdr_o_pval[k] = find_quantile(((oshapley_r.loc[k] - pbsv_rmse_r.loc[k]) ** 2 * oshapley_w.loc[k]).mean(), _rnd_msdr_o_k)
        _rnd_msdr_o[k] = _rnd_msdr_o_k.mean()
    _rnd_msdr_o, _rnd_msdr_o_pval = pd.Series(_rnd_msdr_o), pd.Series(_rnd_msdr_o_pval)

    _rnd_msdr_o_uw, _rnd_msdr_o_pval_uw = {}, {}
    for k in oshapley_r.index:
        _rnd_msdr_o_k_uw = ((np.tile(oshapley_r.loc[k].to_numpy(), (n_mc_sims, 1)) - rnd_ranks(p)) ** 2).mean(axis=1)
        _rnd_msdr_o_pval_uw[k] = find_quantile(((oshapley_r.loc[k] - pbsv_rmse_r.loc[k]) ** 2).mean(), _rnd_msdr_o_k_uw)
        _rnd_msdr_o_uw[k] = _rnd_msdr_o_k_uw.mean()
    _rnd_msdr_o_uw, _rnd_msdr_o_pval_uw = pd.Series(_rnd_msdr_o_uw), pd.Series(_rnd_msdr_o_pval_uw)

    oshapley_vs_pbsv_msdr_rnd_norm_w = ((oshapley_r-pbsv_rmse_r)**2*oshapley_w).mean(axis=1) / _rnd_msdr_o
    ishapley_vs_pbsv_msdr_rnd_norm_w = ((ishapley_r-pbsv_rmse_r)**2*ishapley_w).mean(axis=1) / _rnd_msdr_i
    ishapley_vs_oshapley_msdr_rnd_norm_w = ((ishapley_r-oshapley_r)**2*ishapley_w).mean(axis=1) / _rnd_msdr_i

    oshapley_vs_pbsv_msdr_rnd_norm = ((oshapley_r-pbsv_rmse_r)**2).mean(axis=1) / _rnd_msdr_o_uw
    ishapley_vs_pbsv_msdr_rnd_norm = ((ishapley_r-pbsv_rmse_r)**2).mean(axis=1) / _rnd_msdr_i_uw
    ishapley_vs_oshapley_msdr_rnd_norm = ((ishapley_r-oshapley_r)**2).mean(axis=1) / _rnd_msdr_i_uw

    _rnd_msdr_i_pval.loc["enet"] = _rnd_msdr_enet_i_pval
    _rnd_msdr_o_pval.loc["enet"] = _rnd_msdr_enet_o_pval
    _rnd_msdr_io_pval.loc["enet"] = _rnd_msdr_enet_io_pval

    _rnd_msdr_i_pval_uw.loc["enet"] = _rnd_msdr_enet_i_pval_uw
    _rnd_msdr_o_pval_uw.loc["enet"] = _rnd_msdr_enet_o_pval_uw
    _rnd_msdr_io_pval_uw.loc["enet"] = _rnd_msdr_enet_io_pval_uw

    oshapley_vs_pbsv_msdr_rnd_norm["enet"] = oshapley_vs_pbsv_msdr_rnd_norm_enet
    ishapley_vs_pbsv_msdr_rnd_norm["enet"] = ishapley_vs_pbsv_msdr_rnd_norm_enet
    ishapley_vs_oshapley_msdr_rnd_norm["enet"] = ishapley_vs_oshapley_msdr_rnd_norm_enet

    # oshapley_vs_pbsv_msdr_max_norm["enet"] = oshapley_vs_pbsv_msdr_max_norm_enet
    # ishapley_vs_pbsv_msdr_max_norm["enet"] = ishapley_vs_pbsv_msdr_max_norm_enet
    # ishapley_vs_oshapley_msdr_max_norm["enet"] = ishapley_vs_oshapley_msdr_max_norm_enet

    oshapley_vs_pbsv_msdr_rnd_norm_w["enet"] = oshapley_vs_pbsv_msdr_rnd_norm_enet_w
    ishapley_vs_pbsv_msdr_rnd_norm_w["enet"] = ishapley_vs_pbsv_msdr_rnd_norm_enet_w
    ishapley_vs_oshapley_msdr_rnd_norm_w["enet"] = ishapley_vs_oshapley_msdr_rnd_norm_enet_w

    res_dict = {
        "oshapley_vs_pbsv_msdr_rnd_norm": oshapley_vs_pbsv_msdr_rnd_norm,
        "ishapley_vs_pbsv_msdr_rnd_norm": ishapley_vs_pbsv_msdr_rnd_norm,
        "ishapley_vs_oshapley_msdr_rnd_norm": ishapley_vs_oshapley_msdr_rnd_norm,
        "oshapley_vs_pbsv_msdr_rnd_norm_w": oshapley_vs_pbsv_msdr_rnd_norm_w,
        "ishapley_vs_pbsv_msdr_rnd_norm_w": ishapley_vs_pbsv_msdr_rnd_norm_w,
        "ishapley_vs_oshapley_msdr_rnd_norm_w": ishapley_vs_oshapley_msdr_rnd_norm_w,
        "ishapley_vs_pbsv_msdr_w_pval": _rnd_msdr_i_pval,
        "oshapley_vs_pbsv_msdr_w_pval": _rnd_msdr_o_pval,
        "ishapley_vs_oshapley_msdr_w_pval": _rnd_msdr_io_pval,
        "ishapley_vs_pbsv_msdr_uw_pval": _rnd_msdr_i_pval_uw,
        "oshapley_vs_pbsv_msdr_uw_pval": _rnd_msdr_o_pval_uw,
        "ishapley_vs_oshapley_msdr_uw_pval": _rnd_msdr_io_pval_uw,
    }

    pickle.dump(res_dict, open("_rankdiffs_h%i_mc%i.pickle" % (h, n_mc_sims), "wb"))
    return res_dict


def estimate(h):

    def mapper(key: AnatomyModelProvider.PeriodKey) -> \
            AnatomyModelProvider.PeriodValue:

        run_dir = os.path.join(MODELS_DIR, "20241005_091334", "cpiaucsl_h%i" % h)

        ps = {
            "nn_deep": run_dir,
            "nn_shallow": run_dir,
            "rf_cv": run_dir,
            "xgb_cv": run_dir,
            "enet": run_dir,
            "pca": run_dir,
        }

        d = pickle.load(
            open("%s/%i.bin" % (ps[key.model_name], key.period), "rb")
        )

        train, test = d["train"], d["test"]

        if key.model_name in ["nn_deep", "nn_shallow"]:

            m = d["models"][key.model_name][0]

            scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train.drop("y_h%i" % h, axis=1))

            def pred_fn(x: np.ndarray) -> np.ndarray:
                return np.array(m.predict(scaler.transform(x))).flatten()

        elif key.model_name == "pca":

            (pca_m, x_mean, x_std), m = d["models"][key.model_name]

            def pred_fn(x: np.ndarray) -> np.ndarray:
                return m.predict(pca_m.transform((x - x_mean) / x_std)[:, :m.coef_.shape[0]])

        elif key.model_name == "enet":

            (x_mean, x_std), m = d["models"][key.model_name]

            def pred_fn(x: np.ndarray) -> np.ndarray:
                return np.array(m.predict((x - x_mean) / x_std)).flatten()

        elif key.model_name == "xgb_cv":

            m = d["models"][key.model_name][0]

            def pred_fn(x: np.ndarray) -> np.ndarray:
                x = pd.DataFrame(x, columns=m.get_booster().feature_names)
                return np.array(m.predict(x)).flatten()

        else:

            m = d["models"][key.model_name][0]

            def pred_fn(x: np.ndarray) -> np.ndarray:
                return np.array(m.predict(x)).flatten()

        model = AnatomyModel(pred_fn)
        return AnatomyModelProvider.PeriodValue(train, test, model)

    provider = AnatomyModelProvider(
        n_periods=416,
        n_features=247,
        model_names=["rf_cv", "xgb_cv", "nn_deep", "nn_shallow", "enet", "pca"],
        y_name="y_h%i" % h,
        provider_fn=mapper
    )

    Anatomy(provider=provider, n_iterations=100).precompute(
        n_jobs=30,
        save_path="../../Results/Updated CPI 3/anatomy_h%i_upd.bin" % h
    )


def anatomize(h):

    anatomy = Anatomy.load("../../Results/Updated CPI 3/anatomy_h%i_upd.bin" % h)

    groups = {
        "rf_cv": ["rf_cv"],
        "xgb_cv": ["xgb_cv"],
        "nn_comb": ["nn_deep", "nn_shallow"],
        "enet": ["enet"],
        "pca": ["pca"],
        "lin_comb": ["enet", "pca"],
        "nonlin_comb": ["rf_cv", "xgb_cv", "nn_deep", "nn_shallow"],
        "all_comb": ["enet", "pca", "rf_cv", "xgb_cv", "nn_deep", "nn_shallow"]
    }

    def transform(y_hat, y):
        return np.sqrt(np.mean((y - y_hat) ** 2))

    rmse = anatomy.explain(
        model_sets=AnatomyModelCombination(groups=groups),
        transformer=AnatomyModelOutputTransformer(transform=transform)
    )

    rmse_years = {}
    for year in tqdm(range(1990, 2024+1)):
        rmse_years[year] = anatomy.explain(
            model_sets=AnatomyModelCombination(groups=groups),
            transformer=AnatomyModelOutputTransformer(transform=transform),
            explanation_subset=anatomy.get_forecast_index()[
                (anatomy.get_forecast_index() + pd.DateOffset(months=h)).map(lambda x: x.year == year)
            ]
        )

    def transform(y_hat):
        return y_hat

    raw = anatomy.explain(
        model_sets=AnatomyModelCombination(groups=groups),
        transformer=AnatomyModelOutputTransformer(transform=transform),
        # explanation_subset=anatomy.get_forecast_index()[:anatomy.get_forecast_index().get_loc("2020-01-01")]
    )

    def consolidate_columns(df):
        df["ar"] = df[[x for x in df.columns if x.startswith("y_t-")]].sum(axis=1)
        v_columns = [x for x in df.columns if not x.startswith("y_t-") and not x.endswith("_ma3")]
        return pd.concat([
            df[[x, "%s_ma3" % x]].sum(axis=1).rename(x) if "%s_ma3" % x in df.columns else df[x]
            for x in v_columns
        ], axis=1)

    oshapley = consolidate_columns(raw)
    oshapley_vi = consolidate_columns(raw).abs().groupby(level=0).mean()
    # oshapley_vi = oshapley.rank(axis=1, ascending=False)

    pbsv_rmse = consolidate_columns(rmse.reset_index(level=1, drop=True))
    # pbsv_rmse_vi = pbsv_rmse.rank(axis=1, ascending=True)
    
    pbsv_rmse_years = {k: consolidate_columns(v.reset_index(level=1, drop=True)) for k, v in rmse_years.items()}

    if 1 == 2:
        _pbsv_rmse = pbsv_rmse.drop("base_contribution", axis=1).rank(axis=1, ascending=True)
        _oshapley = oshapley_vi.drop("base_contribution", axis=1).rank(axis=1, ascending=False)
        p = _pbsv_rmse.shape[1]
        max_msdr = (p**2-1) / 3
        msdr_norm = ((_oshapley-_pbsv_rmse)**2).mean(axis=1) / max_msdr
        breakpoint()

    ishapley = pd.read_pickle("../../Results/Updated CPI 3/ishapley_h%i_upd.bin" % h)

    ishapley["nn_comb"] = ishapley.apply(lambda x: (x["nn_deep"] + x["nn_shallow"]) / 2, axis=1)
    ishapley["nonlin_comb"] = ishapley.apply(lambda x: (x["rf_cv"] + x["xgb_cv"] + x["nn_deep"] + x["nn_shallow"]) / 4, axis=1)
    ishapley["lin_comb"] = ishapley.apply(lambda x: (x["enet"] + x["pca"]) / 2, axis=1)
    ishapley["all_comb"] = ishapley.apply(lambda x: (x["enet"] + x["pca"] + x["xgb_cv"] + x["rf_cv"] + x["nn_deep"] + x["nn_shallow"]) / 6, axis=1)

    ishapley = pd.concat([
        ishapley.iloc[i].apply(
            lambda x: consolidate_columns(x.rename({"base_value": "base_contribution"}, axis=1)).abs().mean(axis=0)
        )
        for i in range(ishapley.shape[0])
    ], keys=ishapley.index)

    # have to stop here, but we cannot store ishapleys like this, we cannot take ABS here

    ishapley_vi = ishapley.groupby(level=1).mean()

    pickle.dump(
        ((pbsv_rmse, pbsv_rmse_years), (oshapley, ishapley), oshapley_vi, ishapley_vi),
        open("../../Results/Updated CPI 3/shapleys_h%i_upd.bin" % h, "wb")
    )


def _fix(h):

    (pbsv_rmse, pbsv_rmse_years), oshapley, oshapley_vi, ishapley_vi = pickle.load(
        # open("../../Results/Updated CPI 2/shapleys_h%i.bin" % h, "rb")
        open("../../Results/Updated CPI 2/shapleys_h%i_upd.bin" % h, "rb")
    )

    ishapley = pd.read_pickle("../../Results/Updated CPI 2/ishapley_h%i_upd.bin" % h)

    ishapley["nn_comb"] = ishapley.apply(lambda x: (x["nn_deep"] + x["nn_shallow"]) / 2, axis=1)
    ishapley["nonlin_comb"] = ishapley.apply(lambda x: (x["rf_cv"] + x["nn_deep"] + x["nn_shallow"]) / 3, axis=1)
    ishapley["lin_comb"] = ishapley.apply(lambda x: (x["enet"] + x["pca"]) / 2, axis=1)

    def consolidate_columns(df):
        df["ar"] = df[[x for x in df.columns if x.startswith("y_t-")]].sum(axis=1)
        v_columns = [x for x in df.columns if not x.startswith("y_t-") and not x.endswith("_ma3")]
        return pd.concat([
            df[[x, "%s_ma3" % x]].sum(axis=1).rename(x) if "%s_ma3" % x in df.columns else df[x]
            for x in v_columns
        ], axis=1)

    ishapley = pd.concat([
        ishapley.iloc[i].apply(
            lambda x: consolidate_columns(x.rename({"base_value": "base_contribution"}, axis=1)).abs().mean(axis=0)
        )
        for i in range(ishapley.shape[0])
    ], keys=ishapley.index)

    ishapley_vi = ishapley.groupby(level=1).mean()

    pickle.dump(
        ((pbsv_rmse, pbsv_rmse_years), oshapley, oshapley_vi, ishapley_vi),
        open("../../Results/Updated CPI 2/shapleys_h%i_upd.bin" % h, "wb")
    )


def plot():

    inv = False

    sns.set_theme()
    sns.set_context("paper", font_scale=1.1, rc={"font.family": "Arial"})

    ylim = {
        "lin_comb": {1: (-3, 11.5), 3: (-4, 13.5), 6: (-4, 13.5), 12: (-3.5, 11.5)},
        "all_comb": {1: (-3, 10.5), 3: (-3, 11.5), 6: (-4, 10.5), 12: (-2.5, 9)},
        "pca": {1: (-3, 8), 3: (-3, 8.5), 6: (-2, 8.5), 12: (-2, 8.5)},
        "enet": {1: (-4, 17), 3: (-7, 18.5), 6: (-8.5, 22.5), 12: (-7, 18)},
        "nonlin_comb": {1: (-2, 8.5), 3: (-2.5, 8.5), 6: (-4, 8.5), 12: (-2, 7)},
        "rf_cv": {1: (-2, 18), 3: (-4.5, 9), 6: (-2, 9), 12: (-2.75, 10)},
        "xgb_cv": {1: (-3, 11), 3: (-3, 9), 6: (-2, 9), 12: (-2.5, 7)},
        "nn_comb": {1: (-1.5, 5.5), 3: (-4, 7), 6: (-4, 7), 12: (-3, 7)},
    }
    yticks = {
        "lin_comb": {1: [0, 5, 10], 3: [0, 5, 10], 6: [0, 5, 10], 12: [0, 4, 8]},
        "all_comb": {1: [0, 5, 10], 3: [0, 5, 10], 6: [0, 5, 10], 12: [0, 4, 8]},
        "pca": {1: [-2.5, 0, 2.5, 5], 3: [0, 2.5, 5, 7.5], 6: [0, 2.5, 5, 7.5], 12: [0, 2.5, 5, 7.5]},
        "enet": {1: [0, 5, 10, 15], 3: [-5, 0, 5, 10, 15], 6: [-5, 0, 5, 10, 15, 20], 12: [-5, 0, 5, 10, 15]},
        "nonlin_comb": {1: [0, 2.5, 5], 3: [0, 2.5, 5], 6: [-2.5, 0, 2.5, 5, 7.5], 12: [0, 2.5, 5]},
        "rf_cv": {1: [0, 5, 10, 15], 3: [-2.5, 0, 2.5, 5, 7.5], 6: [-2.5, 0, 2.5, 5, 7.5], 12: [0, 2.5, 5, 7.5]},
        "xgb_cv": {1: [0, 5, 10], 3: [-2.5, 0, 2.5, 5, 7.5], 6: [-2.5, 0, 2.5, 5, 7.5], 12: [0, 2.5, 5]},
        "nn_comb": {1: [0, 2, 4], 3: [-2.5, 0, 2.5, 5], 6: [-2.5, -0, 2.5, 5], 12: [-2.5, 0, 2.5, 5]},
    }

    if not inv:
        top_n, bottom_n = 25, 5
    else:
        top_n, bottom_n = 15, 15

    model_keys = ["pca", "enet", "lin_comb", "nn_comb", "rf_cv", "xgb_cv", "nonlin_comb", "all_comb"]

    for hs in ([1, 3, 6, 12], [1, 3], [6, 12]):

        for model_key in model_keys:

            if len(hs) == 4:
                fig, axs = plt.subplots(4, 1, figsize=(7.2, 7.75))
            else:
                fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.5))

            axs = axs.flatten()

            for ax, h, panel_name in zip(axs, hs, string.ascii_uppercase):

                import os
                if not os.path.exists("../../Results/Updated CPI 2/shapleys_h%i_upd.bin" % h):
                    continue

                (pbsv_rmse, _), oshapley, oshapley_vi, ishapley_vi = pickle.load(
                    #open("../../Results/Updated CPI 2/shapleys_h%i.bin" % h, "rb")
                    open("../../Results/Updated CPI 2/shapleys_h%i_upd.bin" % h, "rb")
                )

                assert pbsv_rmse.shape[1] == oshapley.shape[1] == oshapley_vi.shape[1] == ishapley_vi.shape[1]

                if not inv:
                    bar_df = pd.concat((
                        ishapley_vi.loc[model_key].drop(
                            "base_contribution").sort_values(ascending=False).rename("vi in-sample"),
                        oshapley_vi.loc[model_key].drop(
                            "base_contribution").rename("vi out-of-sample"),
                        (-pbsv_rmse.loc[model_key]).drop(  # note negation
                            "base_contribution").rename("pbsv out-of-sample")
                    ), axis=1)
                else:
                    bar_df = pd.concat((
                        (-pbsv_rmse.loc[model_key]).drop(  # note negation
                            "base_contribution").sort_values(ascending=False).rename("pbsv out-of-sample"),
                        ishapley_vi.loc[model_key].drop(
                            "base_contribution").rename("vi in-sample"),
                        oshapley_vi.loc[model_key].drop(
                            "base_contribution").rename("vi out-of-sample"),
                    ), axis=1)

                bar_df.index = bar_df.index.map(lambda x: x.replace("fred_", ""))

                bar_df_full = bar_df.copy()

                bar_df["vi in-sample"] = (bar_df["vi in-sample"] / bar_df["vi in-sample"].sum()) * 100
                bar_df["vi out-of-sample"] = (bar_df["vi out-of-sample"] / bar_df["vi out-of-sample"].sum()) * 100
                bar_df["pbsv out-of-sample"] *= 100  # contribution to %-RMSE
                bar_df = pd.concat((bar_df[:top_n], bar_df[-bottom_n:]))

                bar_df_alt, bar_df = bar_df["vi out-of-sample"].to_numpy(), bar_df.drop("vi out-of-sample", axis=1)

                if inv:
                    bar_df = bar_df[["vi in-sample", "pbsv out-of-sample"]]

                bar_df_empty = bar_df.copy()
                bar_df_empty.iloc[:, :] = 0

                if not inv:
                    f = bar_df["vi in-sample"].max() / bar_df["pbsv out-of-sample"].max()
                    bar_df["pbsv out-of-sample"] *= f
                else:
                    f = bar_df["vi in-sample"].loc[bar_df["pbsv out-of-sample"].idxmax()] / bar_df["pbsv out-of-sample"].max()
                    bar_df["pbsv out-of-sample"] *= f

                bar_df.plot(kind="bar", edgecolor="none", ax=ax, sharex=False, legend=False, color=["red", "green"])

                ax.axvline(top_n - .5, color="black", alpha=0.75, ls=":")

                bar_df_empty.plot(
                    kind="bar", edgecolor="none", ax=ax, sharex=False, legend=False,
                    yerr=np.array([[[0, 0], [x, 0]] for x in bar_df_alt]).T, ecolor="black", capsize=3
                )

                ax.containers[2].lines[1][0].remove()
                ax.containers[4].lines[1][0].remove()
                ax.containers[4].lines[1][1].remove()

                ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: "%.1f%%" % x))

                ax.set_ylim(*ylim[model_key][h]), ax.set_yticks(yticks[model_key][h]), ax.set_xticklabels([])

                a, b = ax.get_xlim(), ax.get_xticks()

                second_y = ax.twinx()

                x_lab = ax.twiny()
                x_lab.set_xlim(*a), x_lab.set_xticks(b), x_lab.grid(False)
                x_lab.xaxis.tick_bottom(), x_lab.tick_params(bottom=False)
                x_lab.set_xticklabels(
                    bar_df.index, rotation=45, ha="right", rotation_mode="anchor", position=(0, .015), fontsize=7.5
                )

                second_y.set_ylim(ylim[model_key][h]), second_y.set_yticks(yticks[model_key][h])
                second_y.yaxis.set_major_formatter(mtick.FuncFormatter(
                    lambda x, _, ff=f: ("%.3f%%" % (-x / ff * 1)).replace("0.", ".")  # note negation
                ))
                second_y.grid(False)
                """second_y.text(
                    1.013, yticks[model_key][h][-1] * 1.2, r"$\times10^{-5}$",
                    transform=second_y.get_yaxis_transform(), fontsize=8
                )"""

                [b.set_alpha(0) for a, b in zip(ax.get_yticks(), ax.get_yticklabels()) if a < 0]
                ax.tick_params(left=False, bottom=False), second_y.tick_params(right=False, bottom=False)
                ax.text(
                    0.985, 0.82, r"$h=%i$" % h, backgroundcolor="#EAEAF2",
                    horizontalalignment="right", transform=ax.transAxes, fontsize=10
                )

                if not inv:

                    for i, rect in enumerate(ax.patches[:bar_df.shape[0] * 2][bar_df.shape[0]:]):

                        y_pad = bar_df.abs().to_numpy().flatten().mean() * .15

                        neg = bar_df["pbsv out-of-sample"].iloc[i] < 0
                        pos_neg_ranks = pd.concat((
                            -bar_df_full["pbsv out-of-sample"][bar_df_full["pbsv out-of-sample"] < 0].rank(
                                ascending=True),
                            bar_df_full["pbsv out-of-sample"][bar_df_full["pbsv out-of-sample"] >= 0].rank(
                                ascending=False)
                        ), axis=0).reindex(bar_df["pbsv out-of-sample"].index)

                        label = "%i" % pos_neg_ranks.iloc[i]

                        ax.text(
                            rect.get_x() + rect.get_width() / 2, rect.get_height() + (-y_pad if neg else y_pad),
                            label, ha="center", va="top" if neg else "bottom", fontsize=8,
                            bbox=dict(boxstyle="square,pad=.15", fc="#EAEAF2", ec="none", alpha=.6),
                            # fontweight="bold" if abs(pos_neg_ranks.iloc[i]) <= n_highlight else "normal",
                            # alpha=1 if abs(pos_neg_ranks.iloc[i]) <= n_highlight else .75,
                            # color="#820000" if neg else "black",
                        )
                else:
                    for i, rect in enumerate(ax.patches[:bar_df.shape[0] * 2][:bar_df.shape[0]]):
                        y_pad = bar_df.abs().to_numpy().flatten().mean() * .35
                        label = "%i" % bar_df["vi in-sample"].rank(ascending=False).iloc[i]
                        ax.text(
                            rect.get_x() + rect.get_width() / 2, rect.get_height() + y_pad,
                            label, ha="center", va= "bottom", fontsize=8,
                            bbox=dict(boxstyle="square,pad=.15", fc="#EAEAF2", ec="none", alpha=.7),
                            # fontweight="bold" if abs(pos_neg_ranks.iloc[i]) <= n_highlight else "normal",
                            # alpha=1 if abs(pos_neg_ranks.iloc[i]) <= n_highlight else .75,
                            # color="#820000" if neg else "black",
                        )

            axs[-1].plot(0, 0, color="black", lw=1, ls="-", label="vi out-of-sample")

            vi_oos_handle = axs[-1].get_legend_handles_labels()[0][
                axs[-1].get_legend_handles_labels()[1].index("vi out-of-sample")
            ]

            vi_is_handle = axs[-1].get_legend_handles_labels()[0][
                axs[-1].get_legend_handles_labels()[1].index("vi in-sample")
            ]

            pbvi_handle = axs[-1].get_legend_handles_labels()[0][
                axs[-1].get_legend_handles_labels()[1].index("pbsv out-of-sample")
            ]

            axs[-1].legend(
                [vi_is_handle, vi_oos_handle, pbvi_handle],
                ["iShapley-VI (left axis)", "oShapley-VI (left axis)", "PBSV (right axis)"],
                loc="upper center", bbox_to_anchor=(0.5, -0.65) if len(hs) == 4 else (0.5, -0.375),
                ncol=10, frameon=False
            )

            if len(hs) == 4:
                plt.subplots_adjust(left=.07, bottom=.13, right=.923, top=.975, wspace=.1, hspace=.685)
            else:
                plt.subplots_adjust(left=.07, bottom=.19, right=.923, top=.965, wspace=.1, hspace=.42)

            plt.savefig(
                "../../Results/Updated CPI 2/isvi-oospbsv-%s%s-rmse_upd%s.pdf" % (
                    model_key.replace("_", "-"), ("-zoom" if hs[0] == 1 else "-zoom2") if len(hs) == 2 else "",
                    "_inv" if inv else ""
                )
            )


def plot_pbsv_vs_isvi():

    inv = True

    sns.set_theme()
    sns.set_context("paper", font_scale=1.1, rc={"font.family": "Arial"})

    ylim = {
        "lin_comb": {1: (-1, 11.5), 3: (-3, 13.5), 6: (-4, 13.5), 12: (-3.5, 11.5)},
        "all_comb": {1: (-1, 9), 3: (-3, 11.5), 6: (-3, 10.5), 12: (-1, 9)},
        "pca": {1: (-3, 8), 3: (-3, 8.5), 6: (-2, 8.5), 12: (-2, 8.5)},
        "enet": {1: (-4, 17), 3: (-7, 18.5), 6: (-8.5, 22.5), 12: (-4, 18)},
        "nonlin_comb": {1: (-1.5, 7.5), 3: (-2.5, 8.5), 6: (-4, 8.5), 12: (-2, 7)},
        "rf_cv": {1: (-2, 18), 3: (-4.2, 9), 6: (-5, 9), 12: (-2.75, 10)},
        "xgb_cv": {1: (-2, 10), 3: (-5, 9.5), 6: (-3, 9), 12: (-2, 7)},
        "nn_comb": {1: (-1.5, 5.5), 3: (-3, 7), 6: (-4, 7), 12: (-2.5, 7)},
    }
    yticks = {
        "lin_comb": {1: [0, 5, 10], 3: [0, 5, 10], 6: [0, 5, 10], 12: [0, 4, 8]},
        "all_comb": {1: [0, 2.5, 5, 7.5], 3: [0, 5, 10], 6: [0, 5, 10], 12: [0, 4, 8]},
        "pca": {1: [-2.5, 0, 2.5, 5], 3: [0, 2.5, 5, 7.5], 6: [0, 2.5, 5, 7.5], 12: [0, 2.5, 5, 7.5]},
        "enet": {1: [0, 5, 10, 15], 3: [-5, 0, 5, 10, 15], 6: [-5, 0, 5, 10, 15], 12: [0, 5, 10, 15]},
        "nonlin_comb": {1: [0, 2.5, 5], 3: [0, 2.5, 5], 6: [-2.5, 0, 2.5, 5, 7.5], 12: [0, 2.5, 5]},
        "rf_cv": {1: [0, 5, 10, 15], 3: [-2.5, 0, 2.5, 5, 7.5], 6: [-4, 0, 4, 8], 12: [0, 2.5, 5, 7.5]},
        "xgb_cv": {1: [0, 3, 6, 9], 3: [-2.5, 0, 2.5, 5, 7.5], 6: [-2.5, 0, 2.5, 5, 7.5], 12: [0, 2.5, 5]},
        "nn_comb": {1: [0, 2, 4], 3: [-2.5, 0, 2.5, 5], 6: [-2.5, -0, 2.5, 5], 12: [0, 2.5, 5]},
    }

    if not inv:
        top_n, bottom_n = 25, 5
    else:
        top_n, bottom_n = 20, 10

    model_keys = ["enet", "pca", "lin_comb", "nn_comb", "rf_cv", "xgb_cv", "nonlin_comb", "all_comb"]

    for hs in ([1, 3, 6, 12], [1, 3], [6, 12]):

        for model_key in model_keys:

            if len(hs) == 4:
                fig, axs = plt.subplots(4, 1, figsize=(7.2, 7.75))
            else:
                fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.5))

            axs = axs.flatten()

            for ax, h, panel_name in zip(axs, hs, string.ascii_uppercase):

                import os
                if not os.path.exists("../../Results/Updated CPI 3/shapleys_h%i_upd.bin" % h):
                    continue

                (pbsv_rmse, _), (oshapley, ishapley), oshapley_vi, ishapley_vi = pickle.load(
                    open("../../Results/Updated CPI 3/shapleys_h%i_upd.bin" % h, "rb")
                )

                assert pbsv_rmse.shape[1] == oshapley.shape[1] == oshapley_vi.shape[1] == ishapley_vi.shape[1]

                if model_key == "enet":
                    # this won't have much of an impact here since it only trims from the end in terms of ranks
                    predictors_used = ishapley_vi.loc[model_key].index[~np.isclose(ishapley_vi.loc[model_key], 0)]
                    pbsv_rmse = pbsv_rmse[predictors_used]
                    oshapley = oshapley[predictors_used]
                    oshapley_vi = oshapley_vi[predictors_used]
                    ishapley_vi = ishapley_vi[predictors_used]

                if not inv:
                    bar_df = pd.concat((
                        ishapley_vi.loc[model_key].drop(
                            "base_contribution").sort_values(ascending=False).rename("vi in-sample"),
                        oshapley_vi.loc[model_key].drop(
                            "base_contribution").rename("vi out-of-sample"),
                        (-pbsv_rmse.loc[model_key]).drop(  # note negation
                            "base_contribution").rename("pbsv out-of-sample")
                    ), axis=1)
                else:
                    bar_df = pd.concat((
                        (-pbsv_rmse.loc[model_key]).drop(  # note negation
                            "base_contribution").sort_values(ascending=False).rename("pbsv out-of-sample"),
                        ishapley_vi.loc[model_key].drop(
                            "base_contribution").rename("vi in-sample"),
                    ), axis=1)

                bar_df.index = bar_df.index.map(lambda x: x.replace("fred_", ""))

                bar_df_full = bar_df.copy()

                bar_df["vi in-sample"] = (bar_df["vi in-sample"] / bar_df["vi in-sample"].sum()) * 100
                bar_df["pbsv out-of-sample"] *= 100  # contribution to %-RMSE
                bar_df = pd.concat((bar_df[:top_n], bar_df[-bottom_n:]))

                if inv:
                    bar_df = bar_df[["pbsv out-of-sample", "vi in-sample"]]

                bar_df_empty = bar_df.copy()
                bar_df_empty.iloc[:, :] = 0

                if inv:
                    f = bar_df["vi in-sample"].max() / bar_df["pbsv out-of-sample"].max()
                    bar_df["pbsv out-of-sample"] *= f
                else:
                    f = bar_df["vi in-sample"].loc[bar_df["pbsv out-of-sample"].idxmax()] / bar_df["pbsv out-of-sample"].max()
                    bar_df["pbsv out-of-sample"] *= f

                bar_df.plot(kind="bar", edgecolor="none", ax=ax, sharex=False, legend=False, color=["green", "red"])

                ax.axvline(top_n - .5, color="black", alpha=0.75, ls=":")

                """bar_df_empty.plot(
                    kind="bar", edgecolor="none", ax=ax, sharex=False, legend=False,
                    yerr=np.array([[[0, 0], [x, 0]] for x in bar_df_alt]).T, ecolor="black", capsize=3
                )"""

                # ax.containers[2].lines[1][0].remove()
                # ax.containers[4].lines[1][0].remove()
                # ax.containers[4].lines[1][1].remove()

                ax.set_ylim(*ylim[model_key][h]), ax.set_yticks(yticks[model_key][h])
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(
                    lambda x, _, ff=f: ("%.3f%%" % (-x / ff * 1)).replace("0.", ".")  # note negation
                ))

                a, b = ax.get_xlim(), ax.get_xticks()

                second_y = ax.twinx()

                x_lab = ax.twiny()
                x_lab.set_xlim(*a), x_lab.set_xticks(b), x_lab.grid(False)
                x_lab.xaxis.tick_bottom(), x_lab.tick_params(bottom=False)
                x_lab.set_xticklabels(
                    bar_df.index, rotation=45, ha="right", rotation_mode="anchor", position=(0, .015), fontsize=7.5
                )

                second_y.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: "%.1f%%" % x))
                second_y.set_ylim(ylim[model_key][h]), second_y.set_yticks(yticks[model_key][h]), second_y.set_xticklabels([])
                second_y.grid(False)

                """second_y.text(
                    1.013, yticks[model_key][h][-1] * 1.2, r"$\times10^{-5}$",
                    transform=second_y.get_yaxis_transform(), fontsize=8
                )"""

                [b.set_alpha(0) for a, b in zip(second_y.get_yticks(), second_y.get_yticklabels()) if a < 0]
                ax.tick_params(left=False, bottom=False), second_y.tick_params(right=False, bottom=False)
                ax.text(
                    0.985, 0.82, r"$h=%i$" % h, backgroundcolor="#EAEAF2",
                    horizontalalignment="right", transform=ax.transAxes, fontsize=10
                )

                if not inv:

                    for i, rect in enumerate(ax.patches[:bar_df.shape[0] * 2][bar_df.shape[0]:]):

                        y_pad = bar_df.abs().to_numpy().flatten().mean() * .15

                        neg = bar_df["pbsv out-of-sample"].iloc[i] < 0
                        pos_neg_ranks = pd.concat((
                            -bar_df_full["pbsv out-of-sample"][bar_df_full["pbsv out-of-sample"] < 0].rank(
                                ascending=True),
                            bar_df_full["pbsv out-of-sample"][bar_df_full["pbsv out-of-sample"] >= 0].rank(
                                ascending=False)
                        ), axis=0).reindex(bar_df["pbsv out-of-sample"].index)

                        label = "%i" % pos_neg_ranks.iloc[i]

                        ax.text(
                            rect.get_x() + rect.get_width() / 2, rect.get_height() + (-y_pad if neg else y_pad),
                            label, ha="center", va="top" if neg else "bottom", fontsize=8,
                            bbox=dict(boxstyle="square,pad=.15", fc="#EAEAF2", ec="none", alpha=.6),
                            # fontweight="bold" if abs(pos_neg_ranks.iloc[i]) <= n_highlight else "normal",
                            # alpha=1 if abs(pos_neg_ranks.iloc[i]) <= n_highlight else .75,
                            # color="#820000" if neg else "black",
                        )
                else:
                    for i, rect in enumerate(ax.patches[:bar_df.shape[0] * 2][bar_df.shape[0]:]):
                        y_pad = bar_df.abs().to_numpy().flatten().mean() * .5
                        label = "%i" % bar_df_full["vi in-sample"].rank(ascending=False).reindex(bar_df.index).iloc[i]
                        for sfs in [8, 8.5, 9, 9.5, 10, 10.5]:
                            ax.text(
                                rect.get_x() + rect.get_width() / 2, rect.get_height() + y_pad,
                                label, ha="center", va="center", fontsize=sfs, fontweight="bold", color="white"
                                # bbox=dict(boxstyle="square,pad=0.4", fc="#EAEAF2", ec="none", alpha=0.5),
                                # fontweight="bold" if abs(pos_neg_ranks.iloc[i]) <= n_highlight else "normal",
                                # alpha=1 if abs(pos_neg_ranks.iloc[i]) <= n_highlight else .75,
                                # color="#820000" if neg else "black",
                            )
                        ax.text(
                            rect.get_x() + rect.get_width() / 2, rect.get_height() + y_pad,
                            label, ha="center", va="center", fontsize=8,
                            # bbox=dict(boxstyle="square,pad=0.4", fc="#EAEAF2", ec="none", alpha=0.5),
                            # fontweight="bold" if abs(pos_neg_ranks.iloc[i]) <= n_highlight else "normal",
                            # alpha=1 if abs(pos_neg_ranks.iloc[i]) <= n_highlight else .75,
                            # color="#820000" if neg else "black",
                        )

            vi_is_handle = axs[-1].get_legend_handles_labels()[0][
                axs[-1].get_legend_handles_labels()[1].index("vi in-sample")
            ]

            pbvi_handle = axs[-1].get_legend_handles_labels()[0][
                axs[-1].get_legend_handles_labels()[1].index("pbsv out-of-sample")
            ]

            axs[-1].legend(
                [pbvi_handle, vi_is_handle],
                ["PBSV (left axis)", "TS-Shapley-VI (right axis)"],
                loc="upper center", bbox_to_anchor=(0.5, -0.65) if len(hs) == 4 else (0.5, -0.375),
                ncol=10, frameon=False
            )

            if len(hs) == 4:
                plt.subplots_adjust(left=.095, bottom=.13, right=.93, top=.985, wspace=.1, hspace=.75)
            else:
                plt.subplots_adjust(left=.085, bottom=.19, right=.923, top=.965, wspace=.1, hspace=.42)

            plt.savefig(
                "../../Results/Updated CPI 3/Figures/PBSV-iShapley/oospbsv-vs-isvi-%s%s-rmse.pdf" % (
                    model_key.replace("_", "-"), ("-zoom" if hs[0] == 1 else "-zoom2") if len(hs) == 2 else ""
                )
            )


def plot_cumsse():

    ys = pickle.load(open("../../Results/Updated CPI 3/ys_upd.bin", "rb"))

    rec_bg_color = "#EAEAF2"  # "#D8D8E8"

    def rec_shade_yearly(ax, year_start, year_end):
        rec = pd.read_excel("./rec.xlsx", engine="openpyxl")
        rec = rec.set_index(pd.to_datetime(rec["date"]))[["USREC"]]
        rec_years = rec.index.map(lambda x: x.year)
        rec = rec.groupby(rec_years).sum() > 0
        rec = rec.index[rec["USREC"] & (rec.index >= year_start) & (rec.index <= year_end)]
        for rec_year in rec:
            ax.axhspan(
                pd.to_datetime("%i-01-01" % rec_year), pd.to_datetime("%i-12-31" % rec_year),
                color=rec_bg_color, zorder=-1
            )
        return rec

    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.1, rc={"font.family": "Arial"})

    xticks = {
        "y_base": {
            "nn_comb": {
                12: [0, .0004, 0.0008],
                6: [0, .0004, 0.0008],
                1: [0, .0007, 0.0014]
            },
            "xgb_cv": {
                12: [0, .0004, 0.0008],
                6: [0, .0004, 0.0008],
                1: [0, .0007, 0.0014]
            },
            "rf_cv": {
                12: [0, .0004, 0.0008],
                6: [0, .0004, 0.0008],
                1: [0, .0007, 0.0014]
            },
            "nonlin_comb": {
                12: [0, .0004, 0.0008],
                6: [0, .0004, 0.0008],
                1: [0, .0007, 0.0014]
            },
            "pca": {
                12: [0, .0004, 0.0008],
                6: [0, .0004, 0.0008],
                1: [0, .0007, 0.0014]
            },
            "enet": {
                12: [0, .0004, 0.0008],
                6: [0, .0004, 0.0008],
                1: [0, .0007, 0.0014]
            },
            "lin_comb": {
                12: [0, .0004, 0.0008],
                6: [0, .0004, 0.0008],
                1: [0, .0007, 0.0014]
            },
            "all_comb": {
                12: [0, .0004, 0.0008],
                6: [0, .0004, 0.0008],
                1: [0, .0007, 0.0014]
            },
        }
    }

    xlims = {
        "y_base": {
            "nn_comb": {
                12: (-0.00048066867366984047, 0.0017156698283347658),
                6: (-0.0003736300737223644, 0.0013645254170294887),
                1: (-0.0008343945037812866, 0.0027775821482052607)
            },
            "xgb_cv": {
                1: (-0.0008060842976658822, 0.0019706804598753353),
                6: (-0.0005042311221619845, 0.001460652175560179),
                12: (-0.0006511725508605837, 0.0016228394151674604)
            },
            "rf_cv": {
                1: (-0.0009287669572997342, 0.002488298440367931),
                6: (-0.00047367604151279464, 0.0014912072562093706),
                12: (-0.0008244091250215993, 0.001671328723292023)
            },
            "nonlin_comb": {
                1: (-0.0010266367407628413, 0.002429125245424944),
                6: (-0.00047990384766861526, 0.001630658440525581),
                12: (-0.0007083365129827546, 0.001813121443167932)
            },
            "pca": {
                1: (-0.0016165621469550076, 0.002536955189373165),
                6: (-0.0008516551235725534, 0.0014621899458762944),
                12: (-0.00040897482304945963, 0.0014183976156407633)
            },
            "enet": {
                1: (-0.0005867148818522509, 0.0030117140775378217),
                6: (-0.0004001055593503967, 0.001505463650669316),
                12: (-0.00047544205946169725, 0.0014281612412316473)
            },
            "lin_comb": {
                1: (-0.00057969212513932494, 0.0026445623084450406),
                6: (-0.0003995389192735185, 0.001411125649841386),
                12: (-0.0004445677916720299, 0.0016001711212104532)
            },
            "all_comb": {
                1: (-0.0011173926617868084, 0.0024426909585660034),
                6: (-0.00043769961927252275, 0.0015677485880396511),
                12: (-0.0004514309406825588, 0.0017449075613220474)
            }
        },
    }

    paddings = {
        "y_base": {
            "nn_comb": {
                12: .000085, 6: .00007, 1: .0001
            },
            "xgb_cv": {
                12: .000085, 6: .00007, 1: .0001
            },
            "rf_cv": {
                12: .000085, 6: .00007, 1: .0001
            },
            "nonlin_comb": {
                12: .000085, 6: .00007, 1: .0001
            },
            "pca": {
                12: .000085, 6: .00007, 1: .0001
            },
            "enet": {
                12: .000085, 6: .00007, 1: .0001
            },
            "lin_comb": {
                12: .000085, 6: .00007, 1: .0001
            },
            "all_comb": {
                12: .000085, 6: .00007, 1: .0001
            },
        }
    }

    years = np.arange(1990, 2024 + 1)

    key_benchmark = "y_base"

    keys = [
        ("nn_comb", "nn_comb"),
        ("xgb_cv_y_hat", "xgb_cv"),
        ("rf_cv_y_hat", "rf_cv"),
        ("nonlin_comb", "nonlin_comb"),
        ("pca_y_hat", "pca"),
        ("enet_y_hat", "enet"),
        ("lin_comb", "lin_comb"),
        ("all_comb", "all_comb"),
    ]

    for key_a, key_b in keys:

        fig, axs = plt.subplots(1, 3, figsize=(7.2, 5))
        axs = axs.flatten()

        bg_color = "none"  # "#EAEAF2"
        rec_label_bg_color = "none"

        for ax, h in zip(axs, [1, 6, 12]):

            (pbsv_rmse, pbsv_rmse_years), oshapley, oshapley_vi, ishapley_vi = pickle.load(
                open("../../Results/Updated CPI 3/shapleys_h%i_upd.bin" % h, "rb")
            )

            y_key = "y_h%i" % h
            # idx_years = ys[h][y_key].index.map(lambda x: x.year)
            # idx_masks = [idx_years == x for x in years]

            # y_base = average base value of the two neural nets per period. It is shipped as a
            # small CSV (extracted once from ishapley_h*_upd.bin by tools/extract_y_base.py);
            # the multi-GB ishapley bin is only needed if the CSV is absent.
            y_base_csv = "../../Results/Updated CPI 3/y_base_nn_h%i.csv" % h
            if os.path.exists(y_base_csv):
                y_base = pd.read_csv(y_base_csv, index_col=0, float_precision="round_trip")["y_base"]
                if list(y_base.index.astype(str)) != list(ys[h].index.astype(str)):
                    raise ValueError("y_base_nn_h%i.csv does not align with ys_upd.bin" % h)
                ys[h]["y_base"] = y_base.values
            else:
                A = pickle.load(open("../../Results/Updated CPI 3/ishapley_h%i_upd.bin" % h, "rb"))
                ys[h]["y_base"] = pd.Series([
                    (A["nn_deep"][i]["base_value"][0] + A["nn_shallow"][i]["base_value"][0]) / 2
                    for i in range(ys[h].shape[0])
                ], index=ys[h].index)

            if 1 == 1:
                se = lambda x, y: np.sum((x - y) ** 2)
                rmse = lambda x, y: np.sqrt(np.mean((x - y) ** 2))
                cum_diffs = pd.Series([
                    se(ys[h][y_key][ys[h][y_key].index.map(lambda x: x.year <= year)],
                        ys[h][key_benchmark][ys[h][y_key].index.map(lambda x: x.year <= year)]) -
                    se(ys[h][y_key][ys[h][y_key].index.map(lambda x: x.year <= year)],
                        ys[h][key_a][ys[h][y_key].index.map(lambda x: x.year <= year)])
                    for year in years
                ], index=pd.to_datetime(["%i-07-01" % x for x in years]))
                diffs = pd.Series(
                    np.hstack((cum_diffs.iloc[0], np.diff(cum_diffs))),
                    index=pd.to_datetime(["%i-07-01" % x for x in years])
                )
            else:
                se = lambda x, y: (x - y) ** 2
                mse = lambda x, y: np.mean((x - y) ** 2)
                rmse = lambda x, y: np.sqrt(np.mean((x - y) ** 2))
                diffs = pd.Series([
                    se(ys[h][y_key][idx_masks[i]], ys[h][key_benchmark][idx_masks[i]]) -
                    se(ys[h][y_key][idx_masks[i]], ys[h][key_a][idx_masks[i]])
                    for i in range(years.shape[0])
                ], index=pd.to_datetime(["%i-07-01" % x for x in years]))
                cum_diffs = diffs.cumsum()

            def _f(x, y=0):
                return x.replace("fred_", "")
                # return x.replace("fred_", "") + " (%.0f%%)" % y + " " * 5

            rec_years = rec_shade_yearly(ax, years[0], years[-1])

            ax.plot(
                [x for _, x in cum_diffs.iteritems()],
                [x for x, _ in cum_diffs.iteritems()],
                c="#2b2b2b", markerfacecolor="k", lw=1, marker=".", ms=6
            )  # lw=.5 , mew=.75, markerfacecolor="none", c="k", marker="o", ms=3

            # ax.axvline(0, c="#9D9DA6", ls="-", lw=.5)
            ax.axvline(0, c="k", ls="-", lw=.5, alpha=0.25)

            import matplotlib.dates as mdates

            fs = 6.5
            top_n = 1
            padding = paddings[key_benchmark][key_b][h]

            for i, date in enumerate(diffs.index):

                # win_vals = se_decomp[h][key_b][idx_masks[i]].drop("base_value", axis=1).mean(axis=0)
                win_vals = pbsv_rmse_years[date.year].loc[key_b].drop("base_contribution")

                if diffs[date] > 0:
                    if win_vals.loc[win_vals.idxmin()] < 0:
                        ax.text(
                            cum_diffs[date] + padding, mdates.date2num(date),
                            _f("\n".join(win_vals.sort_values(ascending=True)[:min((win_vals < 0).sum(), top_n)].index), 1),
                            bbox=dict(boxstyle="square,pad=0.1",
                                      fc=bg_color if date.year not in rec_years else rec_label_bg_color, ec="none"),
                            color="#009400", rotation=0, ha="left", va="center", fontsize=fs, weight="semibold"
                        )
                    if win_vals.loc[win_vals.idxmax()] > 0:
                        ax.text(
                            cum_diffs[date] - padding, mdates.date2num(date),
                            _f("\n".join(win_vals.sort_values(ascending=False)[:min((win_vals > 0).sum(), top_n)].index),
                               0),
                            bbox=dict(boxstyle="square,pad=0.1",
                                      fc=bg_color if date.year not in rec_years else rec_label_bg_color, ec="none"),
                            color="#808080" if date.year not in rec_years else "#6C6C6C",
                            rotation=0, ha="right", va="center", fontsize=fs
                        )
                elif diffs[date] < 0:
                    if win_vals.loc[win_vals.idxmax()] > 0:
                        ax.text(
                            cum_diffs[date] - padding, mdates.date2num(date),
                            _f("\n".join(win_vals.sort_values(ascending=False)[:min((win_vals > 0).sum(), top_n)].index),
                               0),
                            bbox=dict(boxstyle="square,pad=0.1",
                                      fc=bg_color if date.year not in rec_years else rec_label_bg_color, ec="none"),
                            color="#FF0000", rotation=0, ha="right", va="center", fontsize=fs, weight="semibold"
                        )
                    if win_vals.loc[win_vals.idxmin()] < 0:
                        ax.text(
                            cum_diffs[date] + padding, mdates.date2num(date),
                            _f("\n".join(win_vals.sort_values(ascending=True)[:min((win_vals < 0).sum(), top_n)].index), 1),
                            bbox=dict(boxstyle="square,pad=0.1",
                                      fc=bg_color if date.year not in rec_years else rec_label_bg_color, ec="none"),
                            color="#808080" if date.year not in rec_years else "#6C6C6C",
                            rotation=0, ha="left", va="center", fontsize=fs
                        )
                else:
                    pass

            ax.yaxis.set_major_locator(mdates.YearLocator(base=2, month=7))
            ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y" if h == 1 else ""))

            ax.yaxis.set_minor_locator(mdates.YearLocator(base=1, month=7))
            ax.yaxis.set_minor_formatter(mdates.DateFormatter(""))

            if h == 1:
                ax.yaxis.set_ticks(ax.get_yticks().tolist()[:-1])
                # ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels()[:-1])

            # plt.yticks(cum_diffs.index, years)

            # ax.grid(which="minor", visible=True, color="w")
            # ax.grid(which="major", axis="x", visible=False)

            ax.text(1 - 0.05, 0.965, r"$h=%i$" % h, weight="medium", ha="right", va="center", fontsize=10,
                    transform=ax.transAxes, alpha=1, bbox=dict(boxstyle="square,pad=0.1", fc=bg_color, ec="none"))

            if h == 6:
                ax.text(
                    0.025, 1 - 0.985, r"← underperformance", weight="semibold", ha="left", va="center",
                    fontsize=fs, transform=ax.transAxes, alpha=.6,
                    bbox=dict(boxstyle="square,pad=0.1", fc=bg_color, ec="none")
                )
                ax.text(
                    1 - 0.025, 1 - 0.985, "outperformance →", weight="semibold", ha="right", va="center",
                    fontsize=fs, transform=ax.transAxes, alpha=.6,
                    bbox=dict(boxstyle="square,pad=0.1", fc=bg_color, ec="none")
                )

            ax.set_xticks(xticks[key_benchmark][key_b][h])
            ax.set_xticklabels(
                ["0" if x == 0 else ("%.4f" % x).replace("0.", ".") for x in xticks[key_benchmark][key_b][h]])

            ax.set_xlim(xlims[key_benchmark][key_b][h])

            if h > 1:
                ax.yaxis.set_visible(False)

            ax.invert_yaxis()
            ax.set_ylim(19800+700, 7150)

        sns.despine()
        plt.subplots_adjust(left=.06, bottom=.05, right=.999, top=.999, wspace=0, hspace=.12)
        plt.show(block=False)

        print(key_b)

        plt.savefig(
            "../../Results/Updated CPI 3/Figures/CUMSSE/cumsse-%s-vs-%s_upd.pdf" %
            (key_b.replace("_", "-", ), "ar" if "ar" in key_benchmark else "base")
        )

        plt.close(fig)


def get_ys():

    # a = "20230203_091414"
    # b = "20230131_161507"
    # c = "20230204_163839"
    # a = "20230504_090454"
    # b = "20230510_082421"
    c = "20241005_091334"

    hs = [1, 3, 6, 12] # 12, 3, 1

    res = {}
    for h in hs:
        y = None
        y_base = None
        y_hats = []

        for f in [c,]: #  b, c
            p = os.path.join(MODELS_DIR, f, "cpiaucsl_h%i.bin" % h)
            if os.path.exists(p):
                d = pickle.load(open(p, "rb"))
                if y is None:
                    y = pd.DataFrame(pd.concat([x["test"]["y_h%i" % h] for x in d]))
                    y_base = pd.Series([x["train"]["y_h%i" % h].mean() for x in d], index=y.index).rename("y_base")
                y_hats.append(pd.DataFrame(pd.concat([pd.concat([x[0] for x in z["y_hats"].values()], axis=1) for z in d])))

        y_hats = pd.concat([y, y_base] + y_hats, axis=1)
        y_hats.index += pd.DateOffset(months=h)

        y_hats["nn_comb"] = y_hats[["nn_deep_med_y_hat", "nn_shallow_med_y_hat"]].mean(axis=1)
        y_hats["nonlin_comb"] = y_hats[["rf_cv_y_hat", "xgb_cv_y_hat", "nn_deep_med_y_hat", "nn_shallow_med_y_hat"]].mean(axis=1)
        y_hats["lin_comb"] = y_hats[["enet_y_hat", "pca_y_hat"]].mean(axis=1)
        y_hats["all_comb"] = y_hats[["enet_y_hat", "pca_y_hat", "rf_cv_y_hat", "xgb_cv_y_hat", "nn_deep_med_y_hat", "nn_shallow_med_y_hat"]].mean(axis=1)

        res[h] = y_hats

    pickle.dump(res, open("../../Results/Updated CPI 3/ys_upd.bin", "wb"))


def normed_rank_diffs_tlb(weighted=True):
    suffix_a, suffix_b = "_w" if weighted else "_uw", "_w" if weighted else ""
    table_keys = ["pca", "enet", "rf_cv", "xgb_cv", "nn_comb", "lin_comb", "nonlin_comb", "all_comb"]
    cols_is = []
    cols_os = []
    cols_isos = []
    cols_is_pval = []
    cols_os_pval = []
    cols_isos_pval = []
    for h in [1, 3, 6, 12]:
        smsdr = rankdiffs(h=h, n_mc_sims=1000000)
        cols_is_pval.append(smsdr["ishapley_vs_pbsv_msdr%s_pval" % suffix_a].rename(h))
        cols_os_pval.append(smsdr["oshapley_vs_pbsv_msdr%s_pval" % suffix_a].rename(h))
        cols_isos_pval.append(smsdr["ishapley_vs_oshapley_msdr%s_pval" % suffix_a].rename(h))
        cols_is.append(smsdr["ishapley_vs_pbsv_msdr_rnd_norm%s" % suffix_b].rename(h))
        cols_os.append(smsdr["oshapley_vs_pbsv_msdr_rnd_norm%s" % suffix_b].rename(h))
        cols_isos.append(smsdr["ishapley_vs_oshapley_msdr_rnd_norm%s" % suffix_b].rename(h))
    smsdrs_is_pval = pd.concat(cols_is_pval, axis=1).loc[table_keys]
    smsdrs_os_pval = pd.concat(cols_os_pval, axis=1).loc[table_keys]
    smsdrs_isos_pval = pd.concat(cols_isos_pval, axis=1).loc[table_keys]
    smsdrs_is = pd.concat(cols_is, axis=1).loc[table_keys]
    smsdrs_os = pd.concat(cols_os, axis=1).loc[table_keys]
    smsdrs_isos = pd.concat(cols_isos, axis=1).loc[table_keys]
    inv_smsdrs_is = 1 - smsdrs_is
    inv_smsdrs_os = 1 - smsdrs_os
    inv_smsdrs_isos = 1 - smsdrs_isos
    inv_smsdrs_is["avg"] = inv_smsdrs_is.mean(axis=1)
    inv_smsdrs_os["avg"] = inv_smsdrs_os.mean(axis=1)
    inv_smsdrs_isos["avg"] = inv_smsdrs_isos.mean(axis=1)
    breakpoint()


def normed_rank_diffs_tlb2(alpha):
    table_keys = ["pca", "enet", "rf_cv", "xgb_cv", "nn_comb", "lin_comb", "nonlin_comb", "all_comb"]
    cols_is = []
    cols_os = []
    cols_is_pval = []
    cols_os_pval = []
    for h in [1,3,6,12]:
        # smsdr = rankdiffs2(h=h, a=alpha, n_mc_sims=1000000, override=True)
        smsdr = rankdiffs2_w(h=h, a=alpha, n_mc_sims=1000000, override=False)
        cols_is_pval.append(smsdr["ishapley_vs_pbsv_mas_rnd_norm_p"].rename(h))
        # cols_os_pval.append(smsdr["oshapley_vs_pbsv_mas_rnd_norm_p"].rename(h))
        cols_is.append(smsdr["ishapley_vs_pbsv_mas_rnd_norm"].rename(h))
        # cols_os.append(smsdr["oshapley_vs_pbsv_mas_rnd_norm"].rename(h))
    smsdrs_is_pval = pd.concat(cols_is_pval, axis=1).loc[table_keys]
    # smsdrs_os_pval = pd.concat(cols_os_pval, axis=1).loc[table_keys]
    smsdrs_is = pd.concat(cols_is, axis=1).loc[table_keys]
    # smsdrs_os = pd.concat(cols_os, axis=1).loc[table_keys]
    smsdrs_is["avg"] = smsdrs_is.mean(axis=1)
    # smsdrs_os["avg"] = smsdrs_os.mean(axis=1)

    smsdrs_is.to_excel("../../Results/Updated CPI 3/MAS/wmas_a%.2f.xlsx" % (alpha,))
    smsdrs_is_pval.to_excel("../../Results/Updated CPI 3/MAS/wmas_pval_a%.2f.xlsx" % (alpha,))

    breakpoint()


def quadrant_plot_w():
    perf = pd.read_pickle("../../Results/Updated CPI 3/perf_raw.pickle")

    model_keys_a = [
        "pca_y_hat", "enet_y_hat", "xgb_cv_y_hat", "rf_cv_y_hat", "nn_comb", "lin_comb", "nonlin_comb", "all_comb"
    ]

    model_keys_b = [
        "pca", "enet", "xgb_cv", "rf_cv", "nn_comb", "lin_comb", "nonlin_comb", "all_comb"
    ]

    sns.set_style('whitegrid')
    sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 1.5, "grid.linewidth": 0.5})

    hs = [1, 3, 6, 12]
    ms = ['o', '^', 'X', 's']

    marker = "o"

    names = {
        "rf_cv": "RF",
        "all_comb": "Ens-all",
        "lin_comb": "Ens-linear",
        "nonlin_comb": "Ens-nonlinear",
        "pca": "PCR",
        "enet": "ENet",
        "xgb_cv": "XGB",
        "nn_comb": "NN",
    }

    alpha = 2/3

    smsdrs_is = pd.read_excel("../../Results/Updated CPI 3/MAS/wmas_a%.2f.xlsx" % (alpha,), engine='openpyxl', index_col=0)

    dfs = []
    for h in hs:
        df_rr = perf.loc[h]["rmse"][model_keys_a]
        df_rr.index = model_keys_b
        df_rr_n = (df_rr - df_rr.mean()) / df_rr.std()

        df_wmas = smsdrs_is[h].loc[model_keys_b]
        df_wmas_n = (df_wmas - df_wmas.mean()) / df_wmas.std()

        dfs.append(pd.concat((df_rr_n.rename("rr"), df_wmas_n.rename("wmas")), axis=1))

    dfs = pd.concat(dfs, keys=hs)

    fig, ax = plt.subplots(figsize=(7, 6))
    # ax.text(.75, .75, 'intentional success', zorder=-5, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='gray', alpha=0.5, bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
    # ax.text(.75, .25, 'unintentional success', zorder=-5, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='gray', alpha=0.5, bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
    for h, m in zip(hs, ms):
        df = dfs.xs(h, level=0)
        ax.scatter(df["rr"], df["wmas"], label="h = %i" % h, marker=marker, s=50, edgecolor='black', facecolor='w')
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    ax.grid(False)
    ax.axvline(0, color='black', linestyle='-', zorder=-10)
    ax.axhline(0, color='black', linestyle='-', zorder=-10)
    ax.set_ylabel("MAS (Z-score)")
    ax.set_xlabel("RMSE (Z-score)")
    plt.tight_layout()
    ax.invert_xaxis()
    plt.savefig("../../Results/Updated CPI 3/Figures/MAS/quadrant_plot_all_single_mas2_w.pdf")
    plt.close()

    def is_pareto_optimal(row, df):
            for _, other_row in df.iterrows():
                if (
                        (other_row['rr'] < row['rr'] and other_row['wmas'] >= row['wmas']) or
                        (other_row['wmas'] > row['wmas'] and other_row['rr'] <= row['rr'])
                ):
                    return False
            return True

    fig, axs = plt.subplots(2, 2, figsize=(8.5, 6.5))

    i = 0
    for ax, h, m in zip(axs.flatten(), hs, ms):
        ax.set_title(f'Panel {chr(65 + i)} ($h={h}$)', loc="left")
        # ax.text(.025, .97, f'$h={h}$', fontsize=8, ha='left', va='top', transform=ax.transAxes,
        #         bbox=dict(boxstyle='square,pad=0.2', facecolor='white', linewidth=0)
        # )
        i += 1
        df = dfs.xs(h, level=0)
        pareto_m = df.apply(is_pareto_optimal, axis=1, df=df)
        upper_right_m = (df["rr"] < 0) & (df["wmas"] > 0)
        dominating_m = pareto_m & upper_right_m
        c = pd.concat((pareto_m, upper_right_m), axis=1).apply(lambda x: "black" if x[0] and x[1] else "lightgray" if x[0] else "white", axis=1)
        ax.scatter(df["rr"], df["wmas"], label="h = %i" % h, marker=marker, s=50, edgecolor='black', facecolor=c)
        pareto_line_df = df[pareto_m].sort_values(by='rr')
        pl_rr, pl_wmas = pareto_line_df['rr'].to_list(), pareto_line_df['wmas'].to_list()
        # pl_rr = [-100] + pl_rr + [pl_rr[-1]]
        # pl_wmas = [pl_wmas[0]] + pl_wmas + [-100]
        ax.step(pl_rr, pl_wmas, where='post', color='black', alpha=0.7, ls="-", lw=0.5, zorder=0)
        ax.plot([100, pl_rr[-1]], pl_wmas[-1:]*2, color="black", alpha=0.2, ls="-", lw=0.5, zorder=0)
        ax.plot(pl_rr[:1]*2, [pl_wmas[0], -100], color="black", alpha=0.2, ls="-", lw=0.5, zorder=0)

        for _, row in df.iterrows():
            offset = (
                (-4, -6) if h == 1 and row.name == "nn_comb" else
                (0, -10) if h == 1 and row.name == "enet" else
                (-30, 10) if h == 1 and row.name == "lin_comb" else
                (38, 0) if h == 1 and row.name == "nonlin_comb" else
                (37, 5) if h == 3 and row.name == "nonlin_comb" else
                (0, -8) if h == 3 and row.name == "lin_comb" else
                (0, -8) if h == 3 and row.name == "nn_comb" else
                (0, -8) if h == 6 and row.name == "nonlin_comb" else
                (5, 17) if h == 6 and row.name == "all_comb" else
                (0, -8) if h == 6 and row.name == "rf_cv" else
                (0, -8) if h == 12 and row.name == "rf_cv" else
                (0, -8) if h == 12 and row.name == "nn_comb" else
                (0, -8) if h == 12 and row.name == "nonlin_comb" else
                (0, -8) if h == 12 and row.name == "xgb_cv" else
                (0, -8) if h == 12 and row.name == "pca" else
                (0, -8) if h == 12 and row.name == "enet" else
                # (20, -4) if h == 1 and row.name == "enet" else
                # (-42, 15) if h == 1 and row.name == "nonlin_comb" else
                (0, 15)
            )
            ax.annotate(names[row.name], (row['rr'], row['wmas']),
                        va="top", ha="center", textcoords="offset points", xytext=offset,
                        bbox=dict(boxstyle='square,pad=0.1', facecolor='white', linewidth=0)
                        )

        ax.grid(False)

        # ax.legend(fancybox=False, frameon=False)

        ax.axvline(0, color='black', linestyle='-', zorder=-10)
        ax.axhline(0, color='black', linestyle='-', zorder=-10)

        ax.set_ylabel("MAS (Z-score)", fontsize=9.5)
        ax.set_xlabel("RMSE (Z-score)", fontsize=9.5)

        mag_x = np.abs([df["rr"].min(), df["rr"].max()]).max() * 1.25
        mag_y = np.abs([df["wmas"].min(), df["wmas"].max()]).max() * 1.25

        ax.set_ylim(-mag_y, mag_y)
        ax.set_xlim(-mag_x, mag_x)
        ax.invert_xaxis()

        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

    plt.show(block=False)
    plt.subplots_adjust(0.07, 0.08, 0.99, 0.95, 0.25, 0.35)
    plt.savefig("../../Results/Updated CPI 3/Figures/MAS/quadrant_plot_all_mas2_w.pdf")
    plt.close()


def quadrant_plot_w_old():
    perf = pd.read_pickle("../../Results/Updated CPI 3/perf_raw.pickle")

    model_keys_a = [
        "pca_y_hat", "enet_y_hat", "xgb_cv_y_hat", "rf_cv_y_hat", "nn_comb", "lin_comb", "nonlin_comb", "all_comb"
    ]

    model_keys_b = [
        "pca", "enet", "xgb_cv", "rf_cv", "nn_comb", "lin_comb", "nonlin_comb", "all_comb"
    ]

    sns.set_style('whitegrid')
    sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 1.5, "grid.linewidth": 0.5})

    hs = [1, 3, 6, 12]
    ms = ['o', '^', 'X', 's']

    marker = "o"

    names = {
        "rf_cv": "RF",
        "all_comb": "Ens-all",
        "lin_comb": "Ens-linear",
        "nonlin_comb": "Ens-nonlinear",
        "pca": "PCR",
        "enet": "ENet",
        "xgb_cv": "XGB",
        "nn_comb": "NN",
    }

    dfs = []
    for h in hs:
        df_rr = perf.loc[h]["rmse"][model_keys_a]
        df_rr.index = model_keys_b

        # df_rr_i_n = (df_rr_i - df_rr_i.mean()) / np.max(np.abs(df_rr_i - df_rr_i.mean()))
        df_rr_n = (df_rr - df_rr.mean()) / df_rr.std()

        df_wmas = rankdiffs2_w(h=h, a=0.5, n_mc_sims=10000)["ishapley_vs_pbsv_mas_rnd_norm"][model_keys_b]
        # df_wmas_n = ((df_wmas - df_wmas.min()) / (df_wmas.max() - df_wmas.min())) * 2 - 1
        # df_wmas_n = (df_wmas - df_wmas.mean()) / np.max(np.abs(df_wmas - df_wmas.mean()))
        df_wmas_n = (df_wmas - df_wmas.mean()) / df_wmas.std()

        dfs.append(pd.concat((df_rr_n.rename("rr"), df_wmas_n.rename("wmas")), axis=1))

    dfs = pd.concat(dfs, keys=hs)

    fig, ax = plt.subplots(figsize=(7, 6))
    # ax.text(.75, .75, 'intentional success', zorder=-5, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='gray', alpha=0.5, bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
    # ax.text(.75, .25, 'unintentional success', zorder=-5, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='gray', alpha=0.5, bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
    for h, m in zip(hs, ms):
        df = dfs.xs(h, level=0)
        ax.scatter(df["rr"], df["wmas"], label="h = %i" % h, marker=marker, s=50, edgecolor='black', facecolor='w')
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    ax.grid(False)
    ax.axvline(0, color='black', linestyle='-', zorder=-10)
    ax.axhline(0, color='black', linestyle='-', zorder=-10)
    ax.set_ylabel("MAS (Z-score)")
    ax.set_xlabel("RMSE (Z-score)")
    plt.tight_layout()
    ax.invert_xaxis()
    plt.savefig("../../Results/Updated CPI 3/Figures/MAS/quadrant_plot_all_single_mas2_w.pdf")
    plt.close()

    def is_pareto_optimal(row, df):
            for _, other_row in df.iterrows():
                if (
                        (other_row['rr'] < row['rr'] and other_row['wmas'] >= row['wmas']) or
                        (other_row['wmas'] > row['wmas'] and other_row['rr'] <= row['rr'])
                ):
                    return False
            return True

    fig, axs = plt.subplots(2, 2, figsize=(8.5, 6.5))

    i = 0
    for ax, h, m in zip(axs.flatten(), hs, ms):
        ax.set_title(f'Panel {chr(65 + i)}', loc="left")
        # ax.text(.025, .97, f'Panel {chr(65 + i)}', fontsize=11, ha='left', va='top', transform=ax.transAxes,
                # bbox=dict(boxstyle='square,pad=0.2', facecolor='white', linewidth=0)
        # )
        i += 1
        df = dfs.xs(h, level=0)
        pareto_m = df.apply(is_pareto_optimal, axis=1, df=df)
        upper_right_m = (df["rr"] < 0) & (df["wmas"] > 0)
        dominating_m = pareto_m & upper_right_m
        c = pd.concat((pareto_m, upper_right_m), axis=1).apply(lambda x: "black" if x[0] and x[1] else "lightgray" if x[0] else "white", axis=1)
        ax.scatter(df["rr"], df["wmas"], label="h = %i" % h, marker=marker, s=50, edgecolor='black', facecolor=c)
        pareto_line_df = df[pareto_m].sort_values(by='rr')
        pl_rr, pl_wmas = pareto_line_df['rr'].to_list(), pareto_line_df['wmas'].to_list()
        # pl_rr = [-100] + pl_rr + [pl_rr[-1]]
        # pl_wmas = [pl_wmas[0]] + pl_wmas + [-100]
        ax.step(pl_rr, pl_wmas, where='post', color='black', alpha=0.7, ls="-", lw=0.5, zorder=0)
        ax.plot([100, pl_rr[-1]], pl_wmas[-1:]*2, color="black", alpha=0.2, ls="-", lw=0.5, zorder=0)
        ax.plot(pl_rr[:1]*2, [pl_wmas[0], -100], color="black", alpha=0.2, ls="-", lw=0.5, zorder=0)

        for _, row in df.iterrows():
            offset = (
                (0, -12) if h == 1 and row.name == "enet" else
                (0, 20) if h == 1 and row.name == "nn_comb" else
                (-30, 12) if h == 1 and row.name == "lin_comb" else
                (38, 4) if h == 1 and row.name == "nonlin_comb" else
                (0, -10) if h == 1 and row.name == "pca" else
                (0, -12) if h == 1 and row.name == "rf_cv" else
                (0, -12) if h == 3 and row.name == "nn_comb" else
                (38, 4) if h == 3 and row.name == "nonlin_comb" else
                # (-28, 5) if h == 3 and row.name == "all_comb" else
                # (-8, 20) if h == 6 and row.name == "pca" else
                # (8, -12) if h == 6 and row.name == "enet" else
                (0, -12) if h == 6 and row.name == "nonlin_comb" else
                # (5, 17) if h == 6 and row.name == "all_comb" else
                (0, -12) if h == 6 and row.name == "rf_cv" else
                (0, -10) if h == 6 and row.name == "xgb_cv" else
                (0, -12) if h == 12 and row.name == "rf_cv" else
                (0, -12) if h == 12 and row.name == "nn_comb" else
                (0, -12) if h == 12 and row.name == "nonlin_comb" else
                (0, -10) if h == 12 and row.name == "lin_comb" else
                (0, -12) if h == 12 and row.name == "enet" else
                # (20, -4) if h == 1 and row.name == "enet" else
                # (-42, 15) if h == 1 and row.name == "nonlin_comb" else
                (0, 20)
            )
            ax.annotate(names[row.name], (row['rr'], row['wmas']),
                        va="top", ha="center", textcoords="offset points", xytext=offset,
                        bbox=dict(boxstyle='square,pad=0.1', facecolor='white', linewidth=0)
                        )

        ax.grid(False)

        # ax.legend(fancybox=False, frameon=False)

        ax.axvline(0, color='black', linestyle='-', zorder=-10)
        ax.axhline(0, color='black', linestyle='-', zorder=-10)

        ax.set_ylabel("MAS (Z-score)", fontsize=9.5)
        ax.set_xlabel("RMSE (Z-score)", fontsize=9.5)

        mag_x = np.abs([df["rr"].min(), df["rr"].max()]).max() * 1.25
        mag_y = np.abs([df["wmas"].min(), df["wmas"].max()]).max() * 1.25

        ax.set_ylim(-mag_y, mag_y)
        ax.set_xlim(-mag_x, mag_x)
        ax.invert_xaxis()

        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

    plt.show(block=False)
    plt.subplots_adjust(0.07, 0.08, 0.99, 0.95, 0.25, 0.35)
    plt.savefig("../../Results/Updated CPI 3/Figures/MAS/quadrant_plot_all_mas2_w.pdf")
    plt.close()


def explain_expanding(
    self: Anatomy,
    model_sets: Union[AnatomyModelCombination, None] = None,
    transformer: Union[AnatomyModelOutputTransformer, None] = None,
    explanation_subset: Union[pd.Index, None] = None
) -> pd.DataFrame:
    
    assert self._Y is not None

    if model_sets is None:
        model_sets = AnatomyModelCombination(groups={x: [x] for x in self._model_names})

    if transformer is None:
        transformer = AnatomyModelOutputTransformer(transform=lambda y_hat: y_hat)

    if explanation_subset is None:
        explanation_subset = self._xy_test.index
    subset = self._xy_test.index.isin(explanation_subset)
    
    assert subset.any()

    def _apply(comb_set: Union[list[str], dict[str, float]]):

        models = comb_set if type(comb_set == list) else comb_set.keys()
        weights = np.repeat(1.0, len(comb_set)) if type(comb_set == list) else np.array(comb_set.values())

        model_indices = [self._model_names.index(x) for x in models]

        if "y" in transformer.transform.__code__.co_varnames:
            transform = lambda y_hat: transformer.transform(y_hat=y_hat, y=self._xy_test[subset][self._y_name].to_numpy())
        else:
            transform = lambda y_hat: transformer.transform(y_hat=y_hat)

        Y_combination = np.average(self._Y[model_indices][:, subset, :, :, :], weights=weights, axis=0)
        Y = np.apply_along_axis(transform, axis=0, arr=Y_combination)

        assert len(Y.shape) in [3, 4]  # 3 => AGGREGATED (GLOBAL) ; 4 => LOCAL
        explanation_level = Anatomy.ExplanationLevel.LOCAL if len(Y.shape) == 4 else Anatomy.ExplanationLevel.GLOBAL

        if explanation_level == Anatomy.ExplanationLevel.GLOBAL:
            Y = Y.reshape(1, *Y.shape)

        marginal_contributions = np.diff(Y, axis=1)

        Phi = np.zeros_like(marginal_contributions)
        for m in range(Phi.shape[-2]):
            Phi[:, self._permutations[:, m], m, 0] = marginal_contributions[:, :, m, 0]
            Phi[:, self._permutations[:, m][::-1], m, 1] = marginal_contributions[:, :, m, 1]

        PhiAll = Phi.reshape((*Phi.shape[:2], -1))

        ExpandingPhi = np.cumsum(PhiAll[0], axis=1) / np.tile(np.arange(1, PhiAll.shape[-1]+1), (PhiAll.shape[1], 1))
        
        return ExpandingPhi

    data = { k: _apply(v) for k, v in model_sets.groups.items() }

    return data


def anatomize_convergence(h):

    print(h)

    anatomy = Anatomy.load("../../Results/Updated CPI 3/anatomy_h%i_upd.bin" % h)

    groups = {
        "rf_cv": ["rf_cv"],
        "xgb_cv": ["xgb_cv"],
        "nn_comb": ["nn_deep", "nn_shallow"],
        "enet": ["enet"],
        "pca": ["pca"],
        "lin_comb": ["enet", "pca"],
        "nonlin_comb": ["rf_cv", "xgb_cv", "nn_deep", "nn_shallow"],
        "all_comb": ["enet", "pca", "rf_cv", "xgb_cv", "nn_deep", "nn_shallow"]
    }

    def transform(y_hat, y):
        return np.sqrt(np.mean((y - y_hat) ** 2))

    rmse_expandings = explain_expanding(
        self=anatomy,
        model_sets=AnatomyModelCombination(groups=groups),
        transformer=AnatomyModelOutputTransformer(transform=transform)
    )

    pickle.dump(rmse_expandings, open("../../Results/Updated CPI 3/pbsv_convergence_h%i.bin" % h, "wb"))


def convergence():

    sns.set_theme()
    sns.set_context("paper", font_scale=1.1, rc={"font.family": "Arial"})

    keep_top_n = 50

    hs = [1, 3, 6, 12]    
    group_keys = ["pca", "enet", "lin_comb", "nn_comb", "rf_cv", "xgb_cv", "nonlin_comb", "all_comb"]

    for group_key in group_keys:

        fig, axs = plt.subplots(4, 1, figsize=(7.2, 6))

        for i, h in enumerate(hs):

            ax = axs[i]

            d = pickle.load(open("../../Results/Updated CPI 3/pbsv_convergence_h%i.bin" % h, "rb"))
            
            rmse_expanding = d[group_key]

            most_important = np.argsort(np.abs(rmse_expanding.T[-1]))[::-1][:keep_top_n] # top n abs(pbsv)

            ax.plot(np.arange(2, 200+1), np.abs(rmse_expanding.T[1:, most_important]))

            ax.set_yscale('log')
            

            ax.text(
                0.985, 0.07, r"$h=%i$" % h,
                horizontalalignment="right", transform=ax.transAxes, fontsize=10
            )

            ax.set_yticklabels([])

            if h != 12:
                ax.set_xticklabels([])

            ax.set_xticks([2, 25, 50, 75, 100, 125, 150, 175, 200])

        plt.subplots_adjust(left=.01, bottom=.05, right=.99, top=.99, wspace=.1, hspace=.1)

        plt.savefig(
            "../../Results/Updated CPI 3/Figures/Convergence/convergence-%s-absrmse-top%i.pdf" % (
                group_key, keep_top_n
            )
        )


if __name__ == "__main__":
    quadrant_plot_w()
    exit(-1)

    convergence()
    exit(-1)

    anatomize_convergence(h=1)
    anatomize_convergence(h=3)
    anatomize_convergence(h=6)
    anatomize_convergence(h=12)

    exit(-1)

    xxx = pickle.load(open("../../Results/Updated CPI 3/ys_upd.bin", "rb"))

    exit(-1)

    normed_rank_diffs_tlb2(alpha=2/3)

    exit(-1)

    rankdiffs2_w(h=1, a=2/3, override=True)
    rankdiffs2_w(h=3, a=2/3, override=True)
    rankdiffs2_w(h=6, a=2/3, override=True)
    rankdiffs2_w(h=12, a=2/3, override=True)

    exit(-1)

    plot_cumsse()
    exit(-1)

    get_ys()
    exit(-1)

    quadrant_plot_w()
    exit(-1)

    plot_pbsv_vs_isvi()
    exit(-1)

    anatomize(h=1)
    anatomize(h=3)
    anatomize(h=6)
    anatomize(h=12)

    exit(-1)

    breakpoint()

    (pbsv_rmse, _), oshapley, oshapley_vi, ishapley_vi = pickle.load(
        open("../../Results/Updated CPI 2/shapleys_h%i_upd.bin" % 1, "rb")
    )

    breakpoint()

    # plot_pbsv_vs_isvi()

    # smsdr = rankdiffs2_w(h=3, a=2/3, override=True)
    # quadrant_plot_w()
    # normed_rank_diffs_tlb2(alpha=2/3)
    breakpoint()
    exit(-1)

    # plot_cumsse()
    smsdr = rankdiffs(h=12)
    # get_ys()
    # plot_cumsse()
    # estimate(h=12)
    # estimate(h=6)
    # estimate(h=3)
    # estimate(h=1)
    # anatomize(h=1)
    # anatomize(h=3)
    # anatomize(h=6)
    # anatomize(h=12)
    # plot()
    # anatomize(h=1), anatomize(h=3)
    # anatomize(h=6)
    # estimate(h=6)
