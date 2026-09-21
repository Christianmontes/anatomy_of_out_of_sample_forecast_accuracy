###
#  Copyright (C) 2021 Sander Schwenk-Nebbe
#  sandersn@econ.au.dk
###
from itertools import chain
from tqdm import tqdm
import numpy as np


def t(f, x, X, M):

    # number of observations T and features J in training data X:
    T, J = X.shape

    # allocate matrix to hold one SHAP estimate per feature per permutation run:
    Phi = np.zeros((M, J), dtype=np.float64)

    for m in range(M):


        # obtain order in which features are activated (permutation of np.arange(J)):
        J_m = np.random.permutation(J)
        J_m = np.arange(J)

        masks = np.zeros(2*len(J_m), dtype=int)
        masks[0] = 10000
        last_ind = -1
        for i in range(len(J_m)):
            if i > 0:
                masks[2*i] = -last_ind - 1
            masks[2*i+1] = J_m[i]
            last_ind = J_m[i]

        def _convert_delta_mask_to_full(masks, full_masks):
            """ This converts a delta masking array to a full bool masking array.
            """

            i = -1
            masks_pos = 0
            while masks_pos < len(masks):
                i += 1

                if i > 0:
                    full_masks[i] = full_masks[i - 1]

                while masks[masks_pos] < 0:
                    full_masks[i, -masks[masks_pos] - 1] = ~full_masks[
                        i, -masks[masks_pos] - 1]  # -value - 1 is the original index that needs flipped
                    masks_pos += 1

                if masks[masks_pos] != 10000:
                    full_masks[i, masks[masks_pos]] = ~full_masks[i, masks[masks_pos]]
                masks_pos += 1

        full_masks = np.zeros((int(np.sum(masks >= 0)), 100), dtype=np.bool)
        _convert_delta_mask_to_full(masks, full_masks)


        print()


def approx_shap_simple(f, x, X, M):
    """Approximates SHAP values for x.

    Approximate SHAP values are obtained for x in a simple but inefficient way. Yields
    same approximations as "approx_shap(·)" significantly slower but with code that is
    easier to follow. Function is unused and is provided primarily for illustration.

    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        Prediction function of model for which SHAP values are obtained;
        takes as input TxJ matrix with T number of observations to
        predict and J features and outputs 1xT vector of predictions.
    x : np.ndarray
        Observation to explain as vector with J features.
    X : np.ndarray
        Data on which prediction function was trained as TxJ matrix with
        T number of observations and J number of features.
    M : int
        Number of runs in which the J features are activated in random (permuted) order.

    Returns
    -------
    phi : ndarray
        Approximate SHAP values for x as 1xJ vector.
    base_value: float
        Base value E[f(x)].
    """

    # number of observations T and features J in training data X:
    T, J = X.shape

    # allocate matrix to hold one SHAP estimate per feature per permutation run:
    Phi = np.zeros((M, J, 2), dtype=np.float64)

    base_value = None

    for m in range(M):

        # obtain order in which features are activated (permutation of np.arange(J)):
        J_m = np.random.permutation(J)

        for forwards_backwards in [0, 1]:

            # reverse permutation for antithetic sampling:
            if forwards_backwards == 1:
                J_m = J_m[::-1]

            # obtain copy of original training data which is modified in this run:
            X_m = X.copy()

            for j in J_m:
                # obtain average model prediction with feature j deactivated;
                # note that X_m reflects all previously activated features in J_m;
                # at the first run, no features will be activated (yielding E[f(X)]);
                # at the second run, feature J_m[0] will be activated at this point:
                y_hat_avg_minus_j = f(X_m).mean()
                # activate feature j by replacing column j (for all rows) in the
                # training data (X) with column j of the obs to explain (x):
                X_m[:, j] = x[j]
                # obtain average model prediction, now with feature j activated:
                y_hat_avg_plus_j = f(X_m).mean()
                # calculate marginal contribution of activating feature j in the permuted order:
                Phi[m, j, forwards_backwards] = y_hat_avg_plus_j - y_hat_avg_minus_j
                # store first run with all features deactivated (yields base value / E[f(x)]):
                if base_value is None:
                    base_value = y_hat_avg_minus_j

    # return SHAP value estimates as average over forwards_backwards and all permutation runs M
    return Phi.mean(axis=-1).mean(axis=0), base_value


def exact_shap(f, x, X):
    """Calculates exact SHAP values for x.

    Exact SHAP values are obtained for x from all 2^J ways in which the J features can be activated.

    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        Prediction function of model for which SHAP values are obtained;
        takes as input TxJ matrix with T number of observations to
        predict and J features and outputs 1xT vector of predictions.
    x : np.ndarray
        Observation to explain as vector with J features.
    X : np.ndarray
        Data on which prediction function was trained as TxJ matrix with
        T number of observations and J number of features.

    Returns
    -------
    phi : ndarray
        Exact SHAP values for x as 1xJ vector.
    base_value: float
        Base value E[f(x)].
    """

    # number of observations T and features J in training data X:
    T, J = X.shape

    # allocate vector to hold one SHAP value per feature:
    phi = np.zeros(J, dtype=np.float64)

    # construct the 2^J unique masks of the J features that need be evaluated;
    # obtain the powerset of range(J) and convert it to a (2^J)xJ masking matrix:
    from itertools import chain, combinations
    masks = np.zeros((2 ** J, J), dtype=bool)
    for i, mask in enumerate(chain.from_iterable(combinations(range(J), i) for i in range(J + 1))):
        masks[i, mask] = True

    # repeats the training data 2^J times, once for each masking:
    X_masked = np.tile(X, (2 ** J, 1))
    for i in range(2 ** J):
        # turns on all active features in the unique masking (masks[i]);
        # replaces each active feature in the copy of X with the
        # corresponding value in the observation to explain (x):
        X_masked[T * i:T * (i + 1), masks[i]] = x[masks[i]]

    # evaluate 2^J activations;
    # each activation is evaluated on T obs;
    # thus reshape output to (2^J)xT matrix;
    # take average over all T;
    # this yields the average model output for each activation:
    y_hat_masked_avg = f(X_masked).flatten().reshape(2 ** J, -1).mean(axis=1)

    # store first run with all features deactivated (yields base value / E[f(x)]):
    base_value = y_hat_masked_avg[0]

    # calculate SHAP values for each feature:
    for j in range(J):
        # find and loop through all masks in which feature j is not active:
        for j_minus in np.where(~masks[:, j])[0]:
            # obtain mask in which feature j is active:
            j_plus_mask = masks[j_minus, :].copy()
            j_plus_mask[j] = True
            # find mask in which feature j is active in all unique masks:
            j_plus = np.where((masks == j_plus_mask).all(axis=1))[0][0]
            # obtain number of active features in masking excluding j (|S|):
            j_minus_size = masks[j_minus, :].sum()
            # calculate weight for marginal contribution ([|S|!(p-|S|-1)!]/p!):
            w = (np.math.factorial(j_minus_size) * np.math.factorial(J - j_minus_size - 1)) / np.math.factorial(J)
            # add weighted marginal contribution of activating j to SHAP value for j:
            phi[j] += w * (y_hat_masked_avg[j_plus] - y_hat_masked_avg[j_minus])

    return phi, base_value


def approx_shap(f, x, X, M, l=None):
    """Approximates SHAP values for x.

    Approximate SHAP values are obtained for x. Feature activations for all J
    features are evaluated in one go per permutation run to reduce overhead in
    calling f. Masks invariant to permutation are only evaluated once
    regardless of M.

    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        Prediction function of model for which SHAP values are obtained;
        takes as input TxJ matrix with T number of observations to
        predict and J features and outputs 1xT vector of predictions.
    x : np.ndarray
        Observation to explain as vector with J features.
    X : np.ndarray
        Data on which prediction function was trained as TxJ matrix with
        T number of observations and J number of features.
    M : int
        Number of runs in which the J features are activated in random (permuted) order.

    Returns
    -------
    phi : ndarray
        Approximate SHAP values for x as 1xJ vector.
    base_value: float
        Base value E[f(x)].
    """

    if l is None:
        l = lambda y_hat: y_hat

    # number of observations T and features J in training data X:
    T, J = X.shape

    # allocate matrix to hold one SHAP estimate per feature per permutation (original and reverse):
    Phi = np.zeros((M, J, 2), dtype=np.float64)

    # runs with none and all features activated are invariant to permutation;
    # calculate average model output for the two masks outside loop:
    y_hat_none_all = f(np.vstack((X, x))).flatten()
    y_hat_avg_none, y_hat_avg_all = y_hat_none_all[:T].mean(), y_hat_none_all[T]

    # store first run with all features deactivated (yields base value, aka E[f(x)]):
    base_value = l(y_hat_avg_none)

    for m in range(M):

        # obtain order in which features are activated (permutation of np.arange(J)):
        J_m = np.random.permutation(J)

        for forwards_backwards in [0, 1]:

            # reverse permutation for antithetic sampling:
            if forwards_backwards == 1:
                J_m = J_m[::-1]

            # obtain copy of original training data which is modified in this run:
            X_m = X.copy()

            # allocate matrix to hold all feature activations (excluding none and all active features)
            # to evaluate model once potentially reducing overhead in calling f:
            X_eval_m = np.zeros((T * (J - 1), J), dtype=np.float64)

            # activate feature i in permutation J_m excluding last feature at
            # which all features would be activated (which we already have):
            for i in range(J - 1):
                j = J_m[i]
                # replace all values of column j in X_m with value of column j in
                # row to explain (x) thereby activating feature j in X_m:
                X_m[:, j] = x[j]
                # place current X_m in aggregate evaluation matrix:
                X_eval_m[T * i:T * (i + 1)] = X_m

            # evaluate J-1 activations;
            # each activation is evaluated on T obs;
            # thus reshape output to (J-1)xT matrix;
            # take average over all T;
            # this yields the average model output for each activation:
            y_hat_avg_m = f(X_eval_m).flatten().reshape(J - 1, -1).mean(axis=1)

            # add first masking (no active features) and last masking (all features active):
            y_hat_avg_m = np.hstack((y_hat_avg_none, y_hat_avg_m, y_hat_avg_all))

            # at index i+1 variable J_m[i+1] is activated in y_hat_avg_m
            # marginal contribution is obtained as y_hat_avg_m[i+1] - y_hat_avg_m[i]
            # np.diff(y_hat_avg_m) yields the marginal contributions for J_m
            # store results in that order (J_m) in Phi:
            Phi[m, J_m, forwards_backwards] = np.diff(l(y_hat_avg_m))

    # return SHAP value estimates as average over forwards_backwards and all permutation runs M
    return Phi.mean(axis=-1).mean(axis=0), base_value


def approx_shap_unique(f, x, X, M):
    """Approximates SHAP values for x ensuring no masks are evaluated twice.

    Approximate SHAP values are obtained for x. Generates M permuted orders in which
    the J features are activated and evaluates only the unique masks. This can be
    more efficient than "approx_shap(·)" when duplicated masks are expected among
    the M permutation runs (contingent on J; 2^J unique masks exist), but the exact
    algorithm is not feasible.

    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        Prediction function of model for which SHAP values are obtained;
        takes as input TxJ matrix with T number of observations to
        predict and J features and outputs 1xT vector of predictions.
    x : np.ndarray
        Observation to explain as vector with J features.
    X : np.ndarray
        Data on which prediction function was trained as TxJ matrix with
        T number of observations and J number of features.
    M : int
        Number of runs in which the J features are activated in random (permuted) order.

    Returns
    -------
    phi : ndarray
        Approximate SHAP values for x as 1xJ vector.
    base_value: float
        Base value E[f(x)].
    """

    # number of observations T and features J in training data X:
    T, J = X.shape

    # allocate matrix to hold one SHAP estimate per feature per permutation run:
    Phi = np.zeros((M, J, 2), dtype=np.float64)

    # obtain all masks, one for each activation for a total of
    # (J+1)*M*2 activations (including all features deactivated):
    masks = []
    for m in range(M):
        J_m = np.random.permutation(J)
        for forwards_backwards in [0, 1]:
            # reverse permutation for antithetic sampling:
            if forwards_backwards == 1:
                J_m = J_m[::-1]
            mask_m = np.zeros(J, dtype=bool)
            masks.append(mask_m.copy())
            for j_m in J_m:
                mask_m[j_m] = True
                masks.append(mask_m.copy())
    masks = np.array(masks)

    # obtain indices mapping mask in masks to mask in masks_unique:
    masks_map = np.zeros(masks.shape[0], dtype=int)
    masks_unique = np.unique(masks, axis=0)
    for i in range(masks_unique.shape[0]):
        masks_map[(masks == masks_unique[i, :]).all(axis=1)] = i

    # create one copy of X for every unique masking:
    X_masked = np.tile(X, (masks_unique.shape[0], 1))
    for i in range(masks_unique.shape[0]):
        # activate all masked features in copy of X by replacing them
        # with their respective values in x (the observation to explain):
        X_masked[i * T:(i + 1) * T, masks_unique[i]] = x[masks_unique[i]]

    # evaluate the model once on all unique masks;
    # reshape to number_of_unique_masks x T matrix;
    # average over T to obtain avg model output for each mask:
    y_hat_masked_avg = f(X_masked).flatten().reshape(masks_unique.shape[0], -1).mean(axis=1)

    # store first run with all features deactivated (yields base value / E[f(x)]):
    base_value = y_hat_masked_avg[0]

    for m in range(M):
        for forwards_backwards in [0, 1]:
            m_fb = m*2 + forwards_backwards
            # permutation m spans range m_fb*(J*+1):(m_fb+1)*(J*+1) in the non-unique masks array:
            selector = slice(m_fb * (J + 1), (m_fb + 1) * (J + 1))
            # obtain permutation of indices from order in which features are activated:
            J_m = np.argmax(masks[selector][1:] != masks[selector][:-1], axis=1)

            # permutation m spans range m*(J+1):(m+1)*(J+1) in the non-unique masks_map array:
            y_hat_avg_m = y_hat_masked_avg[masks_map[selector]]

            # marginal contribution is obtained as y_hat_avg_m[i+1] - y_hat_avg_m[i]
            # np.diff(y_hat_avg_m) yields the marginal contributions for J_m
            # store results in that order (J_m) in Phi:
            Phi[m, J_m, forwards_backwards] = np.diff(y_hat_avg_m)

    # return SHAP value estimates as average over forwards_backwards and all permutation runs M
    return Phi.mean(axis=-1).mean(axis=0), base_value


def approx_shap_interaction_unique(f, x, X, M):
    """Approximates SHAP interaction values for x ensuring no masks are evaluated twice.

    Approximate SHAP values are obtained for x. Generates M permuted orders in which
    the J features are activated and evaluates only the unique masks. This can be
    more efficient than "approx_shap(·)" when duplicated masks are expected among
    the M permutation runs (contingent on J; 2^J unique masks exist), but the exact
    algorithm is not feasible.

    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        Prediction function of model for which SHAP values are obtained;
        takes as input TxJ matrix with T number of observations to
        predict and J features and outputs 1xT vector of predictions.
    x : np.ndarray
        Observation to explain as vector with J features.
    X : np.ndarray
        Data on which prediction function was trained as TxJ matrix with
        T number of observations and J number of features.
    M : int
        Number of runs in which the J features are activated in random (permuted) order.

    Returns
    -------
    phi : ndarray
        Approximate SHAP values for x as 1xJ vector.
    base_value: float
        Base value E[f(x)].
    """

    # number of observations T and features J in training data X:
    T, J = X.shape

    # allocate matrix to hold one SHAP estimate per feature per permutation run:
    Phi = np.zeros((M, J, J, 2), dtype=np.float64)

    # obtain all masks, one for each activation for a total of
    # (J+1)*M*2 activations (including all features deactivated):
    masks, permutations = [], []
    for m in range(M):
        J_m = np.random.permutation(J)
        permutations.append(J_m)
        for forwards_backwards in [0, 1]:
            # reverse permutation for antithetic sampling:
            if forwards_backwards == 1:
                J_m = J_m[::-1]
            mask_m = np.zeros(J, dtype=bool)
            masks.append(mask_m.copy())
            for j_m in J_m:
                mask_m[j_m] = True
                masks.append(mask_m.copy())
            mask_m = np.zeros(J, dtype=bool)
            for j_m in J_m:
                mask_m[j_m] = True
                for i_m in range(j_m + 1, J) if forwards_backwards == 0 else range(j_m - 1, -1, -1):
                    mask_m_i = mask_m.copy()
                    mask_m_i[i_m] = True
                    masks.append(mask_m_i)

    # add masks for main effects:
    masks_main_effects = np.eye(J, dtype=bool)
    masks = np.vstack((masks, masks_main_effects))

    # obtain indices mapping mask in masks to mask in masks_unique:
    masks_map = np.zeros(masks.shape[0], dtype=int)
    masks_unique = np.unique(masks, axis=0)
    for i in range(masks_unique.shape[0]):
        masks_map[(masks == masks_unique[i, :]).all(axis=1)] = i

    # create one copy of X for every unique masking:
    X_masked = np.tile(X, (masks_unique.shape[0], 1))
    for i in range(masks_unique.shape[0]):
        # activate all masked features in copy of X by replacing them
        # with their respective values in x (the observation to explain):
        X_masked[i * T:(i + 1) * T, masks_unique[i]] = x[masks_unique[i]]

    # evaluate the model once on all unique masks;
    # reshape to number_of_unique_masks x T matrix;
    # average over T to obtain avg model output for each mask:
    y_hat_masked_avg = f(X_masked).flatten().reshape(masks_unique.shape[0], -1).mean(axis=1)

    # store first run with all features deactivated (yields base value / E[f(x)]):
    base_value = y_hat_masked_avg[0]

    main_effects = y_hat_masked_avg[masks_map[-J:]]

    mask_i = 0
    for o, J_m in enumerate(permutations):
        for forwards_backwards in [0, 1]:
            pred_wo_ij = y_hat_masked_avg[masks_map[mask_i:mask_i+J+1]]
            mask_i += J + 1
            if forwards_backwards == 1:
                J_m = J_m[::-1]
            for k, j_m in enumerate(J_m):
                for i_m in range(j_m + 1, J) if forwards_backwards == 0 else range(j_m - 1, -1, -1):
                    Phi[o, j_m, i_m, forwards_backwards] = Phi[o, i_m, j_m, forwards_backwards] = \
                        (y_hat_masked_avg[masks_map[mask_i]] + base_value - main_effects[i_m] - main_effects[j_m]) / 2
                    mask_i += 1

    Phi = Phi.mean(axis=-1).mean(axis=0)

    main_effects = y_hat_masked_avg[masks_map[-J:]] - base_value
    np.fill_diagonal(Phi, main_effects)

    # return SHAP value estimates as average over forwards_backwards and all permutation runs M
    return Phi, base_value


def taylor_approx_shap_interaction_unique(f, x, X, M):
    """Approximates SHAP interaction values for x ensuring no masks are evaluated twice.

    Approximate SHAP values are obtained for x. Generates M permuted orders in which
    the J features are activated and evaluates only the unique masks. This can be
    more efficient than "approx_shap(·)" when duplicated masks are expected among
    the M permutation runs (contingent on J; 2^J unique masks exist), but the exact
    algorithm is not feasible.

    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        Prediction function of model for which SHAP values are obtained;
        takes as input TxJ matrix with T number of observations to
        predict and J features and outputs 1xT vector of predictions.
    x : np.ndarray
        Observation to explain as vector with J features.
    X : np.ndarray
        Data on which prediction function was trained as TxJ matrix with
        T number of observations and J number of features.
    M : int
        Number of runs in which the J features are activated in random (permuted) order.

    Returns
    -------
    phi : ndarray
        Approximate SHAP values for x as 1xJ vector.
    base_value: float
        Base value E[f(x)].
    """

    def powerset(iterable):
        s = list(iterable)
        from itertools import combinations
        return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

    # number of observations obs and features J in training data X:
    obs, J = X.shape

    masks, pairs = [], {}

    for j in range(J):
        for i in range(j+1, J):
            pairs[(j, i)] = []

    for _ in range(M):
        J_m = np.random.permutation(J)
        for S in pairs.keys():
            T = tuple(J_m[:min([np.where(J_m == s)[0][0] for s in S])])
            T_masks = []
            for W in powerset(S):
                mask = np.zeros(J, dtype=bool)
                mask[list(T+W)] = True
                masks.append(mask)
                mask_idx = len(masks)-1
                T_masks.append((mask_idx, W))
            pairs[S].append((T, T_masks))

    masks = np.stack(masks)

    # obtain indices mapping mask in masks to mask in masks_unique:
    masks_map = np.zeros(masks.shape[0], dtype=int)
    masks_unique = np.unique(masks, axis=0)
    for i in range(masks_unique.shape[0]):
        masks_map[(masks == masks_unique[i, :]).all(axis=1)] = i

    # create one copy of X for every unique masking:
    X_masked = np.tile(X, (masks_unique.shape[0], 1))
    for i in range(masks_unique.shape[0]):
        # activate all masked features in copy of X by replacing them
        # with their respective values in x (the observation to explain):
        X_masked[i * obs:(i + 1) * obs, masks_unique[i]] = x[masks_unique[i]]

    # evaluate the model once on all unique masks;
    # reshape to number_of_unique_masks x T matrix;
    # average over T to obtain avg model output for each mask:
    y_hat_masked_avg = f(X_masked).flatten().reshape(masks_unique.shape[0], -1).mean(axis=1)

    Phi = np.zeros((J, J))
    for S in pairs.keys():
        s = len(S)
        j, i = S
        delta = 0
        for T, W_masks in pairs[S]:
            for W_idx, W in W_masks:
                w = len(W)
                delta += (-1) ** (w - s) * y_hat_masked_avg[masks_map[W_idx]]
        Phi[j, i] = Phi[i, j] = delta / M

    return Phi


def shapley_approx_shap_interaction_unique(f, x, X, pair, main, base):
    """Approximates SHAP interaction values for x ensuring no masks are evaluated twice.

    Approximate SHAP values are obtained for x. Generates M permuted orders in which
    the J features are activated and evaluates only the unique masks. This can be
    more efficient than "approx_shap(·)" when duplicated masks are expected among
    the M permutation runs (contingent on J; 2^J unique masks exist), but the exact
    algorithm is not feasible.

    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        Prediction function of model for which SHAP values are obtained;
        takes as input TxJ matrix with T number of observations to
        predict and J features and outputs 1xT vector of predictions.
    x : np.ndarray
        Observation to explain as vector with J features.
    X : np.ndarray
        Data on which prediction function was trained as TxJ matrix with
        T number of observations and J number of features.
    M : int
        Number of runs in which the J features are activated in random (permuted) order.

    Returns
    -------
    phi : ndarray
        Approximate SHAP values for x as 1xJ vector.
    base_value: float
        Base value E[f(x)].
    """

    def powerset(iterable):
        s = list(iterable)
        from itertools import combinations
        return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

    # number of observations obs and features J in training data X:
    T, J = X.shape

    S = [x for x in np.arange(J) if x not in pair]

    masks = []
    for s in powerset(S):
        mask = np.zeros(J, dtype=bool)
        mask[list(s)] = True
        masks.append(mask.copy())
        mask[pair[0]] = True
        masks.append(mask.copy())
        mask[pair[0]] = False
        mask[pair[1]] = True
        masks.append(mask.copy())
        mask[list(pair)] = True
        masks.append(mask.copy())

    masks = np.stack(masks)

    # obtain indices mapping mask in masks to mask in masks_unique:
    masks_map = np.zeros(masks.shape[0], dtype=int)
    masks_unique = np.unique(masks, axis=0)
    for i in range(masks_unique.shape[0]):
        masks_map[(masks == masks_unique[i, :]).all(axis=1)] = i

    # create one copy of X for every unique masking:
    X_masked = np.tile(X, (masks_unique.shape[0], 1))
    for i in range(masks_unique.shape[0]):
        # activate all masked features in copy of X by replacing them
        # with their respective values in x (the observation to explain):
        X_masked[i * T:(i + 1) * T, masks_unique[i]] = x[masks_unique[i]]

    # evaluate the model once on all unique masks;
    # reshape to number_of_unique_masks x T matrix;
    # average over T to obtain avg model output for each mask:
    y_hat_masked_avg = f(X_masked).flatten().reshape(masks_unique.shape[0], -1).mean(axis=1)

    out = 0

    i = 0
    for s in powerset(S):
        a = masks_map[i]
        i += 1
        b = masks_map[i]
        i += 1
        c = masks_map[i]
        i += 1
        d = masks_map[i]
        i += 1

        #delta = y_hat_masked_avg[d] - y_hat_masked_avg[c] - y_hat_masked_avg[b] + y_hat_masked_avg[a]
        delta = y_hat_masked_avg[d] - (main[pair[0]]+base) - (main[pair[1]]+base) + base

        import math
        coef = (math.factorial(len(s)) * math.factorial(J-len(s)-2)) / (2*math.factorial(J-1))
        # coef = (2 * math.factorial(len(s)) * math.factorial(J-len(s)-1)) / (math.factorial(J))
        out += coef * delta
    return out


def get_shap_values(f, x, X, M, enforce_unique=False):
    """Model agnostic function to obtain approximate or exact SHAP values for x.

    Obtains exact SHAP values if the number of permutation runs M is larger or equal to
    the number of unique masks that can be evaluated (2^J); otherwise, obtains
    approximate SHAP values by permuting repeatedly the order in which features are
    activated to obtain the marginal contribution of a feature to the prediction of f(x).
    With enforce_unique=True enforces that no same feature masks are evaluated twice.
    For medium J contingent on M this is appropriate. For large J, enforce_unique=False
    is appropriate as no duplicate masks can be expected for reasonable M among the 2^J
    unique masks.

    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        Prediction function of model for which SHAP values are obtained;
        takes as input TxJ matrix with T number of observations to
        predict and J features and outputs 1xT vector of predictions.
    x : np.ndarray
        Observation to explain as vector with J features.
    X : np.ndarray
        Data on which prediction function was trained as TxJ matrix with
        T number of observations and J number of features.
    M : int
        Maximum number of runs. For exact algorithm, this equals 2^J and for approximate
        algorithms, this is the number of times the J features are activated in
        random (permuted) order.
    enforce_unique : bool
        Ensures that the same masking is never evaluated twice

    Returns
    -------
    phi : ndarray
        Exact or approximate SHAP values for x as 1xJ vector.
    base_value: float
        Base value E[f(x)].
    """

    if 2 ** X.shape[1] <= M:
        phi, base_value = exact_shap(f, x, X)
    elif enforce_unique:
        phi, base_value = approx_shap_unique(f, x, X, M)
    else:
        phi, base_value = approx_shap(f, x, X, M)

    return phi, base_value


def _compare_to_shap_pkg():

    import xgboost
    import shap
    from tqdm import tqdm

    X, y = shap.datasets.boston()
    X = X.to_numpy()

    model = xgboost.XGBRegressor()
    model.fit(X, y)

    bench = shap.TreeExplainer(model, data=shap.maskers.Independent(X, max_samples=X.shape[0]))(X).values

    m = 200

    approx_shap_simple_err = np.abs(np.stack([approx_shap_simple(
        model.predict, X[x, :], X, m)[0] for x in tqdm(range(X.shape[0]))]) - bench).mean()

    approx_shap_err = np.abs(np.stack([approx_shap(
        model.predict, X[x, :], X, m)[0] for x in tqdm(range(X.shape[0]))]) - bench).mean()

    approx_shap_unique_err = np.abs(np.stack([approx_shap_unique(
        model.predict, X[x, :], X, m)[0] for x in tqdm(range(X.shape[0]))]) - bench).mean()

    permutation_explainer_err = np.abs(shap.PermutationExplainer(
        model.predict, max_evals=m*(2*X.shape[1]+1),
        masker=shap.maskers.Independent(X, max_samples=X.shape[0]))(X).values - bench).mean()

    # 0.024337823855398354 VS. 0.024141258289690826 0.023489270322109828 0.023647336648061307 @ m=25:
    # XXX @ m=200:
    print(permutation_explainer_err, "VS.", approx_shap_simple_err, approx_shap_err, approx_shap_unique_err)

    exit(-1)


def interaction_correlation():

    import shap
    import statsmodels.api as sm

    rho = .7

    corr = np.array([[1, rho, rho, rho],
                     [rho, 1, rho, rho],
                     [rho, rho, 1, rho],
                     [rho, rho, rho, 1]])

    std = np.array([1, 1, 1, 1])

    cov = np.matmul(np.matmul(np.diag(std), corr), np.diag(std))

    X = np.random.multivariate_normal(np.repeat(0, 4), cov, 1000)

    y = X[:, 0] + X[:, 1] + 1.5*(X[:, 2] * X[:, 3]) + np.random.normal(0, 2, 1000)

    model = sm.OLS(y, np.stack((X[:, 0], X[:, 1], X[:, 2] * X[:, 3])).T).fit()

    pred_fn = lambda x: model.predict(np.stack((x[:, 0], x[:, 1], x[:, 2] * x[:, 3])).T)

    zz = shap.explainers.Exact(pred_fn, X)(X[:, :], interactions=2)

    main = np.diag(zz.values[0])
    base = zz.base_values[0]

    a = shapley_approx_shap_interaction_unique(pred_fn, X[0, :], X, pair=(2, 3), main=main, base=base)
    # b = shapley_approx_shap_interaction_unique(pred_fn, X[0, :], X, M=100, me=True)


    yy = np.stack([shapley_approx_shap_interaction_unique(pred_fn, X[x, :], X, M=1) for x in tqdm(range(1000))]).mean(axis=0)

    print()


noop_val = 99999

def gray_code_indexes(nbits):
    """ Produces an array of which bits flip at which position.
    We assume the masks start at all zero and -1 means don't do a flip.
    This is a more efficient represenation of the gray_code_masks version.
    """
    out = np.ones(2 ** nbits, dtype=np.int) * noop_val
    li = np.zeros(nbits, dtype=np.bool)
    for term in range((1 << nbits) - 1):
        if term % 2 == 1:  # odd
            for i in range(-1, -nbits, -1):
                if li[i] == 1:
                    li[i - 1] = li[i - 1] ^ 1
                    out[term + 1] = nbits + (i - 1)
                    break
        else:  # even
            li[-1] = li[-1] ^ 1
            out[term + 1] = nbits - 1
    return out

J = 4

delta_indexes = gray_code_indexes(J)

extended_delta_indexes = np.zeros(2 ** J, dtype=np.int)
for i in range(2 ** J):
    if delta_indexes[i] == noop_val:
        extended_delta_indexes[i] = delta_indexes[i]
    else:
        extended_delta_indexes[i] = np.arange(J)[delta_indexes[i]]

mask = np.zeros(J, dtype=np.bool)

set_size = 0
M = J
for i in range(2 ** M):
    # update the mask
    delta_ind = extended_delta_indexes[i]
    if delta_ind != noop_val:
        mask[delta_ind] = ~mask[delta_ind]
        if mask[delta_ind]:
            set_size += 1
        else:
            set_size -= 1

    print()



interaction_correlation()

import xgboost
import shap
from tqdm import tqdm


def fn_paper(X):
    return 1.5*(X[:, 0] * X[:, 1]) + 2*(X[:, 2] * X[:, 3])

XX = np.random.randint(0, 2, (100, 4))

ZZ = _approx_shap_interaction_unique(fn_paper, XX[0, :], XX, M=100)

explainer = shap.explainers.Exact(fn_paper, shap.maskers.Independent(XX, max_samples=XX.shape[0]))
A = explainer(XX[[0], :], interactions=2).values#[0, :, :]

X, y = shap.datasets.boston()
X = X.to_numpy()[:, :5]

model = xgboost.XGBRegressor()
model.fit(X, y)

explainer = shap.explainers.Exact(model.predict, shap.maskers.Independent(X, max_samples=X.shape[0]))
A = explainer(X[[0], :], interactions=2).values[0, :, :]

ZZ = _approx_shap_interaction_unique(model.predict, X[0, :], X, M=1)

print()

##-----
def subset_before(S, ordering, ordering_dict):
    end_idx = min(ordering_dict[s] for s in S)
    return ordering[:end_idx]

def powerset(iterable):
    s = list(iterable)
    from itertools import combinations
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

def collect_att(S, S_T_Z_dict, Z_score_dict, n):
    s = len(S)
    subsetsW = powerset(S)

    total_att = 0

    for T in S_T_Z_dict[S]:

        att = 0
        for i, W in enumerate(subsetsW):
            w = len(W)
            att += (-1) ** (w - s) * Z_score_dict[S_T_Z_dict[S][T][i]]

        total_att += att

    num_orderings = len(S_T_Z_dict[S])
    return total_att / num_orderings


##---

print("OK")

##---

num_features = 5  # X.shape[1]
num_orderings = 1

Ss = []
for i in range(num_features):
    for j in range(i + 1, num_features):
        S = (i, j)
        Ss.append(S)

Z_set = set()
S_T_Z_dict = dict()
for S in Ss:
    subsetsW = powerset(S)
    S_T_Z_dict[S] = {}

    for _ in range(num_orderings):
        ordering = np.arange(num_features) #  np.random.permutation(list(range(num_features)))
        ordering_dict = {ordering[i]: i for i in range(len(ordering))}

        end_idx = min(ordering_dict[s] for s in S)
        T = tuple(ordering[:end_idx])

        S_T_Z_dict[S][T] = []

        set_indices = []
        for W in subsetsW:
            Z = tuple(set(W) | set(T))
            Z_set.add(Z)
            S_T_Z_dict[S][T].append(Z)

Z_list = list(Z_set)
##-----

# B = _approx_shap_interaction_unique(model.predict, X[0, :], X, 1)[0]

# https://github.com/mtsang/archipelago/blob/a8e11c89cbad3edf92ac6c899541d35b1fc0c2af/baselines/shapley_taylor_interaction_index/sti_explainer.py

print()

def _xx(x):
    print(x)
    return model.predict(x)

#
#B = shap.TreeExplainer(model).shap_interaction_values(X[[0], :])[0, :, :].sum(axis=1)
C = _approx_shap_interaction_unique(_xx, XXX, X, 1)[0].sum(axis=1)

perm_exp = shap.PermutationExplainer(_xx, masker=shap.maskers.Independent(X, max_samples=X.shape[0]))
XX = shap.utils.MaskedModel(perm_exp.model, perm_exp.masker, perm_exp.link, XXX).main_effects(need_interactions=True)[1]

# explain_full

shap.utils.MaskedModel(perm_exp.model, perm_exp.masker, perm_exp.link, X[0, :]).main_effects()
shap.utils.MaskedModel(perm_exp.model, perm_exp.masker, perm_exp.link, X[0, :])._x_main_effects()


bench = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
A = bench.shap_interaction_values(X[[0], :]).sum(axis=1)
B = bench(X[[0], :]).values


## max_evals // (2*len(inds)+1)

np.random.seed(0)
data = np.random.randint(0, 2, size=(100,5))
def model(data):
    return data[:,0] * data[:,2] + data[:,1] + data[:,2] + data[:,2] * data[:,3]

XX = np.stack([approx_shap_unique(model, data[x, :], np.zeros((1,5)), 1)[0] for x in range(100)])

right_answer = np.zeros(data.shape)
right_answer[:, 0] += (data[:, 0] * data[:, 2]) / 2
right_answer[:, 2] += (data[:, 0] * data[:, 2]) / 2
right_answer[:, 1] += data[:, 1]
right_answer[:, 2] += data[:, 2]
right_answer[:, 2] += (data[:, 2] * data[:, 3]) / 2
right_answer[:, 3] += (data[:, 2] * data[:, 3]) / 2
shap_values = shap.explainers.Permutation(model, np.zeros((1,5)))(data)

XX = shap.explainers.Permutation(model, np.zeros((1,5)))  # (data)



print()

import statsmodels.api as sm

X = np.random.normal(0, 1, (100, 10))
e = np.random.normal(0, .1, (100, 10))

y = np.sum(X + e, axis=1)

model = sm.OLS(y, sm.add_constant(X)).fit()

def fn(x):
    return model.predict(sm.add_constant(x))

t(fn, X[0, :], X, 10)



print()
