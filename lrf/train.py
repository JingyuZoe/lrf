# lrf/train.py

import numpy as np
from scipy.optimize import minimize
from .model import predict_inflow_dynamic,compute_saturation
from .response import b_spline_response, compute_kappa


def loss_function(params, rainfall, true_inflow, knots, k,
                  lambda_mse, lambda_smooth, lambda_nonneg, lambda_start, lambda_tail):
    """
    Loss function for learnable and time-warped RDII modeling using a dynamic B-spline response.

    Parameters
    ----------
    params : array-like
        Model parameters in the format:
        [T0, c0, ..., cn, beta, alpha, gamma, S_th]
    rainfall : array-like
        Input rainfall time series.
    true_inflow : array-like
        Observed inflow time series.
    knots : array-like
        B-spline knot vector.
    k : int
        Spline degree.
    lambda_mse, lambda_smooth, lambda_nonneg, lambda_start, lambda_tail : float
        Weights for respective loss terms.

    Returns
    -------
    total_loss : float
        Combined loss value including prediction error and regularizations.
    """
    T_0 = params[0]
    c = params[1:-3]
    beta = params[-4]
    alpha = params[-3]
    gamma = params[-2]
    S_th = params[-1]

    S = compute_saturation(rainfall, beta)
    h_0 = b_spline_response(T_0, c, knots, k)
    kappa_dynamic = compute_kappa(S, gamma, S_th)
    pred_inflow = predict_inflow_dynamic(rainfall, T_0, h_0, alpha, kappa_dynamic, S)

    mse = np.mean((pred_inflow - true_inflow) ** 2)
    R_smooth = np.sum(np.diff(h_0, n=2) ** 2)
    R_nonneg = np.sum(np.minimum(0, h_0) ** 2)
    R_start = h_0[0] ** 2
    T_tail = max(1, int(0.001 * T_0))
    R_tail = np.sum(h_0[-T_tail:] ** 2)

    total_loss = (
        lambda_mse    * mse +
        lambda_smooth * R_smooth +
        lambda_nonneg * R_nonneg +
        lambda_start  * R_start +
        lambda_tail   * R_tail
    )
    return total_loss


def build_partial_loss(loss_fn, full_params_init, active_mask, loss_args):
    """
    Construct a partial loss function for stage-wise optimization.

    Parameters
    ----------
    loss_fn : callable
        Full loss function to optimize.
    full_params_init : array-like
        Initial full parameter vector.
    active_mask : list of bool
        Boolean mask indicating which parameters to optimize.
    loss_args : tuple
        Additional arguments to be passed to loss_fn.

    Returns
    -------
    partial_loss : callable
        Function of subset of parameters.
    init_subset : list
        Initial values of active parameters.
    active_indices : list
        Indices of active parameters.
    """
    active_indices = [i for i, active in enumerate(active_mask) if active]

    def partial_loss(opt_vars):
        full_params = full_params_init.copy()
        for i, idx in enumerate(active_indices):
            full_params[idx] = opt_vars[i]
        return loss_fn(full_params, *loss_args)

    init_subset = [full_params_init[i] for i in active_indices]
    return partial_loss, init_subset, active_indices


 


def fit_lrf_model( rainfall,true_rdii,init_params, bounds,knots,k,strategy,weights=None):
    """
    Train a learnable response function model via stage-wise optimization.

    Parameters
    ----------
    rainfall : array-like
        Input rainfall series.
    true_rdii : array-like
        Observed RDII series (target).
    init_params : array-like
        Initial parameter values.
    bounds : list of tuple
        Bounds for each parameter (used by optimizer).
    knots : array-like
        B-spline knot vector.
    k : int
        Spline degree.
    strategy : list of dict
        Training phases, each with 'name', 'method', and 'opt_mask'.
    weights : dict
        Dictionary with keys like 'mse', 'smooth', 'nonneg', 'start', 'tail'.

    Returns
    -------
    params : ndarray
        Optimized parameter vector.
    all_results : dict
        Optimization result for each phase.
    """
    
    # === Default weights ===
    if weights is None:
        weights = {
            'mse': 1e5,
            'smooth': 1e1,
            'nonneg': 1e3,
            'start': 1e1,
            'tail': 1e3
        }

    all_results = {}
    params = np.array(init_params)

    for phase in strategy:
        name = phase['name']
        method = phase.get('method', 'L-BFGS-B')
        opt_mask = np.array(phase['opt_mask'])

        def partial_loss(sub_params):
            full = params.copy()
            full[opt_mask] = sub_params
            return loss_function(full, rainfall, true_rdii, knots, k,
                                 weights['mse'],
                                 weights['smooth'],
                                 weights['nonneg'],
                                 weights['start'],
                                 weights['tail'])

        sub_init = params[opt_mask]
        sub_bounds = [b for i, b in enumerate(bounds) if opt_mask[i]]
        result = minimize(partial_loss, sub_init, method=method, bounds=sub_bounds)

        # Update full parameter vector
        params[opt_mask] = result.x
        all_results[name] = result

    return params, all_results


def get_default_strategy(n_c):
    """
    Generate default stage-wise training strategy.

    Parameters
    ----------
    n_c : int
        Number of B-spline control points.

    Returns
    -------
    strategy : list of dict
        Training phases with name, method, and parameter mask.
    """
    return [
        {"name": "phase1", "method": "Powell", "opt_mask": [True] + [False] * n_c + [False]*4},
        {"name": "phase2", "method": "L-BFGS-B", "opt_mask": [False] + [True] * n_c + [False]*4},
        {"name": "phase3", "method": "L-BFGS-B", "opt_mask": [False] * (1+n_c) + [True]*4},
    ]


def get_default_model_config(T0_init=120, n_points=12, k=3):
    """
    Return default model initialization and bounds.

    Parameters
    ----------
    T0_init : int
        Initial base response window length (default: 120)
    n_points : int
        Number of B-spline control points (default: 12)
    k : int
        B-spline degree (default: 3)

    Returns
    -------
    dict with keys:
        - T0_init : int
        - k : int
        - n_points : int
        - c_init : np.ndarray
        - knots : np.ndarray
        - params_init : np.ndarray
        - bounds : list of tuples
    """
    # === B-spline control points (Reasonable shape guesses) ===
    c_init = np.array([
        0.0, 0.3, 0.11, 0.10,
        0.08, 0.055, 0.03, 0.002,
        0.002, 0.0018, 0.0015, 0.0013
    ])

    # === Knots ===
    internal_knots = np.linspace(0, T0_init, n_points - k + 1)
    knots = np.concatenate(([0] * k, internal_knots, [T0_init] * k))

    # === Other parameters ===
    beta_init = 0.4
    alpha_init = 0.5
    gamma_init = 10
    S_th_init = 0.12

    # === Combine parameters ===
    params_init = np.concatenate([[T0_init], c_init, [beta_init, alpha_init, gamma_init, S_th_init]])

    # === Bounds ===
    T0_bounds = (24, 150)
    c_bounds = [(0, 1)] * len(c_init)
    beta_bounds = (0.4, 0.99)
    alpha_bounds = (0, 5)
    gamma_bounds = (5, 30)
    S_th_bounds = (0.05, 0.5)

    bounds = [T0_bounds] + c_bounds + [beta_bounds, alpha_bounds, gamma_bounds, S_th_bounds]

    return {
        "T0_init": T0_init,
        "n_points": n_points,
        "k": k,
        "c_init": c_init,
        "knots": knots,
        "params_init": params_init,
        "bounds": bounds
    }
