# lrf/utils.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from lrf.response import b_spline_response
from lrf.model import compute_saturation


def calculate_r2(pred, obs):
    """
    Compute the coefficient of determination (R²) between predicted and observed values.

    Parameters
    ----------
    pred : array-like
        Model predictions.
    obs : array-like
        Ground truth observations.

    Returns
    -------
    r2 : float
        R² score, where 1 indicates perfect prediction.
    """
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return 1 - ss_res / ss_tot


def calculate_kge(pred, obs):
    """
    Compute Kling-Gupta Efficiency (KGE) between predicted and observed values.

    KGE balances correlation, variability, and bias components.

    Parameters
    ----------
    pred : array-like
        Model predictions.
    obs : array-like
        Ground truth observations.

    Returns
    -------
    kge : float
        Kling-Gupta Efficiency value.
    """
    r = np.corrcoef(pred, obs)[0, 1]
    alpha = np.std(pred) / np.std(obs)
    beta = np.mean(pred) / np.mean(obs)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)



def build_param_dict(params, knots, k):
    """
    Build parameter dictionary from optimized flat parameter array.

    Parameters
    ----------
    params : array-like
        Optimized parameter array in format:
        [T0, c_0, ..., c_n, beta, alpha, gamma, S_th]

    knots : array-like
        Knot vector for B-spline.

    k : int
        Degree of B-spline.

    Returns
    -------
    param_dict : dict
        Structured parameter dictionary for simulate_rdii()
    """
    return {
        'T0': params[0],
        'c': params[1:-4],
        'knots': knots,
        'k': k,
        'beta': params[-4],
        'alpha': params[-3],
        'gamma': params[-2],
        'S_th': params[-1]
    }

def print_param_dict(param_dict, prefix="[Fitted Parameters]"):
    """
    Nicely print fitted parameter dictionary.

    Parameters
    ----------
    param_dict : dict
        Dictionary containing all model parameters.
    prefix : str
        Optional title or header to print.
    """
    print(prefix)
    print(f"  T0     = {param_dict['T0']:.2f}")
    print(f"  beta   = {param_dict['beta']:.4f}")
    print(f"  alpha  = {param_dict['alpha']:.4f}")
    print(f"  gamma  = {param_dict['gamma']:.4f}")
    print(f"  S_th   = {param_dict['S_th']:.4f}")
    print("  c      = [")
    for i, ci in enumerate(param_dict['c']):
        print(f"    {ci:.6f},")
    print("  ]")



def plot_response_at_saturation_levels(
    param_dict,
    rainfall,
    compute_kappa_fn,
    time_dependent_response_fn,
    num_bins=6,
    figsize=(8, 5)
):
    """
    Plot time-dependent response h(τ, S) at selected saturation levels using learned parameters.

    Parameters
    ----------
    param_dict : dict
        Dictionary of fitted model parameters (T0, c, knots, k, beta, alpha, gamma, S_th).
    rainfall : array-like
        Input rainfall time series used to compute saturation S(t).
    compute_kappa_fn : function
        Function to compute κ(S).
    time_dependent_response_fn : function
        Function to compute h(τ, S).
    num_bins : int
        Number of saturation bins (default: 6).
    figsize : tuple
        Figure size for plotting.
    """
    # === Unpack parameters ===
    T0 = int(param_dict["T0"])
    c = param_dict["c"]
    knots = param_dict["knots"]
    k = param_dict["k"]
    beta = param_dict["beta"]
    alpha = param_dict["alpha"]
    gamma = param_dict["gamma"]
    S_th = param_dict["S_th"]

    # === Compute derived values ===
    h0 = b_spline_response(T0, c, knots, k)
    S_array = compute_saturation(rainfall, beta)

    # === Select representative time points by S(t) range ===
    S_max = np.max(S_array)
    bins = np.linspace(0.001 * S_max, 0.8 * S_max, num_bins - 1)
    bins = np.append(bins, np.inf)

    selected_indices = []
    for i in range(len(bins) - 1):
        indices = np.where((S_array >= bins[i]) & (S_array < bins[i + 1]))[0]
        if len(indices) > 0:
            selected_indices.append(indices[0])

    # === Plot each response function ===
    colors = cm.viridis(np.linspace(0, 1, len(selected_indices)))
    plt.figure(figsize=figsize)

    for i, t in enumerate(selected_indices):
        S_t = S_array[t]
        kappa_t = compute_kappa_fn(S_t, gamma, S_th)
        h_t = time_dependent_response_fn(T0, h0, alpha, kappa_t, S_t)
        plt.plot(np.arange(len(h_t)), h_t, label=f"h(τ, S={S_t:.2f})", color=colors[i])

    # === Plot baseline response h0 ===
    plt.plot(np.arange(len(h0)), h0, label="$h_0(\\tau)$ (baseline)", linestyle="dotted", color="black")

    plt.xlabel("Time Steps")
    plt.ylabel("Response Function")
    plt.title("Time-Dependent Response Function at Different Saturation Levels")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
