# lrf/model.py

import numpy as np
from .response import b_spline_response, compute_kappa, time_dependent_response


def compute_saturation(rainfall, beta):
    """
    Compute the saturation time series S(t) based on a recursive exponential smoothing.

    Parameters
    ----------
    rainfall : array-like
        Time series of rainfall data.
    beta : float
        Smoothing coefficient (0 < beta < 1).
        Larger beta retains more memory (slower decay).

    Returns
    -------
    S : ndarray
        Saturation index time series (same length as rainfall).
    """
    S = np.zeros_like(rainfall)
    for t in range(1, len(rainfall)):
        S[t] = beta * S[t - 1] + (1 - beta) * rainfall[t]
    return S


def predict_inflow_dynamic(rainfall, T_0, h0, alpha, kappa_array, S):
    """
    Predict inflow (e.g., RDII) using a dynamic response function h(τ, S(t)).

    Parameters
    ----------
    rainfall : array-like
        Input rainfall time series.
    T_0 : int
        Base response window size.
    h0 : array-like
        Base response function h₀(τ), constructed from B-spline.
    alpha : float
        Exponential decay factor based on saturation.
    kappa_array : array-like
        Stretch factor κ(t) for each time step.
    S : array-like
        Saturation time series.

    Returns
    -------
    inflow : ndarray
        Simulated inflow (same length as rainfall).
    """
    time_steps = len(rainfall)
    inflow = np.zeros_like(rainfall)

    for t in range(time_steps):
        response_t = time_dependent_response(T_0, h0, alpha, kappa_array[t], S[t])
        T_dyn = len(response_t)

        for tau in range(min(t + 1, T_dyn)):
            inflow[t] += rainfall[t - tau] * response_t[tau]

    return inflow

def simulate_rdii(rainfall, params):
    """
    High-level wrapper for simulating RDII from rainfall using learnable response model.

    Parameters
    ----------
    rainfall : array-like
        Input rainfall time series.
    params : dict
        Model parameters, must include:
            - 'T0' : int, base response window size
            - 'c' : array-like, B-spline control points
            - 'knots' : array-like, B-spline knots
            - 'k' : int, spline degree
            - 'beta' : float, saturation accumulation coefficient
            - 'alpha' : float, response amplitude decay factor
            - 'gamma' : float, time stretch intensity
            - 'S_th' : float, saturation threshold

    Returns
    -------
    inflow : ndarray
        Simulated RDII time series.
    """
    T0 = int(params['T0'])
    c = params['c']
    knots = params['knots']
    k = params['k']
    beta = params['beta']
    alpha = params['alpha']
    gamma = params['gamma']
    S_th = params['S_th']

    # Generate components
    S = compute_saturation(rainfall, beta)
    h0 = b_spline_response(T0, c, knots, k)
    kappa = compute_kappa(S, gamma, S_th)

    # Run simulation
    inflow = predict_inflow_dynamic(rainfall, T0, h0, alpha, kappa, S)
    return inflow


