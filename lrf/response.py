# lrf/response.py

import numpy as np
from scipy.interpolate import splrep, BSpline, interp1d


def b_spline_response(T0, c, knots, k):
    """
    Construct a base B-spline response function h₀(τ) with non-negativity constraint.

    Parameters
    ----------
    T0 : int
        Length of the response window.
    c : array-like
        B-spline control points.
    knots : array-like
        Knot vector for the B-spline.
    k : int
        Degree of the B-spline.

    Returns
    -------
    h0 : ndarray
        Response function of length T0, values clipped to non-negative.
    """
    t = np.arange(T0) 
    spline = BSpline(knots, c, k)
    return np.maximum(0, spline(t))  # Ensuring non-negativity


def compute_kappa(S, gamma, S_th):
    """
    Compute the dynamic stretch coefficient κ(S) = γ ⋅ S ⋅ (S − S_th)

    Parameters
    ----------
    S : array-like
        Saturation time series.
    gamma : float
        Scaling factor for the stretch.
    S_th : float
        Threshold of saturation for switching from compression to stretching.

    Returns
    -------
    kappa : ndarray
        Time-dependent stretch coefficients.
    """
    return gamma * S * (S - S_th)


def time_dependent_response(T0, h_0, alpha, kappa, S, dt=1.0):
    """
    Apply time warping and amplitude modulation to the base response h₀(τ),
    generating a dynamic response h(τ, S(t)) at a given time step.

    Parameters
    ----------
    T0 : int
        Length of the base response function.
    h_0 : array-like
        Base response function h₀(τ).
    alpha : float
        Amplitude decay factor depending on saturation S.
    kappa : float
        Stretch coefficient at the current time step.
    S : float
        Saturation at the current time step.
    dt : float, optional
        Time resolution (default is 1.0).

    Returns
    -------
    h : ndarray
        Time-stretched and amplitude-modulated response function.
    """
    # Original time vector
    t = np.arange(T0) * dt

    # Compute stretch factor (>1 means slower/wider response)
    time_stretch = 1 + kappa * S

    # Length of stretched response (ceil ensures integer)
    stretched_T = int(np.ceil(T0 * time_stretch))

    # New time base for interpolation (compressed to match original domain)
    stretched_t = np.arange(stretched_T) * dt / time_stretch

    # Cubic interpolation of base response
    h_interp = interp1d(t, h_0, kind='cubic', fill_value="extrapolate", assume_sorted=True)
    stretched_h0 = h_interp(stretched_t)

    # Apply exponential amplitude decay
    h = stretched_h0 * np.exp(-alpha * S)

    return h
