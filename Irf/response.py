# lrf/response.py
import numpy as np
from scipy.interpolate import BSpline, interp1d


def b_spline_response(T0, c, knots, k):
    t = np.arange(T0)
    spline = BSpline(knots, c, k)
    return np.maximum(0, spline(t))  # make sure non-nigtive


def compute_kappa(S, gamma, S_th):
    return gamma * S * (S - S_th)


def time_dependent_response(T0, h0, alpha, kappa, S, dt=1.0):
    t = np.arange(T0) * dt
    time_stretch = 1 + kappa * S
    stretched_T = int(np.ceil(T0 * time_stretch))
    stretched_t = np.arange(stretched_T) * dt / time_stretch
    h_interp = interp1d(t, h0, kind='cubic', fill_value="extrapolate", assume_sorted=True)
    stretched_h0 = h_interp(stretched_t)
    h = stretched_h0 * np.exp(-alpha * S)
    return h