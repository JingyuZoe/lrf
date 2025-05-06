# lrf/model.py
import numpy as np
from .response import b_spline_response, compute_kappa, time_dependent_response


def predict_inflow_dynamic(rainfall, T0, h0, alpha, kappa_array, S):
    time_steps = len(rainfall)
    inflow = np.zeros_like(rainfall)

    for t in range(time_steps):
        response_t = time_dependent_response(T0, h0, alpha, kappa_array[t], S[t])
        T_dyn = len(response_t)

        for tau in range(min(t + 1, T_dyn)):
            inflow[t] += rainfall[t - tau] * response_t[tau]

    return inflow


def simulate_rdii(rainfall, S, params):
    T0 = int(params['T0'])
    c = np.array(params['c'])
    knots = np.array(params['knots'])
    k = int(params['k'])
    alpha = float(params['alpha'])
    beta = float(params['beta'])  # For S generation
    gamma = float(params['gamma'])
    S_th = float(params['S_th'])

    h0 = b_spline_response(T0, c, knots, k)
    kappa_array = compute_kappa(S, gamma, S_th)

    return predict_inflow_dynamic(rainfall, T0, h0, alpha, kappa_array, S)


# lrf/utils.py
import numpy as np

def generate_s(rainfall, alpha):
    S = np.zeros_like(rainfall)
    for t in range(1, len(rainfall)):
        S[t] = alpha * S[t - 1] + (1 - alpha) * rainfall[t]
    return S
