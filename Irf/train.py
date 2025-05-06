# lrf/train.py
import numpy as np
from scipy.optimize import minimize
from .model import simulate_rdii
from .response import b_spline_response
from .utils import generate_s


def loss_function(params, rainfall, true_rdii, knots, k, weights):
    """
    Compute loss including MSE and optional regularizations.
    params: [T0, c0...cn, beta, alpha, gamma, S_th]
    weights: dict with keys like 'mse', 'smooth', 'start', 'tail'
    """
    T0 = int(params[0])
    c = np.array(params[1:-4])
    beta, alpha, gamma, S_th = params[-4:]

    h0 = b_spline_response(T0, c, knots, k)
    S = generate_s(rainfall, alpha)
    params_dict = {
        'T0': T0, 'c': c, 'knots': knots, 'k': k,
        'beta': beta, 'alpha': alpha, 'gamma': gamma, 'S_th': S_th
    }
    pred_rdii = simulate_rdii(rainfall, S, params_dict)

    mse = np.mean((pred_rdii - true_rdii) ** 2)
    smooth = np.sum((np.diff(h0, n=2)) ** 2)
    start = h0[0] ** 2
    tail = h0[-1] ** 2

    total_loss = (
        weights.get('mse', 1.0) * mse +
        weights.get('smooth', 0.0) * smooth +
        weights.get('start', 0.0) * start +
        weights.get('tail', 0.0) * tail
    )
    return total_loss


def fit_lrf_model(rainfall, true_rdii, init_params, bounds, knots, k, strategy, weights):
    """
    Multi-phase training interface.
    strategy: list of dicts, each with 'name', 'method', 'opt_mask'
    weights: dict of loss weights
    """
    all_results = {}
    params = np.array(init_params)

    for phase in strategy:
        name = phase['name']
        method = phase.get('method', 'L-BFGS-B')
        opt_mask = np.array(phase['opt_mask'])

        def partial_loss(sub_params):
            full = params.copy()
            full[opt_mask] = sub_params
            return loss_function(full, rainfall, true_rdii, knots, k, weights)

        sub_init = params[opt_mask]
        sub_bounds = [b for i, b in enumerate(bounds) if opt_mask[i]]
        result = minimize(partial_loss, sub_init, method=method, bounds=sub_bounds)

        # update full params
        params[opt_mask] = result.x
        all_results[name] = result

    return params, all_results


# Example default strategy configuration
def get_default_strategy(n_c):
    return [
        {"name": "phase1", "method": "Powell", "opt_mask": [True] + [False] * n_c + [False]*4},
        {"name": "phase2", "method": "L-BFGS-B", "opt_mask": [False] + [True] * n_c + [False]*4},
        {"name": "phase3", "method": "L-BFGS-B", "opt_mask": [False] * (1+n_c) + [True]*4},
    ]
