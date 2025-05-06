# example_simulation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lrf import simulate_rdii, generate_s, fit_lrf_model, get_default_strategy

# === Load rainfall and RDII data ===
file_path = 'P3Case1_RDII.xlsx'  # Make sure this file is in the same folder
sheet_name = 'Sheet1'
data = pd.read_excel(file_path, sheet_name=sheet_name)
data.index = pd.to_datetime(data['Date'])  # Ensure datetime index

# Resample to hourly
data_resampled = data.resample('h').sum()
data_resampled['RDII'] = data['RDII'].resample('h').first()

# Extract input/output arrays
rainfall = data_resampled['R'].to_numpy()
true_rdii = data_resampled['RDII'].to_numpy()

# === Define model configuration ===
T0_init = 40
n_c = 8
k = 3

# Generate uniform initial knots and coefficients
internal_knots = np.linspace(0, T0_init, n_c - k + 1)
knots = np.concatenate(([0] * k, internal_knots, [T0_init] * k))
c_init = np.ones(n_c)

# Combine into full parameter list: [T0, c0...cn, beta, alpha, gamma, S_th]
init_params = [T0_init] + c_init.tolist() + [0.4, 0.1, 2.0, 0.6]

# Define parameter bounds
bounds = [(20, 100)] + [(0, 10)] * n_c + [(0, 2), (0, 1), (0, 10), (0, 1)]

# Define loss weights
weights = {
    'mse': 1.0,
    'smooth': 0.01,
    'start': 0.01,
    'tail': 0.01
}

# Use default multi-phase training strategy
strategy = get_default_strategy(n_c)

# === Train the model ===
params_opt, log = fit_lrf_model(rainfall, true_rdii, init_params, bounds, knots, k, strategy, weights)

# === Predict using optimized parameters ===
S = generate_s(rainfall, alpha=params_opt[-3])
params_dict = {
    'T0': int(params_opt[0]),
    'c': params_opt[1:1+n_c],
    'knots': knots,
    'k': k,
    'beta': params_opt[-4],
    'alpha': params_opt[-3],
    'gamma': params_opt[-2],
    'S_th': params_opt[-1],
}
pred_rdii = simulate_rdii(rainfall, S, params_dict)

# === Plot results ===
plt.figure(figsize=(10, 4))
plt.plot(data_resampled.index, true_rdii, label='Observed RDII', alpha=0.6)
plt.plot(data_resampled.index, pred_rdii, label='Predicted RDII', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('RDII')
plt.title('LRF Model: Observed vs Predicted RDII')
plt.legend()
plt.tight_layout()
plt.show()
