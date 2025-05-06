# Learnable Response Function (LRF)

A Python package for simulating and training dynamic rainfall-derived inflow and infiltration (RDII) models using learnable and adaptive response functions.

---

## 📦 Features
- B-spline-based response function with learnable control points
- Saturation-dependent dynamic response adjustment via \( \kappa(S) \)
- Multi-phase training strategy with user-defined optimization masks
- Simple and extensible API for simulation and training

---

## 📁 Installation

### Option 1: Install from GitHub
```bash
pip install git+https://github.com/JingyuZoe/lrf.git
```

### Option 2: Local install
Clone this repo and run:
```bash
pip install -e .
```

---

## 🚀 Quick Start

### Example Simulation
```python
from lrf import simulate_rdii, generate_s

R = ...  # rainfall time series
S = generate_s(R, alpha=0.1)
params = {
    'T0': 40,
    'c': [1.0]*8,
    'knots': [...],
    'k': 3,
    'beta': 0.4,
    'alpha': 0.1,
    'gamma': 2.0,
    'S_th': 0.6,
}
II = simulate_rdii(R, S, params)
```

### Training with Real Data
```python
from lrf import fit_lrf_model, get_default_strategy

params_opt, log = fit_lrf_model(
    rainfall=R,
    true_rdii=Y,
    init_params=[...],
    bounds=[...],
    knots=knots,
    k=3,
    strategy=get_default_strategy(n_c=8),
    weights={'mse': 1.0, 'smooth': 0.01, 'start': 0.01, 'tail': 0.01}
)
```

---

## 📂 Files and Structure
```
lrf/
├── model.py           # simulate_rdii logic
├── response.py        # b-spline response + dynamic κ(S)
├── train.py           # training with custom loss + phase strategy
├── utils.py           # helper functions (e.g., generate_s)
└── __init__.py
```

---

## 📄 License
UQ License

---

## 👤 Author
[Jingyu Zoe] – PhD Candidate