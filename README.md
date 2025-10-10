# Learnable Response Function (LRF)

## 💡 Overview
📑This repository accompanies the paper "A Simple Model for Long-Term Prediction of Sewage Flow in a Changing Climate"

📈Rainfall-derived inflow and infiltration (RDII) is a major contributor to variability in sewage flow, particularly under the increasing frequency of extreme weather events driven by climate change . The developed RDII model simulates the response of sewer systems to rainfall-induced inflow, demonstrating strong potential for predicting long-term sewage dynamics.

🧩This repository introduces the Learnable Response Function (LRF), the fundamental modelling framework, with the RDII model provided as a quickstart example [here](example_application.ipynb).




## ✨ Highlights
- **Automatic** learning of response function based on B-spline
- **Adaptive** response adjustment mechanism based on latest data
- **Advanced** multi-phase optimization featuring tailored regularization
- **Simple** and **extensible** API for beginners



## ⚙️ Installation

**Pre-requisites**
```
Python 3.9.7 or higher
pip
```

**Installation via github**
```bash
pip install git+https://github.com/JingyuZoe/lrf.git
```

**For developers**

```bash
git clone https://github.com/JingyuZoe/lrf.git   # Clone this repo
cd lrf                                           # Go to the directory
pip install -e .                                 # Install the package in editable mode
```



## 🚀 Quick Start

**Example Simulation**
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

**Training with Real Data**
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

## 📚 Tutorial

Get started with [example_application.ipynb](./example_application.ipynb) notebook.


## 🗂️ File Structure

**lrf** — Core module implementing LRF simulation, response modeling, training, and utilities.

| File           | Description |
|----------------|-------------|
| `model.py`     | Implements the core `simulate_rdii` logic. |
| `response.py`  | Computes B-spline responses and dynamic κ(S). |
| `train.py`     | Training pipeline with custom loss and phased strategy. |
| `utils.py`     | Helper functions (e.g., `generate_s`). |
| `__init__.py`  | Initializes the `lrf` package. |

**Note:** Files are organized by functionality to simplify maintenance and future extensions.


## ✉️ Contact
If you have any questions, please reach out to Jingyu Ge at jingyu.ge@uq.edu.au

## ⚖️ License
This project is licensed under the [MIT License](LICENSE.txt).
