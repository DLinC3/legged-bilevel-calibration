# Simultaneous Calibration of Noise Covariance and Kinematics for State Estimation of Legged Robots via Bi-level Optimization
---

## Overview

This repository implements a bi-level optimization framework for the joint calibration of noise covariances and kinematic parameters in legged-robot state estimation. Conventional legged-robot estimators depend critically on process and measurement noise covariances that are typically hand-tuned. Inaccurate tuning or uncertain kinematics can lead to inconsistent or drifting estimates. This work eliminates manual tuning by embedding the estimator itself inside an optimization loop.

At the lower level, a Full-Information Estimator(FIE) reconstructs the base pose, velocity, contact states, and bias terms given proprioceptive data (IMU, encoders, contact).
At the upper level, we treat all covariance matrices and kinematic parameters—including leg-tip offsets and base-to-mocap alignment—as optimization variables, and minimize a trajectory-level loss measuring the deviation between estimated and ground-truth motion.
The coupling between the two levels is realized by differentiating through the estimator’s KKT conditions, enabling gradient-based updates of physically-constrained parameters. The outer optimization adopts an adaptive Frank–Wolfe algorithm with a convex Linear Minimization Oracle (LMO) and Armijo line search to ensure stable convergence within the positive-definite and box-bounded constraint set.

The detailed mathematical formulation and derivations underlying this framework are presented in the paper. [[ArXiv]](https://arxiv.org/pdf/2510.11539)

In short:

> **FIE (inner)  ⟶  KKT-based Differentiation  ⟶  Frank–Wolfe (outer)  ⟶  calibrated legged state estimator.**

This repo is split into modules:

- data loading and downsampling,
- Pinocchio-based kinematics and measurement construction,
- Estimator definition,
- the bilevel / FW solver,
- plotting and CSV exports.

---

## Installation / Dependencies

This repository requires a standard robotics + optimization Python stack, plus Pinocchio and CasADi for kinematics and estimator-in-the-loop differentiation. Below is the concise list of prerequisites and their official installation pages.

### 1. Core Runtime
- **Python 3.10+**  
  https://www.python.org/downloads/

### 2. Python Packages
- **CasADi ≥ 3.6** (symbolic modeling, codegen, and differentiating the estimator)  
  https://web.casadi.org/get/
- **Pinocchio** (rigid-body kinematics, frame placements, Jacobians)  
  https://stack-of-tasks.github.io/pinocchio/  
  Conda (recommended): https://anaconda.org/conda-forge/pinocchio
- **NumPy / SciPy / Matplotlib** (numerics, utilities, plotting)
  - https://pypi.org/project/numpy/
  - https://pypi.org/project/scipy/
  - https://pypi.org/project/matplotlib/
- **pypardiso** (fast sparse linear solver used when differentiating through the estimator’s KKT system)  
  https://github.com/haasad/PyPardisoProject
- **CVXPY** (outer-level Frank–Wolfe linear minimization oracle)  
  https://www.cvxpy.org/install/

### 3. Convex Solvers for CVXPY
- **MOSEK**
  - Install: https://docs.mosek.com/latest/install/installation.html
  - Academic license: https://www.mosek.com/products/academic-licenses/
  - Python package: https://pypi.org/project/Mosek/
- **IPOPT**  
  https://coin-or.github.io/Ipopt/INSTALL.html  
---

## Launch Example

Your repo expects the following tree:

```text
legged-bilevel-calib/
├── models/
│   └── B1.urdf
├── data/
│   ├── y_mocap.csv
│   ├── u_mocap.csv
│   ├── q_mocap.csv
│   ├── v_mocap.csv
│   ├── x_mocap.csv
│   ├── foot_mocap.csv
│   └── contact_mocap.csv
├── casadi_cache/
│   └── B1_H3000/
└── external_libs/
    ├── libyv_and_J_codegen.so
    └── libpf_and_J_codegen.so
```

If you rename the CSVs or move the URDF, update `src/bilevel/config.py` or `src/bilevel/data_io.py`.

---

> **First launch is slow, this is expected.**  
> The first time you run the code on a given horizon (e.g. 3000 steps), CasADi will:
> 1. build and differentiate the full KKT system for the estimator,
> 2. build the Jacobians w.r.t. states, measurements, weight parameters, and the big stacked G,
> 3. build the quaternion-alignment loss and its gradient (`dL_dQ_fn`, `L_att_fn`),
> 4. save all of them in `casadi_cache/B1_H3000/`.
> Next runs will simply **load** from this cache and start the FW loop immediately.

Run:

```bash
# from repo root
python -m src.bilevel.run_bilevel
```

You should see something like:

```text
[cache miss] ... Rebuilding derivatives...
=== Frank–Wolfe iter 1 ===
loss = ...
KKT ||.||_inf = ...
LMO status: optimal
Armijo (single) gamma=...
...
```

and new files under `outputs/`:

- `outputs/cur_pos.png`, `outputs/cur_vel.png`, ...
- `outputs/fw_iters/iter_000_state.csv`, `iter_001_state.csv`, ...
- `outputs/theta_history.csv`

---

## Code Structure

```text
/
├── README.md
├── requirements.txt
├── models/
├── data/
├── casadi_cache/
├── external_libs/
├── src/
│   └── bilevel/
│       ├── config.py         # paths, horizon, start_idx, basic constants
│       ├── data_io.py        # CSV loading, downsampling, window slicing
│       ├── kinematics.py     # Pinocchio FK + Jacobians + zero-base helper
│       ├── measurements.py   # build_y_and_dY(...) and G_meas builder
│       ├── codegen.py        # load CasADi external shared libraries
│       ├── exports.py        # plotting + CSV export (plot_results in here)
│       ├── fw_solver.py      # Frank–Wolfe outer loop + LMO (cvxpy) + Armijo
│       ├── run_bilevel.py    # main entry point
│       ├── utils.py          # skew(), unflatten(), leg column helpers
│       ├── dynamics/
│       │   └── srb_dynamics.py  # original SRB dynamics, minimal edits
│       └── estimator/
│           └── srb_mhe.py       # SRB-based estimator used as the inner problem
│   └── MATLAB-planar-five-linkage-robot-example/ # This folder contains a 2-D, planar five-link walker example in MATLAB
│       ├── main.m               # runs the walker sim, adds noise, and launches the outer calibration loop
│       ├── estimation_FIE.m     # defines the full-information estimator used by the MATLAB demo
│       ├── plot_FIE.m           # plots estimator trajectories against ground truth for quick inspection

```

- **`src/bilevel/run_bilevel.py`**  
  ties everything together: load data → build measurements/G → run estimator once → start Frank-Wolfe step.

- **`src/bilevel/fw_solver.py`**  
  implements the outer Frank–Wolfe iterations, including:
  - KKT linearization at current (Z*, θ),
  - sensitivity solve with `pypardiso`,
  - gradient assembly (core weights + 12 tip + 3 base),
  - cvxpy LMO with PSD/box constraints,
  - Armijo backtracking,
  - calling `exports.py` to save plots and CSV each iter.

- **`src/bilevel/kinematics.py` / `measurements.py`**  
  do all the Pinocchio work: for every frame in the horizon, they build
  - foot position measurement (world) and ∂/∂tip,
  - foot velocity measurement (world) and ∂/∂tip,
  - 24×9 G matrix per step (4 feet × (v block + p block)).

- **`src/estimator/srb_mhe.py`**  
  is kept very close to your original version. It:
  - defines the estimator cost with per-leg contact gates,
  - builds the KKT and its derivatives (`diffKKT()`),
  - builds the quaternion-alignment loss and gradient (`diffquat()`),
  - supports **caching** to disk via `load_or_build_derivatives(...)`.

---

## Citation

If you find this code useful in your research, please cite the original paper