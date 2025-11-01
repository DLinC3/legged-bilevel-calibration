# src/bilevel/config.py

import os
from dataclasses import dataclass, field

@dataclass
class BilevelConfig:
    # ---------------------------
    # paths
    # ---------------------------
    repo_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir: str = field(init=False)
    urdf_path: str = field(init=False)
    casadi_cache_dir: str = field(init=False)
    external_lib_dir: str = field(init=False)

    # ---------------------------
    # MHE / dataset settings
    # ---------------------------
    start_idx: int = 22000
    horizon: int = 3000
    dt_mhe: float = 0.005
    downsample_factor: int = 1

    # ---------------------------
    # Frank–Wolfe settings
    # ---------------------------
    max_fw_iters: int = 75
    armijo_rho: float = 1e-4
    armijo_beta: float = 0.5
    armijo_gamma_init: float = 0.25

    # tip / base bounds
    tip_bound: float = 0.10     # [-0.1, 0.1] m
    base_bound: float = 0.50    # [-0.5, 0.5] m

    # (used for LMO box)
    big_box: float = 1e6
    arrival_min: float = 1e3
    arrival_max: float = 1e6

    # LMO PSD
    eps_psd: float = 1e-9
    trace_cap: float = 1e6

    # contact threshold (to binarize raw contact)
    contact_thres: float = 100.0

    def __post_init__(self):
        self.data_dir = os.path.join(self.repo_root, "data")
        self.urdf_path = os.path.join(self.repo_root, "models", "B1.urdf")
        self.casadi_cache_dir = os.path.join(self.repo_root, "casadi_cache", "B1_H3000")
        self.external_lib_dir = os.path.join(self.repo_root, "external_libs")


def default_weight_vector():
    """
    This is your original concatenation:
      weight_para = w_arrival + w_measurement + w_noise
    I keep the numbers exactly so you get bitwise identical behavior.
    """
    w_arrival = [
        76856.5238, 3937.66335, 76034.4139, 76866.4983, 3933.43176, 76866.304,
        76456.5967, 34071.9027, 1.66866290e+05, 1.66864129e+05, 33934.5599, 1.66434989e+05,
        33933.6452, 1.66558707e+05, 34024.6321, 34065.243, 4039.40503, 4108.43315,
        4064.68587, 76470.8917, 4068.06991, 4068.43011, 3934.71836, 4066.53934,
        3934.18209, 4221.47295, 4045.64797, 4064.49198
    ]

    w_measurement = [
        683.353989, 683.333695, 683.334597, 666.66717, 666.668991, 666.666801,
        683.334057, 66716.6662, 683.333678, 666.666946, 666.666843, 666.666743
    ]

    w_noise = [
        67666.5389, 67666.1789, 67666.6653, 666.789331, 66666.6263, 666.698867,
        66766.6556, 700.001095, 700.000764, 666.667215, 666.676632, 666.667307,
        71662.483, 2334.06604, 71666.5611, 2336.10964, 2334.9465, 2333.39454,
        0.00400003408, 0.0699999719, 0.00400641566, 1.00000000e+06, 1.00000000e+06, 1.00000000e+06
    ]

    return w_arrival + w_measurement + w_noise

