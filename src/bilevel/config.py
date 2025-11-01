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
    This is original setup:
      weight_para = w_arrival + w_measurement + w_noise
    """
    w_arrival = [
        # base position (x,y,z):
        50000, 50000, 50000,
        # base velocity (x,y,z):
        30000, 30000, 30000,
        # accel bias / accel walk (3):
        10000, 10000, 10000,
        # quaternion (x,y,z,w):
        80000, 80000, 80000, 80000,
        # gyro bias / omega walk (3):
        5000, 5000, 5000,
        # foot positions (12):
        20000, 20000, 20000,   # FR
        20000, 20000, 20000,   # FL
        20000, 20000, 20000,   # RR
        20000, 20000, 20000    # RL
    ]

    w_measurement = [
        # R_q diag
        600, 600, 600,
        # R_q off-diag
        0, 0, 0,
        # R_qdot diag
        600, 600, 600,
        # R_qdot off-diag
        0, 0, 0,
    ]

    w_noise = [
        # Qa_blk (3x3) diag
        70000, 70000, 70000,
        # Qa_blk off-diag
        0, 0, 0,
        # Qw_blk (3x3) diag
        70000, 70000, 70000,
        # Qw_blk off-diag
        0, 0, 0,
        # accel walk (3)
        2000, 2000, 2000,
        # omega walk (3)
        2000, 2000, 2000,
        # swing foot process (3)
        400, 400, 400,
        # stance foot process (3)
        1000000, 1000000, 1000000,
    ]

    return w_arrival + w_measurement + w_noise

