# src/bilevel/data_io.py

import os
import numpy as np
from .config import BilevelConfig

def read_from_csv(path: str) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data

def downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return arr
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.ndim != 2:
        raise ValueError("downsample: input must be 2D array")
    return arr[::factor]

def load_dataset(cfg: BilevelConfig):
    """
    Loads all raw CSVs, downsamples, and returns a dict.
    """
    def _p(name): return os.path.join(cfg.data_dir, name)

    y_mocap = read_from_csv(_p("y_mocap.csv"))
    u_mocap = read_from_csv(_p("u_mocap.csv"))
    q_mocap = read_from_csv(_p("q_mocap.csv"))
    v_mocap = read_from_csv(_p("v_mocap.csv"))
    x_mocap = read_from_csv(_p("x_mocap.csv"))
    foot_mocap = read_from_csv(_p("foot_mocap.csv"))
    contact_mocap = read_from_csv(_p("contact_mocap.csv"))

    ds = cfg.downsample_factor
    y_data = downsample(y_mocap, ds)
    u_data = downsample(u_mocap, ds)
    q_data = downsample(q_mocap, ds)
    v_data = downsample(v_mocap, ds)
    x_data = downsample(x_mocap, ds)
    foot_data = downsample(foot_mocap, ds)
    contact_data = downsample(contact_mocap, ds)

    # you had 4 manual z-offsets for mocap feet
    z_off = np.array([0.01960054, 0.02402977, 0.04499581, 0.03318461], dtype=float)
    foot_data = foot_data.copy()
    foot_data[:,  2] += z_off[0]  # FR z
    foot_data[:,  5] += z_off[1]  # FL z
    foot_data[:,  8] += z_off[2]  # RR z
    foot_data[:, 11] += z_off[3]  # RL z

    # simple force -> contact gate
    contact_data = (contact_data >= cfg.contact_thres).astype(float)

    return dict(
        y_data=y_data,
        u_data=u_data,
        q_data=q_data,
        v_data=v_data,
        x_data=x_data,
        foot_data=foot_data,
        contact_data=contact_data,
    )

