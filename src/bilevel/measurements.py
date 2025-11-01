# src/bilevel/measurements.py

import numpy as np
from .kinematics import build_zeroed, compute_pf_meas, compute_yv_kin

def build_y_and_dY(model, data, fids, q_meas, v_meas, u_meas, tip_offset):
    """
    Build stacked measurement y and its sensitivity wrt 12 tip offsets.

    Inputs:
      model, data, fids : pinocchio stuff
      q_meas, v_meas, u_meas : arrays of length (H+1, nq/nv/6)
      tip_offset : (12,) array, current guess of tip offsets

    Returns:
      y_meas      : (H+1, 24)
      dY_doffset  : (24*(H+1), 12)
    """
    H1 = q_meas.shape[0]
    pf_list, yv_list = [], []
    dY_blocks = []

    for k in range(H1):
        q_i = q_meas[k, :]
        v_i = v_meas[k, :]
        u_i = u_meas[k, :]

        q_zero, v_zero = build_zeroed(q_i, v_i)

        pf_k,  J_pf_off = compute_pf_meas(model, data, q_zero, fids, tip_offset)
        yv_k,  J_v_off  = compute_yv_kin(model, data, q_zero, v_zero, u_i[3:6], fids, tip_offset)

        pf_list.append(pf_k)
        yv_list.append(yv_k)
        dY_blocks.append(np.vstack([J_v_off, J_pf_off]))

    pf_meas = np.vstack(pf_list)
    v_B_kin = np.vstack(yv_list)
    y_meas  = np.hstack([v_B_kin, pf_meas])
    dY_doffset = np.vstack(dY_blocks)

    return y_meas, dY_doffset

