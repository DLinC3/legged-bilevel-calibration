# src/bilevel/exports.py

import os
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
from .kinematics import build_zeroed

def _quat_to_rpy_batch(q_arr):
    """
    q_arr: (T, 4) as [qx,qy,qz,qw]
    returns (T, 3) as [roll, pitch, yaw]
    """
    T = q_arr.shape[0]
    rpy = np.zeros((T, 3), dtype=float)
    for k in range(T):
        q = np.asarray(q_arr[k], dtype=float)
        nq = np.linalg.norm(q)
        if nq > 0:
            q = q / nq
        R = pin.Quaternion(q).toRotationMatrix()
        rpy[k, :] = pin.rpy.matrixToRpy(R)
    return rpy

def export_fw_snapshot_csv(prefix,
                           xmhe_traj, x_data, foot_data,
                           start_idx, horizon,
                           base_ofs=None,
                           tip_ofs=None,
                           model=None, fids=None,
                           q_meas_win=None,
                           add_tip_offset=True,
                           out_dir=".",
                           base_ofs_calib=None,
                           tip_ofs_calib=None):
    """
    Dump current MHE trajectory, GT, and feet to several CSVs so that
    you can plot them in MATLAB or pandas.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- slice window ---
    xr = x_data[start_idx:start_idx+horizon+1, :]
    fr = foot_data[start_idx:start_idx+horizon+1, :]
    t  = np.arange(xr.shape[0])

    # estimated
    p_est  = np.asarray(xmhe_traj[:, 0:3], dtype=float)
    v_est  = np.asarray(xmhe_traj[:, 3:6], dtype=float)
    q_est  = np.asarray(xmhe_traj[:, 9:13], dtype=float)
    pf_est = np.asarray(xmhe_traj[:, 16:28], dtype=float).reshape(-1, 4, 3)

    # mocap
    p_gt = np.asarray(xr[:, 0:3], dtype=float)
    q_gt = np.asarray(xr[:, 3:7], dtype=float)
    q_gt = q_gt / (np.linalg.norm(q_gt, axis=1, keepdims=True) + 1e-16)

    vB   = np.asarray(xr[:, 19:22], dtype=float)
    vW_gt = np.zeros_like(vB)
    for k in range(xr.shape[0]):
        Rk = pin.Quaternion(q_gt[k]).toRotationMatrix()
        vW_gt[k] = Rk @ vB[k]

    # offsets
    if base_ofs is None:       base_ofs = np.zeros(3, dtype=float)
    if base_ofs_calib is None: base_ofs_calib = np.zeros(3, dtype=float)
    base_total = np.asarray(base_ofs).reshape(3,) + np.asarray(base_ofs_calib).reshape(3,)

    if tip_ofs is None:        tip_ofs = np.zeros(12, dtype=float)
    if tip_ofs_calib is None:  tip_ofs_calib = np.zeros(12, dtype=float)
    tip_total = np.asarray(tip_ofs).reshape(12,) + np.asarray(tip_ofs_calib).reshape(12,)

    # apply base offset in world
    p_ofs_world = np.vstack([pin.Quaternion(q_gt[k]).toRotationMatrix() @ base_total
                             for k in range(xr.shape[0])])
    p_plot = p_est + p_ofs_world

    # apply tip offsets to feet
    pf_plot = pf_est.copy()
    if add_tip_offset and (model is not None) and (fids is not None) and (q_meas_win is not None):
        data_loc = model.createData()
        assert q_meas_win.shape[0] == xr.shape[0]
        for k in range(xr.shape[0]):
            q_i = q_meas_win[k, :]
            v_dummy = np.zeros(model.nv)
            q_zero, _ = build_zeroed(q_i, v_dummy)
            pin.forwardKinematics(model, data_loc, q_zero)
            pin.updateFramePlacements(model, data_loc)
            for j, fid in enumerate(fids):
                jid  = model.frames[fid].parentJoint
                Rj_i = np.asarray(data_loc.oMi[jid].rotation, dtype=float)
                r_off = tip_total[3*j:3*j+3]
                pf_plot[k, j, :] = pf_plot[k, j, :] + (Rj_i @ r_off)

    # reference feet
    pf_ref = np.asarray(fr, dtype=float).reshape(-1, 4, 3)

    # save
    pos_mat = np.column_stack([t, p_plot, p_gt])
    np.savetxt(os.path.join(out_dir, f"{prefix}_pos.csv"),
               pos_mat, delimiter=",",
               header="t,px_est,py_est,pz_est,px_gt,py_gt,pz_gt", comments="")

    vel_mat = np.column_stack([t, v_est, vW_gt])
    np.savetxt(os.path.join(out_dir, f"{prefix}_vel.csv"),
               vel_mat, delimiter=",",
               header="t,vx_est,vy_est,vz_est,vx_gt,vy_gt,vz_gt", comments="")

    # feet
    feet_labels = ["FR","FL","RR","RL"]
    cols = [t]; headers = ["t"]
    for j, name in enumerate(feet_labels):
        for d, axis in enumerate(["x","y","z"]):
            cols += [pf_plot[:, j, d], pf_ref[:, j, d]]
            headers += [f"{name}_{axis}_est", f"{name}_{axis}_gt"]
    feet_mat = np.column_stack(cols)
    np.savetxt(os.path.join(out_dir, f"{prefix}_feet.csv"),
               feet_mat, delimiter=",", header=",".join(headers), comments="")

    # rpy
    rpy_est = _quat_to_rpy_batch(q_est)
    rpy_gt  = _quat_to_rpy_batch(q_gt)
    rpy_mat = np.column_stack([t, rpy_est, rpy_gt])
    np.savetxt(os.path.join(out_dir, f"{prefix}_rpy.csv"),
               rpy_mat, delimiter=",",
               header="t,roll_est,pitch_est,yaw_est,roll_gt,pitch_gt,yaw_gt", comments="")

def _build_zeroed(q_i, v_i, model_nv=None):
    """
    Create "zeroed" base pose/velocity for kinematic evaluations:
    - zero base position
    - unit quaternion
    - zero base linear & angular velocity
    """
    q_zero = np.array(q_i, dtype=float).copy()
    v_zero = np.array(v_i, dtype=float).copy()
    q_zero[0:3] = [0.0, 0.0, 0.0]
    q_zero[3:7] = [0.0, 0.0, 0.0, 1.0]
    v_zero[0:3] = [0.0, 0.0, 0.0]
    v_zero[3:6] = [0.0, 0.0, 0.0]
    return q_zero, v_zero

def plot_results(xmhe_traj,
                 x_data,
                 foot_data,
                 start_idx,
                 horizon,
                 save_prefix=None,
                 base_ofs=None,
                 tip_ofs=None,
                 model=None,
                 fids=None,
                 q_meas_win=None,
                 add_tip_offset=True):
    """
    Quick visualization utility (used by FW loop):
      - Position/velocity/quaternion vs mocap GT
      - Feet positions (estimated vs mocap), with optional tip offsets applied

    Args:
        xmhe_traj: (H+1, nx) MHE state trajectory
        x_data:    full mocap state array (T, ?), columns assumed as in pipeline
        foot_data: mocap feet positions (T, 12) in world frame
        start_idx, horizon: window indices
        save_prefix: if provided, figures are saved as PNG files (prefix_*). Else plt.show()
        base_ofs: (3,) rigid base offset in body frame; rotated into world by q_gt then added to p_est
        tip_ofs:  (12,) per-foot offsets in each foot's parent joint frame; rotated into world and added to pf
        model, fids, q_meas_win: needed to rotate tip offsets by each foot's parent joint rotation
        add_tip_offset: whether to apply tip offsets on the feet traces
    """
    # --- window slices ---
    xr = x_data[start_idx:start_idx + horizon + 1, :]
    fr = foot_data[start_idx:start_idx + horizon + 1, :]
    t  = np.arange(xr.shape[0])

    # --- estimates pulled from MHE state layout ---
    p_est  = np.asarray(xmhe_traj[:, 0:3], dtype=float)
    v_est  = np.asarray(xmhe_traj[:, 3:6], dtype=float)
    q_est  = np.asarray(xmhe_traj[:, 9:13], dtype=float)
    pf_est = np.asarray(xmhe_traj[:, 16:28], dtype=float).reshape(-1, 4, 3)  # (H+1,4,3)

    # --- mocap ground truth (position + quaternion) ---
    p_gt = np.asarray(xr[:, 0:3], dtype=float)
    q_gt = np.asarray(xr[:, 3:7], dtype=float)
    q_gt = q_gt / (np.linalg.norm(q_gt, axis=1, keepdims=True) + 1e-16)

    # body velocity -> world velocity via GT rotation
    vB   = np.asarray(xr[:, 19:22], dtype=float)
    vW_gt = np.zeros_like(vB)
    for k in range(xr.shape[0]):
        Rk = pin.Quaternion(q_gt[k]).toRotationMatrix()
        vW_gt[k] = Rk @ vB[k]

    # --- apply base offset on position (rotate into world by GT quaternion) ---
    if base_ofs is None:
        base_ofs = np.zeros(3, dtype=float)
    base_ofs = np.asarray(base_ofs, dtype=float).reshape(3,)
    p_ofs_world = np.vstack([pin.Quaternion(q_gt[k]).toRotationMatrix() @ base_ofs
                             for k in range(xr.shape[0])])
    p_plot = p_est + p_ofs_world

    # --- apply tip offsets on feet traces (rotate by each foot's parent joint rotation) ---
    pf_plot = pf_est.copy()
    if add_tip_offset and (tip_ofs is not None) and (model is not None) and (fids is not None) and (q_meas_win is not None):
        tip_ofs = np.asarray(tip_ofs, dtype=float).reshape(12,)
        data_loc = model.createData()
        assert q_meas_win.shape[0] == xr.shape[0], "q_meas_win length must equal horizon+1"
        for k in range(xr.shape[0]):
            q_i = q_meas_win[k, :]
            v_dummy = np.zeros(model.nv)
            q_zero, _ = _build_zeroed(q_i, v_dummy)
            pin.forwardKinematics(model, data_loc, q_zero)
            pin.updateFramePlacements(model, data_loc)
            for j, fid in enumerate(fids):
                jid  = model.frames[fid].parentJoint
                Rj_i = np.asarray(data_loc.oMi[jid].rotation, dtype=float)  # parent joint -> world
                r_off = tip_ofs[3*j:3*j+3]
                pf_plot[k, j, :] = pf_plot[k, j, :] + (Rj_i @ r_off)

    # --- mocap feet in world (reference) ---
    pf_ref = fr.reshape(-1, 4, 3)

    # ================== figures ==================
    # position (world)
    fig1, axs1 = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    labs = ['x', 'y', 'z']
    for i in range(3):
        axs1[i].plot(t, p_plot[:, i], '-', label='est (+base_ofs)')
        axs1[i].plot(t, p_gt[:, i],  '--', label='gt')
        axs1[i].set_ylabel(f'p_{labs[i]}'); axs1[i].grid(True); axs1[i].legend()
    axs1[-1].set_xlabel('t')
    fig1.suptitle('pos (world)'); fig1.tight_layout()

    # velocity (world)
    fig2, axs2 = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for i in range(3):
        axs2[i].plot(t, v_est[:, i], '-', label='est')
        axs2[i].plot(t, vW_gt[:, i], '--', label='gt')
        axs2[i].set_ylabel(f'v_{labs[i]}'); axs2[i].grid(True); axs2[i].legend()
    axs2[-1].set_xlabel('t')
    fig2.suptitle('vel (world)'); fig2.tight_layout()

    # quaternion components
    fig3, axs3 = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    qlabs = ['qx', 'qy', 'qz', 'qw']
    for i in range(4):
        axs3[i].plot(t, q_est[:, i], '-', label='est')
        axs3[i].plot(t, q_gt[:, i],  '--', label='gt')
        axs3[i].set_ylabel(qlabs[i]); axs3[i].grid(True); axs3[i].legend()
    axs3[-1].set_xlabel('t')
    fig3.suptitle('quat'); fig3.tight_layout()

    # feet positions (world)
    fig4, axs4 = plt.subplots(4, 3, figsize=(12, 9), sharex=True)
    feet = ['FR', 'FL', 'RR', 'RL']
    for f in range(4):
        for d in range(3):
            ax = axs4[f, d]
            ax.plot(t, pf_plot[:, f, d], '-', label='est (+tip_ofs)')
            ax.plot(t, pf_ref[:, f, d],  '--', label='mocap')
            if f == 0: ax.set_title(['px','py','pz'][d])
            if d == 0: ax.set_ylabel(feet[f])
            if f == 3: ax.set_xlabel('t')
            if f == 0 and d == 0: ax.legend()
            ax.grid(True)
    fig4.suptitle('feet (world)'); fig4.tight_layout()

    if save_prefix is None:
        plt.show()
    else:
        fig1.savefig(f"{save_prefix}_pos.png",  dpi=200, bbox_inches='tight')
        fig2.savefig(f"{save_prefix}_vel.png",  dpi=200, bbox_inches='tight')
        fig3.savefig(f"{save_prefix}_quat.png", dpi=200, bbox_inches='tight')
        fig4.savefig(f"{save_prefix}_feet.png", dpi=200, bbox_inches='tight')
        plt.close(fig1); plt.close(fig2); plt.close(fig3); plt.close(fig4)

# ---- CSV / history exports ----
def export_iter_est_csv(prefix, xmhe_traj, q_meas_win, out_dir="fw_iters", dt=None):
    """
    Export per-iteration state snapshot:
      t[, t_sec], p_base(3), q_base(4), v_base(3), and joint encoder positions (from q_meas_win[7:])
    """
    if q_meas_win is None:
        raise ValueError("q_meas_win must be provided for exporting joint positions.")
    H1 = xmhe_traj.shape[0]
    if q_meas_win.shape[0] != H1:
        raise ValueError(f"q_meas_win rows ({q_meas_win.shape[0]}) must equal xmhe_traj rows ({H1}).")

    t_idx = np.arange(H1).reshape(-1, 1)
    cols = [t_idx]
    headers = ["t"]
    if dt is not None:
        t_sec = (t_idx * float(dt))
        cols.append(t_sec)
        headers.append("t_sec")

    p_est = np.asarray(xmhe_traj[:, 0:3], dtype=float)
    v_est = np.asarray(xmhe_traj[:, 3:6], dtype=float)
    q_est = np.asarray(xmhe_traj[:, 9:13], dtype=float)

    cols += [p_est, q_est, v_est]
    headers += [
        "p_base_x","p_base_y","p_base_z",
        "q_base_x","q_base_y","q_base_z","q_base_w",
        "v_base_x","v_base_y","v_base_z"
    ]

    nJ = q_meas_win.shape[1] - 7
    if nJ > 0:
        jpos = np.asarray(q_meas_win[:, 7:7+nJ], dtype=float)
        cols.append(jpos)
        headers += [f"joint_pos_{i}" for i in range(nJ)]

    mat = np.column_stack(cols)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_state.csv")
    np.savetxt(path, mat, delimiter=",", header=",".join(headers), comments="")
    print(f"[iter csv] saved: {path} (rows={mat.shape[0]}, cols={mat.shape[1]})")

def save_theta_history_csv(stdHist, P_core, idx_tip, idx_base, out_path="theta_history.csv"):
    """
    Save FW parameter history:
      [iter | core(P_core) | tip(12) | base(3)]
    Takes rows up to the last non-NaN iteration.
    """
    valid_rows = np.where(~np.isnan(stdHist[:, 0]))[0]
    if valid_rows.size == 0:
        raise RuntimeError("stdHist contains no valid rows.")
    last_it = int(valid_rows[-1])
    hist = stdHist[:last_it+1, :]

    it_col = np.arange(last_it+1).reshape(-1, 1)

    core = hist[:, :P_core]
    tip  = hist[:, idx_tip:idx_tip+12]
    base = hist[:, idx_base:idx_base+3]

    headers = ["iter"]
    headers += [f"core_{i}" for i in range(P_core)]
    feet = ["FR","FL","RR","RL"]; axes = ["x","y","z"]
    for leg in feet:
        for ax in axes:
            headers.append(f"tip_{leg}_{ax}")
    headers += ["base_x","base_y","base_z"]

    mat = np.column_stack([it_col, core, tip, base])
    np.savetxt(out_path, mat, delimiter=",", header=",".join(headers), comments="")
    print(f"[theta CSV] saved to {out_path} (rows={mat.shape[0]}, cols={mat.shape[1]})")

def plot_theta_history(stdHist, idx_tip, idx_base, upto_iter, save_prefix=None,
                       tip_labels=None, base_labels=('base_x','base_y','base_z')):
    """
    Plot the evolution of tip(12) and base(3) offsets up to `upto_iter`.
    """
    T = np.arange(upto_iter + 1)

    tip_hist = stdHist[:upto_iter+1, idx_tip:idx_tip+12]
    if tip_labels is None:
        tip_labels = ['FR_x','FR_y','FR_z','FL_x','FL_y','FL_z',
                      'RR_x','RR_y','RR_z','RL_x','RL_y','RL_z']

    fig1, axs = plt.subplots(4, 3, figsize=(12, 9), sharex=True)
    for j in range(12):
        r, c = divmod(j, 3)
        ax = axs[r, c]
        ax.plot(T, tip_hist[:, j], '-', linewidth=1.5)
        ax.grid(True)
        ax.set_ylabel(tip_labels[j])
        if r == 3: ax.set_xlabel('Iteration')
    fig1.suptitle('Tip offsets (12) across iterations')
    fig1.tight_layout()

    base_hist = stdHist[:upto_iter+1, idx_base:idx_base+3]
    fig2, axs2 = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for i in range(3):
        axs2[i].plot(T, base_hist[:, i], '-', linewidth=1.5)
        axs2[i].grid(True)
        axs2[i].set_ylabel(base_labels[i])
    axs2[-1].set_xlabel('Iteration')
    fig2.suptitle('Base offset (3) across iterations')
    fig2.tight_layout()

    if save_prefix is None:
        plt.show()
    else:
        fig1.savefig(f"{save_prefix}_theta_tip.png",  dpi=200, bbox_inches='tight')
        fig2.savefig(f"{save_prefix}_theta_base.png", dpi=200, bbox_inches='tight')
        plt.close(fig1); plt.close(fig2)

def export_fw_snapshot_csv(prefix,
                           xmhe_traj, x_data, foot_data,
                           start_idx, horizon,
                           base_ofs=None,         # from theta (body-frame)
                           tip_ofs=None,          # from theta (12,)
                           model=None, fids=None,
                           q_meas_win=None,
                           add_tip_offset=True,
                           out_dir=".",
                           base_ofs_calib=None,   # calibrated base offset
                           tip_ofs_calib=None):   # calibrated tip offsets (12,)
    """
    Export a full snapshot (pos/vel/feet/rpy) to CSVs for quick inspection.
    Applies (theta + calibrated) offsets.
    """

    xr = x_data[start_idx:start_idx+horizon+1, :]
    fr = foot_data[start_idx:start_idx+horizon+1, :]
    t  = np.arange(xr.shape[0])

    p_est  = np.asarray(xmhe_traj[:, 0:3], dtype=float)
    v_est  = np.asarray(xmhe_traj[:, 3:6], dtype=float)
    q_est  = np.asarray(xmhe_traj[:, 9:13], dtype=float)
    pf_est = np.asarray(xmhe_traj[:, 16:28], dtype=float).reshape(-1, 4, 3)

    p_gt = np.asarray(xr[:, 0:3], dtype=float)
    q_gt = np.asarray(xr[:, 3:7], dtype=float)
    q_gt = q_gt / (np.linalg.norm(q_gt, axis=1, keepdims=True) + 1e-16)

    vB   = np.asarray(xr[:, 19:22], dtype=float)
    vW_gt = np.zeros_like(vB)
    for k in range(xr.shape[0]):
        Rk = pin.Quaternion(q_gt[k]).toRotationMatrix()
        vW_gt[k] = Rk @ vB[k]

    # compose total offsets: theta + calibration
    if base_ofs is None:       base_ofs = np.zeros(3, dtype=float)
    if base_ofs_calib is None: base_ofs_calib = np.zeros(3, dtype=float)
    base_total = np.asarray(base_ofs, dtype=float).reshape(3,) + np.asarray(base_ofs_calib, dtype=float).reshape(3,)

    if tip_ofs is None:        tip_ofs = np.zeros(12, dtype=float)
    if tip_ofs_calib is None:  tip_ofs_calib = np.zeros(12, dtype=float)
    tip_total  = np.asarray(tip_ofs, dtype=float).reshape(12,) + np.asarray(tip_ofs_calib, dtype=float).reshape(12,)

    # base offset rotated into world by mocap quaternion
    p_ofs_world = np.vstack([pin.Quaternion(q_gt[k]).toRotationMatrix() @ base_total
                             for k in range(xr.shape[0])])
    p_plot = p_est + p_ofs_world

    # tip offsets rotated by each foot's parent joint rotation
    pf_plot = pf_est.copy()
    if add_tip_offset and (model is not None) and (fids is not None) and (q_meas_win is not None):
        data_loc = model.createData()
        assert q_meas_win.shape[0] == xr.shape[0], "q_meas_win length must equal horizon+1"
        for k in range(xr.shape[0]):
            q_i = q_meas_win[k, :]
            v_dummy = np.zeros(model.nv)
            q_zero, _ = _build_zeroed(q_i, v_dummy)
            pin.forwardKinematics(model, data_loc, q_zero)
            pin.updateFramePlacements(model, data_loc)
            for j, fid in enumerate(fids):
                jid  = model.frames[fid].parentJoint
                Rj_i = np.asarray(data_loc.oMi[jid].rotation, dtype=float)
                r_off = tip_total[3*j:3*j+3]
                pf_plot[k, j, :] = pf_plot[k, j, :] + (Rj_i @ r_off)

    pf_ref = np.asarray(fr, dtype=float).reshape(-1, 4, 3)

    # rpy (rad) from quaternions
    rpy_est = _quat_to_rpy_batch(q_est)
    rpy_gt  = _quat_to_rpy_batch(q_gt)

    os.makedirs(out_dir, exist_ok=True)

    pos_mat = np.column_stack([t, p_plot, p_gt])
    np.savetxt(os.path.join(out_dir, f"{prefix}_pos.csv"),
               pos_mat, delimiter=",",
               header="t,px_est,py_est,pz_est,px_gt,py_gt,pz_gt", comments="")

    vel_mat = np.column_stack([t, v_est, vW_gt])
    np.savetxt(os.path.join(out_dir, f"{prefix}_vel.csv"),
               vel_mat, delimiter=",",
               header="t,vx_est,vy_est,vz_est,vx_gt,vy_gt,vz_gt", comments="")

    feet_labels = ["FR","FL","RR","RL"]
    cols = [t]; headers = ["t"]
    for j, name in enumerate(feet_labels):
        for d, axis in enumerate(["x","y","z"]):
            cols += [pf_plot[:, j, d], pf_ref[:, j, d]]
            headers += [f"{name}_{axis}_est", f"{name}_{axis}_gt"]
    feet_mat = np.column_stack(cols)
    np.savetxt(os.path.join(out_dir, f"{prefix}_feet.csv"),
               feet_mat, delimiter=",", header=",".join(headers), comments="")

    rpy_mat = np.column_stack([t, rpy_est, rpy_gt])
    np.savetxt(os.path.join(out_dir, f"{prefix}_rpy.csv"),
               rpy_mat, delimiter=",",
               header="t,roll_est,pitch_est,yaw_est,roll_gt,pitch_gt,yaw_gt", comments="")

# keep explicit export list tidy
try:
    __all__
except NameError:
    __all__ = []
for _name in [
    "plot_results",
    "export_iter_est_csv",
    "save_theta_history_csv",
    "plot_theta_history",
    "export_fw_snapshot_csv",
]:
    if _name not in __all__:
        __all__.append(_name)
