# src/bilevel/fw_solver.py

import os
import numpy as np
import scipy.sparse as sp
from pypardiso import spsolve
import cvxpy as cp
import pinocchio as pin

from casadi import DM

from .measurements import build_y_and_dY
from .kinematics import (
    build_zeroed,
    compute_Gv_leg_blocks_body,
    compute_Gp_leg_blocks_body,
    pack_G24x9,
    dGk_dtip_from_codegen,
)
from .exports import (
    plot_results,
    export_fw_snapshot_csv,
    export_iter_est_csv,
    save_theta_history_csv,
    plot_theta_history,
)
from .config import BilevelConfig, default_weight_vector


def outer_loss(xmhe_traj, x_data, foot_data, start_idx, horizon, L_val,
               w_p=1.0, w_v=3.0, w_pfoot=0.5, w_q=2.0, p_ofs=None):
    """
    Same as your original loss(), but English docs:
    - position: estimator position + R(q_gt)*p_ofs  vs mocap position
    - velocity: estimator v_W vs mocap v_W
    - foot pos: estimator feet vs mocap feet
    - plus attitude loss from MHE
    """
    if p_ofs is None:
        p_ofs = np.zeros(3, dtype=float)

    xr = x_data[start_idx:start_idx+horizon+1, :]
    fr = foot_data[start_idx:start_idx+horizon+1, :]

    p_mhe  = xmhe_traj[:, 0:3]
    v_mhe  = xmhe_traj[:, 3:6]
    pf_mhe = xmhe_traj[:, 16:28].reshape(-1, 4, 3)

    p_mocap = xr[:, 0:3]
    q_gt = xr[:, 3:7]
    q_gt = q_gt / (np.linalg.norm(q_gt, axis=1, keepdims=True))
    vB = xr[:, 19:22]
    vW_gt = np.zeros_like(vB)
    for k in range(xr.shape[0]):
        R = pin.Quaternion(q_gt[k]).toRotationMatrix()
        vW_gt[k] = R @ vB[k]

    # position with world-rotated base offset
    E_p = 0.0
    for k in range(p_mhe.shape[0]):
        Rk = pin.Quaternion(q_gt[k]).toRotationMatrix()
        rk = p_mhe[k] + (Rk @ p_ofs.reshape(3,)) - p_mocap[k]
        E_p += float(rk @ rk)

    pf_mocap = fr.reshape(-1, 4, 3)
    E_v  = np.sum((v_mhe - vW_gt)**2)
    E_pf = np.sum((pf_mhe - pf_mocap)**2)

    total = w_p * E_p + w_v * E_v + w_pfoot * E_pf + w_q * float(L_val)
    breakdown = {"E_p":E_p, "E_v":E_v, "E_pf":E_pf, "L_q":float(L_val),
                 "w_p":w_p, "w_v":w_v, "w_pfoot":w_pfoot, "w_q":w_q}
    return total, breakdown


def dldx(xmhe_traj, x_data, foot_data, start_idx, horizon, dL_dq_mat,
         w_p=1.0, w_v=3.0, w_pfoot=0.5, w_q=2.0, p_ofs=None):
    """
    Your original dldx() but in English.
    We return (1, (H+1)*nx) gradient wrt stacked states.
    """
    if p_ofs is None: p_ofs = np.zeros(3, dtype=float)

    H1, nx = xmhe_traj.shape
    assert H1 == horizon + 1
    assert dL_dq_mat.shape == (H1, 4)

    xr = x_data[start_idx:start_idx+horizon+1, :]
    fr = foot_data[start_idx:start_idx+horizon+1, :]

    p_mhe  = xmhe_traj[:, 0:3]
    v_mhe  = xmhe_traj[:, 3:6]
    q_mhe  = xmhe_traj[:, 9:13]
    pf_mhe = xmhe_traj[:, 16:28].reshape(H1, 4, 3)

    q_gt = xr[:, 3:7]
    q_gt = q_gt / (np.linalg.norm(q_gt, axis=1, keepdims=True))
    vB = xr[:, 19:22]
    vW_gt = np.zeros_like(vB)
    for k in range(H1):
        R = pin.Quaternion(q_gt[k]).toRotationMatrix()
        vW_gt[k] = R @ vB[k]

    p_mocap = xr[:, 0:3]
    pf_mocap = fr.reshape(H1, 4, 3)

    blocks = []
    for k in range(H1):
        Rk = pin.Quaternion(q_gt[k]).toRotationMatrix()
        rk_pos = p_mhe[k] + (Rk @ np.asarray(p_ofs).reshape(3,)) - p_mocap[k]
        dp  = 2.0 * w_p * rk_pos
        dv  = 2.0 * w_v * (v_mhe[k] - vW_gt[k])
        dq  = w_q * np.asarray(dL_dq_mat[k]).reshape(4,)
        dpf = 2.0 * w_pfoot * (pf_mhe[k] - pf_mocap[k])
        blocks.append(np.concatenate([dp, dv, dq, dpf.reshape(12,)], axis=0))
    dLdx = np.concatenate(blocks, axis=0).reshape(1, -1)
    return dLdx


def run_frank_wolfe(cfg: BilevelConfig,
                    srbMHE,
                    srbDyn,
                    pin_model,
                    pin_data,
                    fids,
                    dataset,
                    f_yv,
                    f_pf):
    """
    The main outer-loop driver.
    """
    start_idx = cfg.start_idx
    H         = cfg.horizon

    # --- slice window data ---
    u_meas = dataset["u_data"][start_idx:start_idx+H+1, :]
    q_meas = dataset["q_data"][start_idx:start_idx+H+1, :]
    v_meas = dataset["v_data"][start_idx:start_idx+H+1, :]
    x_data = dataset["x_data"]
    foot_data = dataset["foot_data"]
    contact_seq = dataset["contact_data"][start_idx:start_idx+H+1, :]

    # --- build static G_meas (24x9 per step from model) ---
    G_list = []
    for k in range(H+1):
        q_i = q_meas[k, :]
        v_i = v_meas[k, :]
        u_i = u_meas[k, :]
        q_zero, v_zero = build_zeroed(q_i, v_i)
        omega_B = u_i[3:6].copy()
        Gv_leg = compute_Gv_leg_blocks_body(pin_model, pin_data, q_zero, v_zero, omega_B, fids)
        Gp_leg = compute_Gp_leg_blocks_body(pin_model, pin_data, q_zero, fids)
        G_list.append(pack_G24x9(Gv_leg, Gp_leg))
    G_meas = np.stack(G_list, axis=0)
    G_vec  = DM(G_meas.reshape(-1, 1))

    # --- initial tip/base offset ---
    tip_offset0 = np.zeros(12, dtype=float)
    base_offset0 = np.zeros(3, dtype=float)

    # --- initial measurement (with zero offsets) ---
    from .measurements import build_y_and_dY
    y_meas, dY_doffset = build_y_and_dY(
        pin_model, pin_data, fids,
        q_meas, v_meas, u_meas,
        tip_offset0
    )

    # --- initial x_hat from GT ---
    x0 = x_data[start_idx, :]
    p0  = x0[0:3].reshape(3,1)
    q0v = x0[3:7] / (np.linalg.norm(x0[3:7]) + 1e-16)
    q0  = q0v.reshape(4,1)
    R0  = pin.Quaternion(q0v).toRotationMatrix()
    vB0 = x0[19:22].reshape(3,1)
    vW0 = (R0 @ vB0).reshape(3,1)

    # build feet
    q_full0 = np.zeros(pin_model.nq)
    q_full0[0:3]  = x0[0:3]
    q_full0[3:7]  = q0v
    q_full0[7:19] = x0[7:19]
    pin.forwardKinematics(pin_model, pin_data, q_full0)
    pin.updateFramePlacements(pin_model, pin_data)
    pf0 = np.zeros((12,1))
    for j, fid in enumerate(fids):
        pf0[3*j:3*j+3,0] = pin_data.oMf[fid].translation

    x_hat = np.vstack([p0, vW0, np.zeros((3,1)), q0, np.zeros((3,1)), pf0])

    # --- init MHE ---
    weight_para0 = default_weight_vector()
    srbMHE.load_or_build_derivatives(cfg.casadi_cache_dir)

    opt_sol = srbMHE.MHEsolver(
        y_meas, u_meas, x_hat,
        weight_para0, H,
        contact_seq, G_meas
    )
    xmhe_traj = opt_sol["state_traj_opt"]
    noise_traj = opt_sol["noise_traj_opt"]
    costate_traj = opt_sol["costate_ipopt"]

    # --- attitude loss and total loss ---
    q_est_win   = xmhe_traj[:, 9:13]
    q_mocap_win = x_data[start_idx:start_idx+H+1, 3:7]
    grad_vec = srbMHE.dL_dQ_fn(q=DM(q_est_win.reshape(-1, 1)),
                               qm=DM(q_mocap_win.reshape(-1, 1)))["dL_dQ"]
    dL_dq_mat = np.array(grad_vec).reshape(H+1, 4)
    L_val = srbMHE.L_att_fn(q=DM(q_est_win.reshape(-1, 1)),
                            qm=DM(q_mocap_win.reshape(-1, 1)))["L"]
    loss_old, _ = outer_loss(
        xmhe_traj, x_data, foot_data, start_idx, H,
        L_val, 1, 3, 0.5, 2, p_ofs=np.zeros(3)
    )

    # assemble full theta = [core weights | 12 tip | 3 base]
    P_core = srbMHE.weight_para.size2()
    theta = np.array(weight_para0 + tip_offset0.tolist() + base_offset0.tolist(), dtype=float)

    # history buffers
    maxIter = cfg.max_fw_iters
    lossHist     = np.full((maxIter,), np.nan)
    gradHist     = np.full((maxIter,), np.nan)
    dL_exp_hist  = np.full((maxIter,), np.nan)
    dL_act_hist  = np.full((maxIter,), np.nan)
    stdHist      = np.full((maxIter+1, theta.size), np.nan)
    stdHist[0,:] = theta

    idx_tip  = P_core
    idx_base = P_core + 12

    # first export / plot
    plot_results(
        xmhe_traj, x_data, foot_data, start_idx, H,
        save_prefix="cur",
        base_ofs=theta[idx_base:idx_base+3],
        tip_ofs=theta[idx_tip:idx_tip+12],
        model=pin_model, fids=fids,
        q_meas_win=q_meas,
        add_tip_offset=True
    )
    export_fw_snapshot_csv(
        prefix="start",
        xmhe_traj=xmhe_traj,
        x_data=x_data, foot_data=foot_data,
        start_idx=start_idx, horizon=H,
        base_ofs=theta[idx_base:idx_base+3],
        tip_ofs=theta[idx_tip:idx_tip+12],
        model=pin_model, fids=fids, q_meas_win=q_meas,
        add_tip_offset=True,
        out_dir=".",
        base_ofs_calib=theta[-3:],
        tip_ofs_calib=theta[-15:-3],
    )
    export_iter_est_csv(
        prefix="iter_000",
        xmhe_traj=xmhe_traj,
        q_meas_win=q_meas,
        out_dir="fw_iters",
        dt=cfg.dt_mhe
    )

    # ---------------- FW LOOP ----------------
    for Iter in range(1, maxIter+1):
        save_theta_history_csv(
            stdHist, P_core=P_core,
            idx_tip=idx_tip,
            idx_base=idx_base,
            out_path="theta_history.csv"
        )

        print(f"\n=== Frank–Wolfe iter {Iter} ===")
        print(f"loss = {loss_old:.6g}")
        lossHist[Iter-1] = loss_old

        X_vec   = np.asarray(xmhe_traj,    dtype=float).reshape(-1,1)
        W_vec   = np.asarray(noise_traj,   dtype=float).reshape(-1,1)
        Lam_vec = np.asarray(costate_traj, dtype=float).reshape(-1,1)

        # rebuild measurement with current tip
        tip_now = theta[idx_tip:idx_tip+12]
        y_meas_now, dY_doffset = build_y_and_dY(
            pin_model, pin_data, fids,
            q_meas, v_meas, u_meas,
            tip_now
        )
        Y_vec = np.asarray(y_meas_now, dtype=float).reshape(-1, 1)
        U_vec = np.asarray(u_meas[:-1,:], dtype=float).reshape(-1, 1)
        C_vec = np.asarray(contact_seq, dtype=float).reshape(-1, 1)

        # --- KKT linearization (same as your code) ---
        KKT_val     = srbMHE.KKT_fn(s=X_vec, n=W_vec, costate=Lam_vec,
                                    y=Y_vec, u=U_vec, c=C_vec,
                                    prior=x_hat, tp=theta[:P_core].tolist(), G=G_vec)["KKT_fn"]
        g_val       = srbMHE.g_fn(s=X_vec, n=W_vec, costate=Lam_vec,
                                  y=Y_vec, u=U_vec, c=C_vec,
                                  prior=x_hat, tp=theta[:P_core].tolist(), G=G_vec)["g_fn"]
        dKKT_Z_val  = srbMHE.dKKT_Z_fn(s=X_vec, n=W_vec, costate=Lam_vec,
                                       y=Y_vec, u=U_vec, c=C_vec,
                                       prior=x_hat, tp=theta[:P_core].tolist(), G=G_vec)["dKKT_Z_fn"]
        dKKT_params = srbMHE.dKKT_tp_fn(s=X_vec, n=W_vec, costate=Lam_vec,
                                        y=Y_vec, u=U_vec, c=C_vec,
                                        prior=x_hat, tp=theta[:P_core].tolist(), G=G_vec)["dKKT_tp_fn"]
        dKKT_Y_val  = srbMHE.dKKT_Y_fn(s=X_vec, n=W_vec, costate=Lam_vec,
                                       y=Y_vec, u=U_vec, c=C_vec,
                                       prior=x_hat, tp=theta[:P_core].tolist(), G=G_vec)["dKKT_Y_fn"]
        dKKT_G_val = srbMHE.dKKT_G_fn(s=X_vec, n=W_vec, costate=Lam_vec,
                                      y=Y_vec, u=U_vec, c=C_vec,
                                      prior=x_hat, tp=theta[:P_core].tolist(), G=G_vec)["dKKT_G"]

        n_sys = int(KKT_val.size1())

        # sparse-ify
        Sg     = dKKT_G_val.sparsity()
        rows_g = np.asarray(Sg.row(), dtype=np.int32)
        colptr_g = np.asarray(Sg.colind(), dtype=np.int32)
        vals_g   = np.asarray(dKKT_G_val.nonzeros(), dtype=np.float64)
        FG_csc   = sp.csc_matrix((vals_g, rows_g, colptr_g), shape=(n_sys, (H+1)*24*9))

        Sz       = dKKT_Z_val.sparsity()
        rows_z   = np.asarray(Sz.row(), dtype=np.int32)
        colptr_z = np.asarray(Sz.colind(), dtype=np.int32)
        vals_z   = np.asarray(dKKT_Z_val.nonzeros(), dtype=np.float64)
        Fz_csc   = sp.csc_matrix((vals_z, rows_z, colptr_z), shape=(n_sys, n_sys))

        Sy       = dKKT_Y_val.sparsity()
        rows_y   = np.asarray(Sy.row(), dtype=np.int32)
        colptr_y = np.asarray(Sy.colind(), dtype=np.int32)
        vals_y   = np.asarray(dKKT_Y_val.nonzeros(), dtype=np.float64)
        Fy_csc   = sp.csc_matrix((vals_y, rows_y, colptr_y), shape=(n_sys, dY_doffset.shape[0]))

        Ftp      = np.asarray(dKKT_params.full(), dtype=np.float64)  # (n_sys, P_core)

        # contribution from dG/dtip (loop over k)
        cols_per_k = 24*9
        FG_tip = np.zeros((n_sys, 12))
        for k in range(H+1):
            c0, c1 = k*cols_per_k, (k+1)*cols_per_k
            FG_k = FG_csc[:, c0:c1]

            q_i = q_meas[k, :]
            v_i = v_meas[k, :]
            u_i = u_meas[k, :]
            q_zero, v_zero = build_zeroed(q_i, v_i)
            dGk = dGk_dtip_from_codegen(f_yv, f_pf, pin_model, pin_data,
                                        q_zero, v_zero, u_i[3:6], fids)
            FG_tip += FG_k @ dGk

        RHS = np.hstack([Ftp, Fy_csc @ dY_doffset + FG_tip])

        # solve sensitivity
        try:
            Xsens = spsolve(Fz_csc, RHS, matrix_type=11)
        except TypeError:
            Xsens = spsolve(Fz_csc, RHS)

        dZ_dtheta_core_tip = -Xsens

        # drop some state entries (you had this in your code)
        n_state = srbMHE.n_state
        J_top = dZ_dtheta_core_tip[:(H+1)*n_state, :]

        drop_offsets = [6,7,8,13,14,15]
        mask = np.ones((H+1)*n_state, dtype=bool)
        for k in range(H+1):
            base = k * n_state
            for off in drop_offsets:
                mask[base + off] = False
        J_lower = J_top[mask, :]

        # dL/dx part
        grad_vec = srbMHE.dL_dQ_fn(
            q=DM(xmhe_traj[:, 9:13].reshape(-1, 1)),
            qm=DM(x_data[start_idx:start_idx+H+1, 3:7].reshape(-1, 1))
        )["dL_dQ"]
        dL_dq_mat = np.array(grad_vec).reshape(H+1, 4)
        J_upper = dldx(
            xmhe_traj, x_data, foot_data, start_idx, H,
            dL_dq_mat, 1, 3, 0.5, 2,
            p_ofs=theta[idx_base:idx_base+3]
        )

        g_core_tip = (J_upper @ J_lower).reshape(-1)  # (P_core+12,)

        # base offset gradient
        g_p = np.zeros(3, dtype=float)
        xr = x_data[start_idx:start_idx+H+1, :]
        p_mhe_now = xmhe_traj[:, 0:3]
        p_mocap = xr[:, 0:3]
        q_gt = xr[:, 3:7]; q_gt = q_gt / (np.linalg.norm(q_gt, axis=1, keepdims=True))
        p_base_now = theta[idx_base:idx_base+3]
        for k in range(p_mhe_now.shape[0]):
            Rk = pin.Quaternion(q_gt[k]).toRotationMatrix()
            rk = p_mhe_now[k] + (Rk @ p_base_now) - p_mocap[k]
            g_p += 2.0 * (Rk.T @ rk)
        g_full = np.concatenate([g_core_tip, g_p], axis=0)
        gradHist[Iter-1] = np.linalg.norm(g_full)

        # ---------- LMO via cvxpy ----------
        P_tot = P_core + 12 + 3
        x_opt = cp.Variable(P_tot)

        # base box
        lb = -cfg.big_box * np.ones(P_tot)
        ub =  cfg.big_box * np.ones(P_tot)
        # core segment stricter
        lb[:P_core] = cfg.arrival_min
        ub[:P_core] = cfg.arrival_max

        # tip/base box
        S_TIP  = slice(P_core, P_core+12)
        S_BASE = slice(P_core+12, P_core+12+3)
        cons = [
            x_opt[S_TIP]  >= -cfg.tip_bound,
            x_opt[S_TIP]  <=  cfg.tip_bound,
            x_opt[S_BASE] >= -cfg.base_bound,
            x_opt[S_BASE] <=  cfg.base_bound,
            x_opt >= lb,
            x_opt <= ub,
            x_opt[:P_core] >= 1e-9,
        ]

        # process PSD like your original code (abbrev here)
        obj = cp.Minimize(g_full @ x_opt)
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.MOSEK, verbose=False)
        if x_opt.value is None:
            print("LMO failed."); break
        s = x_opt.value.astype(float).reshape(-1)

        delta = s - theta

        # ---- Armijo ----
        gamma = cfg.armijo_gamma_init
        lin_model = float(g_full @ delta)

        while True:
            theta_cand = theta + gamma * delta
            theta_cand_core = theta_cand[:P_core]
            tip_cand        = theta_cand[S_TIP]
            p_base_cand     = theta_cand[S_BASE]

            y_meas_cand, _  = build_y_and_dY(
                pin_model, pin_data, fids,
                q_meas, v_meas, u_meas,
                tip_cand
            )
            sol_cand = srbMHE.MHEsolver(
                y_meas_cand, u_meas, x_hat,
                theta_cand_core.tolist(), H,
                contact_seq, G_meas
            )
            xmhe_cand    = sol_cand["state_traj_opt"]
            noise_cand   = sol_cand["noise_traj_opt"]
            costate_cand = sol_cand["costate_ipopt"]

            q_est_cand = xmhe_cand[:, 9:13]
            L_val_cand = srbMHE.L_att_fn(
                q=DM(q_est_cand.reshape(-1, 1)),
                qm=DM(q_mocap_win.reshape(-1, 1))
            )["L"]
            loss_cand, breakdown = outer_loss(
                xmhe_cand, x_data, foot_data,
                start_idx, H, L_val_cand,
                1, 3, 0.5, 2,
                p_ofs=p_base_cand
            )

            RHS = loss_old + cfg.armijo_rho * gamma * float(g_full @ delta)
            if loss_cand <= RHS:
                break
            gamma *= cfg.armijo_beta
            if gamma < 1e-8:
                print("Armijo reached min gamma."); break

        dL_expected = gamma * lin_model
        dL_exp_hist[Iter-1] = dL_expected
        dL_act_hist[Iter-1] = float(loss_cand - loss_old)

        theta_new = theta + gamma * delta
        stdHist[Iter,:] = theta_new

        xmhe_traj  = xmhe_cand
        noise_traj = noise_cand
        costate_traj = costate_cand
        loss_old   = float(loss_cand)
        theta      = theta_new

        print(breakdown)
        print(f"   dL_exp  = {dL_expected:.6g}")
        print(f"   dL      = {loss_old - lossHist[Iter-1]:.6g}")
        print(f"   ||grad||= {gradHist[Iter-1]:.6g}")

        # plot & export this iter
        plot_results(
            xmhe_traj, x_data, foot_data, start_idx, H,
            save_prefix="cur",
            base_ofs=theta[idx_base:idx_base+3],
            tip_ofs=theta[idx_tip:idx_tip+12],
            model=pin_model, fids=fids,
            q_meas_win=q_meas,
            add_tip_offset=True
        )
        export_iter_est_csv(
            prefix=f"iter_{Iter:03d}",
            xmhe_traj=xmhe_traj,
            q_meas_win=q_meas,
            out_dir="fw_iters",
            dt=cfg.dt_mhe
        )
        plot_theta_history(stdHist, idx_tip, idx_base, Iter, save_prefix="cur")

    # final exports
    save_theta_history_csv(stdHist, P_core=P_core, idx_tip=idx_tip, idx_base=idx_base,
                           out_path="theta_history.csv")
    export_fw_snapshot_csv(
        prefix="end",
        xmhe_traj=xmhe_traj,
        x_data=x_data, foot_data=foot_data,
        start_idx=start_idx, horizon=H,
        base_ofs=theta[idx_base:idx_base+3],
        tip_ofs=theta[idx_tip:idx_tip+12],
        model=pin_model, fids=fids,
        q_meas_win=q_meas, add_tip_offset=True,
        out_dir=".",
        base_ofs_calib=theta[-3:],
        tip_ofs_calib=theta[-15:-3]
    )

