import casadi as cs
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import os
import json
from casadi import Function

URDF_PATH = "B1.urdf"
FOOT_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]  # your list

def skew_cs(v):
    v0, v1, v2 = v[0], v[1], v[2]
    z = cs.SX(0)  # 1x1 zero SX
    return cs.vertcat(
        cs.hcat([z,  -v2,  v1]),
        cs.hcat([v2,   z, -v0]),
        cs.hcat([-v1, v0,   z])
    )

def compute_pf_meas(model, data, q_zero, fids, offset_calf):
    # ------------------------------------------------------
    ## offset_calf: the offset of foot position in the calf frame
    # ------------------------------------------------------
    pin.forwardKinematics(model, data, q_zero)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q_zero)

    pf_i = np.zeros(12)
    for j, fid in enumerate(fids):
        pf_i[3*j:3*j+3] = data.oMf[fid].translation
        
        # Offset foot in the calf frame
        # ------------------------------------------------------
        jid = model.frames[fid].parentJoint # find the parent non fixed joint of the foot end
        Rj_i = data.oMi[jid].rotation
        pj_i = data.oMi[jid].translation 
        pf_i_offset = Rj_i @ offset_calf[3*j:3*j+3]
        pf_i[3*j:3*j+3] += pf_i_offset
        
        # Updated Frame Jacobian for offseted foot postion
        # ------------------------------------------------------
        Jj_i = pin.getJointJacobian(model, data, jid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        Jf_i = pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        skew_offset_i = pin.skew(Rj_i @ offset_calf[3*j:3*j+3])
        Jy_i = Jf_i[0:3,:] - skew_offset_i @ Jj_i[3:6, :] # Use this as measurement jacoabin with resp to joint position/ encoder
        
        # Gradient of Measurement over offset (tunable parameter) 3 by 3 for each foot. 12 by 3 intotal
        # ------------------------------------------------------
        dy_dtheta = Rj_i
        
        # Sanity Check
        # ------------------------------------------------------
        # J_num = np.zeros((3, model.nv))
        # dq = 1e-8
        # for i in range(model.nv):
        #     q_plus = pin.integrate(model, q_zero, np.eye(model.nv)[i] * dq)
        #     pin.forwardKinematics(model, data, q_plus)
        #     pin.updateFramePlacements(model, data)
            
        #     Rj_p = data.oMi[jid].rotation
        #     pf_p = data.oMf[fid].translation + Rj_p @ offset_calf[3*j:3*j+3]
        #     J_num[:, i] = (pf_p - pf_i[3*j:3*j+3]) / (dq)
        #     print("Analytical Jacobian:\n", Jy_i)
        #     print("\n Numerical Jacobian:\n", J_num)
    return pf_i

def compute_yv_kin(model, data, q_zero, v_zero, omega, fids, offset_calf):
    pin.forwardKinematics(model, data, q_zero, v_zero)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q_zero)
    v_foot_i = np.zeros(12, dtype=float)
    for j, fid in enumerate(fids):
        J_lwa = pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_lin = np.asarray(J_lwa[0:3, 6:18])

        v_foot_i[3*j:3*j+3] = -((J_lin @ v_zero[6:18]) + np.cross(omega, np.asarray(data.oMf[fid].translation)).reshape(3,))
    
        # Offset velocity
        # ------------------------------------------------------
        jid = model.frames[fid].parentJoint # find the parent non fixed joint of the foot end
        Rj_i = data.oMi[jid].rotation
        pj_i = data.oMi[jid].translation 
        pf_i_offset = Rj_i @ offset_calf[3*j:3*j+3]
        Jj_i = pin.getJointJacobian(model, data, jid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        skew_offset_i = pin.skew(pf_i_offset)
        Jy_i = J_lwa[0:3,:] -skew_offset_i @ Jj_i[3:6, :] # Jacobian same as in pf meas
        v_foot_offset_i = (-skew_offset_i @ Jj_i[3:6, 6:18] @ v_zero[6:] + np.cross(omega, pf_i_offset))
        
        # Gradient of Measurement over offset (tunable parameter) 3 by 3 for each foot. 12 by 3 intotal
        # ------------------------------------------------------
        dy_dtheta = pin.skew(Jj_i[3:6, :] @ v_zero) @ Rj_i + pin.skew(omega) @ Rj_i

        # # Sanity Check
        # # ------------------------------------------------------
        # epsilon = 1e-6
        # J_num = np.zeros((3, 3))
        # for i in range(3):
        #     dr = np.zeros(3)
        #     dr[i] = epsilon

        #     pf_offset_plus = Rj_i @ (offset_calf[3*j:3*j+3] + dr)
        #     omega_joint = Jj_i[3:6, :] @ v_zero
        #     v_plus = (-pin.skew(pf_offset_plus) @ omega_joint + np.cross(omega, pf_offset_plus))

        #     J_num[:, i] = (v_plus - v_foot_offset_i) / epsilon

        # print("Analytical:\n", dy_dtheta)
        # print("Numerical:\n", J_num)

        v_foot_i[3*j:3*j+3] += - v_foot_offset_i # Use this as measurement jacoabin with resp to joint position/ encoder

        # Sanity Check
        # ------------------------------------------------------
        # v_zero_omega = v_zero.copy()
        # v_zero_omega[3:6] = omega
        # v_foot_i_check = (J_lwa[0:3,:] -skew_offset_i @ Jj_i[3:6, :]) @ v_zero_omega
        # print("Analytical Jacobian:\n", -v_foot_i[3*j:3*j+3])
        # print("\n Numerical Jacobian:\n", v_foot_i_check)      
    return v_foot_i

def build_pf_and_J_codegen(model: pin.Model, fids: list[int], shared_offset: bool = False):
    """
    Build CasADi functions that return:
      y(q, theta): stacked foot positions with per-foot calf-frame offset
      Jy(q, theta): dy/dq
      DyDtheta(q, theta): dy/dtheta  (block-diagonal R_j if per-foot offsets)
    If shared_offset=True, theta is 3x1 shared across all feet => DyDtheta is (3m x 3).
    """
    try:
        cmodel = cpin.Model(model)
    except Exception:
        raise RuntimeError("Could not construct CasADi model directly. "
                           "Consider rebuilding with cpin.buildModelFromUrdf(urdf_path).")
    cdata = cmodel.createData()

    nq, nv = cmodel.nq, cmodel.nv
    m = len(fids)

    q_joint = cs.SX.sym("q", nq-7)  # Body Frame value without base
    q_zero = cs.vertcat(cs.DM.zeros(6),cs.DM(1), q_joint)   # base replaced with zero

    if shared_offset:
        theta = cs.SX.sym("theta", 3)         # one 3x1 offset shared by all feet
    else:
        theta = cs.SX.sym("theta", 3*m)       # per-foot offsets concatenated [θ1; θ2; ...; θm]

    # Forward kinematics + frame placements
    cpin.forwardKinematics(cmodel, cdata, q_zero)
    cpin.updateFramePlacements(cmodel, cdata)

    # Compose measurement y = [p_fid + R_parentJoint * theta_j]_j
    y_blocks = []
    R_blocks = []  # useful if you want explicit Dy/Dtheta without AD
    for j, fid in enumerate(fids):
        # Parent non-fixed joint of the frame (same as your Python code)
        jid = cmodel.frames[fid].parentJoint
        Rj = cdata.oMi[jid].rotation      # 3x3
        pf = cdata.oMf[fid].translation   # 3x1 (world)

        if shared_offset:
            theta_j = theta
        else:
            theta_j = theta[3*j:3*j+3]

        yj = pf + Rj @ theta_j
        y_blocks.append(yj)
        R_blocks.append(Rj)

    y = cs.vertcat(*y_blocks)          # (3m x 1)

    # Jacobians via CasADi AD (robust and simple)
    Jy = cs.jacobian(y, q_joint)             # (3m x nq)
    DyDtheta = cs.jacobian(y, theta)   # (3m x (3 or 3m))

    Jy_flat = cs.vec(Jy)
    dJy_dtheta = cs.jacobian(Jy_flat, theta)   # (3m*nq) x p

    # Pack functions
    outs = [y, Jy, DyDtheta, dJy_dtheta]
    outnames = ["y", "Jy", "DyDtheta", "dJy_dtheta"]

    f_pf = cs.Function("pf_and_J", [q_joint, theta], outs, ["q", "theta"], outnames)

    # --- Codegen ---
    cg = cs.CodeGenerator("pf_and_J_codegen.c")
    cg.add(f_pf)
    cg.generate()
    # This creates pf_and_J_codegen.c in the current folder.
    # Compile example (Linux):
    #   gcc -fPIC -shared pf_and_J_codegen.c -o libpf_and_J.so
    # Then you can load it back with casadi.Function.load()

    return f_pf

def build_yv_and_J_codegen(model: pin.Model, fids, shared_offset: bool=False,
                           base_ang_slice=(3, 6), c_filename="yv_and_J_codegen.c"):
    """
    Create CasADi function yv_and_J(q, v, omega, theta) and codegen C file.

    Args:
      model: pin.Model (same URDF + FreeFlyer used at runtime)
      fids:  list of frame IDs for feet in order [FR, FL, RR, RL]
      shared_offset: if True, theta is 3x1 shared; else per-foot 12x1
      base_ang_slice: tuple (lo,hi) for where base angular vel sits in v
                      (defaults to (3,6) to mirror your code)
      c_filename: output C filename

    Returns:
      casadi.Function with outputs:
        y (12x1), Jy_q (12xnq), Jy_v (12xnv), Jy_omega (12x3), DyDtheta (12xp)
    """
    cmodel = cpin.Model(model)
    cdata  = cmodel.createData()

    nq, nv = cmodel.nq, cmodel.nv
    m = len(fids)
    assert m == 4, "This script assumes 4 feet (12-dim output)."

    # Symbols
    q_joint      = cs.SX.sym("q", nq-7)
    v_joint      = cs.SX.sym("v", nv-6)
    omega  = cs.SX.sym("omega", 3)
    theta  = cs.SX.sym("theta", 3 if shared_offset else 3*m)

    q_zero = cs.vertcat(cs.DM.zeros(6),cs.DM(1), q_joint)   # base replaced with zero
    v_zero = cs.vertcat(cs.DM.zeros(6), v_joint)   # base replaced with zero
    
    # Kinematics + Jacobians
    cpin.forwardKinematics(cmodel, cdata, q_zero, v_zero)  # provide v to define time-derivs for AD
    cpin.updateFramePlacements(cmodel, cdata)
    cpin.computeJointJacobians(cmodel, cdata, q_zero)

    # y_i = - (Jf_pos - skew(Rj * theta_j) @ Jj_ang) @ v_aug
    v = []
    for j, fid in enumerate(fids):
        jid = cmodel.frames[fid].parentJoint

        # Rot/pos of parent joint, and frame jacobians
        Rj = cdata.oMi[jid].rotation              # 3x3
        Jf = cpin.getFrameJacobian(cmodel, cdata, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)  # 6xnv
        Jj = cpin.getJointJacobian(cmodel, cdata, jid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)  # 6xnv
        pf = cdata.oMf[fid].translation
        J_lin = Jf[0:3, 6:18]
        Jj_lin = Jj[3:6, 6:18]

        theta_j = theta if shared_offset else theta[3*j:3*j+3]
        off_world = Rj @ theta_j
        skew_off = skew_cs(off_world)
        
        v_base = -((J_lin @ v_joint) + cs.cross(omega, pf))
        
        v_foot_offset_i = (-skew_off @ Jj_lin @ v_joint + cs.cross(omega, Rj @ theta_j))
        
        v_i= v_base - v_foot_offset_i
        
        v.append(v_i)
    v_foot = cs.vertcat(*v)   

    # Jacobians (CasADi AD)
    Jy_q     = cs.jacobian(v_foot, q_joint)                 # 12 x nq
    Jy_v     = cs.jacobian(v_foot, v_joint)                 # 12 x nv   (note: v_aug depends on v outside base_ang_slice)
    Jy_omega = cs.jacobian(v_foot, omega)             # 12 x 3
    DyDtheta = cs.jacobian(v_foot, theta)             # 12 x (12 or 3)

    Jy_q_flat = cs.vec(Jy_q)
    dJy_q_dtheta = cs.jacobian(Jy_q_flat, theta)
    Jy_v_flat = cs.vec(Jy_v)
    dJy_v_dtheta = cs.jacobian(Jy_v_flat, theta)
    Jy_omega_flat = cs.vec(Jy_omega)
    dJy_omega_dtheta = cs.jacobian(Jy_omega_flat, theta)
    # Pack function
    f_vf = cs.Function(
        "yv_and_J",
        [q_joint, v_joint, omega, theta],
        [v_foot,Jy_q,Jy_v,Jy_omega,DyDtheta,dJy_q_dtheta,dJy_v_dtheta,dJy_omega_dtheta],
        ["q", "v", "omega", "theta"],
        ["v_foot", "Jy_q" , "Jy_v", "Jy_omega", "DyDtheta", "dJy_q_dtheta", "dJy_v_dtheta", "dJy_omega_dtheta"],
    )

    # Codegen
    cg = cs.CodeGenerator(c_filename)
    cg.add(f_vf)
    cg.generate()
    print(f"[ok] Wrote {c_filename}")
    return f_vf


if __name__ == "__main__":
    import numpy as np
    import casadi as cs
    import pinocchio as pin

    # --------- helpers: column-major (Fortran) unflatten ----------
    def unvec_colmajor(vec, rows, cols):
        return np.reshape(vec, (rows, cols), order="F")

    def unflatten_jac_colmajor(J_flat, rows, cols):
        """
        J_flat: (rows*cols, p) = d vec(M) / d theta
        returns T with T[:,:,k] = dM/dtheta_k, shape (rows, cols, p)
        """
        J_flat = np.asarray(J_flat)
        p = J_flat.shape[1]
        mats = [unvec_colmajor(J_flat[:, k], rows, cols) for k in range(p)]
        return np.stack(mats, axis=2)

    # --------- build model & feet ----------
    URDF_PATH = "B1.urdf"
    FOOT_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    data  = model.createData()

    fids = []
    for nm in FOOT_NAMES:
        fid = model.getFrameId(nm)
        if fid == len(model.frames):
            raise RuntimeError(f"Frame '{nm}' not found in model.")
        fids.append(fid)
    print("[ok] Foot frame IDs:", fids)

    # # --------- build functions (and codegen) ----------
    # f_pf = build_pf_and_J_codegen(model, fids, shared_offset=False)
    # f_vf = build_yv_and_J_codegen(model, fids, shared_offset=False)

    # # --------- numeric test inputs ----------
    # q0 = pin.neutral(model)          # (nq,)
    # vj = np.zeros(model.nv - 6)      # joint velocities only
    # vj[:3] = [0.2, -0.1, 0.05]
    # omega = np.array([0.1, -0.2, 0.3])

    # theta = np.zeros(12)             # per-foot offsets
    # for j in range(4):
    #     theta[3*j:3*j+3] = [0.05, 0.02, -0.03]

    # # --------- evaluate (KWARGS -> DICT) ----------
    # out_pf = f_pf(q=q0[7:], theta=theta)
    # # Dict keys are your outnames: "y", "Jy", "DyDtheta", "dJy_dtheta"
    # y           = np.array(out_pf["y"].full()).reshape(-1)
    # Jy          = np.array(out_pf["Jy"].full())
    # DyDtheta    = np.array(out_pf["DyDtheta"].full())
    # dJy_dtheta  = np.array(out_pf["dJy_dtheta"].full())

    # out_vf = f_vf(q=q0[7:], v=vj, omega=omega, theta=theta)
    # # Keys: "v_foot", "Jy_q", "Jy_v", "Jy_omega", "DyDtheta",
    # #       "dJy_q_dtheta", "dJy_v_dtheta", "dJy_omega_dtheta"
    # v_foot           = np.array(out_vf["v_foot"].full()).reshape(-1)
    # Jy_q             = np.array(out_vf["Jy_q"].full())
    # Jy_v             = np.array(out_vf["Jy_v"].full())
    # Jy_omega         = np.array(out_vf["Jy_omega"].full())
    # DyDtheta_v       = np.array(out_vf["DyDtheta"].full())
    # dJy_q_dtheta     = np.array(out_vf["dJy_q_dtheta"].full())      # (12*(nq-7), p)
    # dJy_v_dtheta     = np.array(out_vf["dJy_v_dtheta"].full())      # (12*(nv-6), p)
    # dJy_omega_dtheta = np.array(out_vf["dJy_omega_dtheta"].full())  # (12*3,      p)

    # # --------- infer dims ----------
    # rows   = 12
    # cols_q = Jy_q.shape[1]      # = nq-7
    # cols_v = Jy_v.shape[1]      # = nv-6
    # cols_o = Jy_omega.shape[1]  # = 3
    # p      = DyDtheta.shape[1]  # = len(theta) (12 here)

    # # --------- unflatten back to tensors ----------
    # dJy_q_tensor     = unflatten_jac_colmajor(dJy_q_dtheta,     rows, cols_q)  # (12, nq-7, p)
    # dJy_v_tensor     = unflatten_jac_colmajor(dJy_v_dtheta,     rows, cols_v)  # (12, nv-6, p)
    # dJy_omega_tensor = unflatten_jac_colmajor(dJy_omega_dtheta, rows, cols_o)  # (12, 3,    p)

    # # --------- prints ----------
    # np.set_printoptions(precision=4, suppress=True)

    # print("\n=== Base quantities ===")
    # print("y shape:", y.shape)
    # print("Jy shape:", Jy.shape)
    # print("DyDtheta shape:", DyDtheta.shape)
    # print("v_foot shape:", v_foot.shape)
    # print("Jy_q shape:", Jy_q.shape)
    # print("Jy_v shape:", Jy_v.shape)
    # print("Jy_omega shape:", Jy_omega.shape)

    # print("\n=== Flattened Jacobians (CasADi) ===")
    # print("dJy_q_dtheta (flat) shape:", dJy_q_dtheta.shape)
    # print("dJy_v_dtheta (flat) shape:", dJy_v_dtheta.shape)
    # print("dJy_omega_dtheta (flat) shape:", dJy_omega_dtheta.shape)

    # print("\n=== Unflattened tensors (rows, cols, p) ===")
    # print("dJy_q_tensor shape:", dJy_q_tensor.shape)
    # print("dJy_v_tensor shape:", dJy_v_tensor.shape)
    # print("dJy_omega_tensor shape:", dJy_omega_tensor.shape)

    # k = 0
    # print("\nSlice dJy_q_tensor[0:3, 0:3, k=0]:\n", dJy_q_tensor[0:3, 0:3, k])
    # print("Slice dJy_v_tensor[0:3, 0:3, k=0]:\n", dJy_v_tensor[0:3, 0:3, k])
    # print("Slice dJy_omega_tensor[0:3, 0:3, k=0]:\n", dJy_omega_tensor[0:3, 0:3, k])

    # # Quick FD check for one theta component:
    # eps = 1e-7
    # theta_pert = theta.copy(); theta_pert[k] += eps
    # out_vf_plus = f_vf(q=q0[7:], v=vj, omega=omega, theta=theta_pert)
    # Jy_q_plus = np.array(out_vf_plus["Jy_q"].full())
    # FD = (Jy_q_plus - Jy_q) / eps
    # print("\n[FD check] ||FD - dJy_q_tensor[:,:,k]||_F:",
    #       np.linalg.norm(FD - dJy_q_tensor[:, :, k]))

    #     # ================== Finite-Difference checks for y ==================
    # # We check: Jy ≈ (y(q+ε e_i) - y(q))/ε  and  DyDtheta ≈ (y(θ+ε e_k) - y(θ))/ε
    # eps_q = 1e-7
    # eps_t = 1e-7

    # q_joint_base = np.array(q0[7:], dtype=float)
    # theta_base   = np.array(theta, dtype=float)

    # rows   = y.size                 # = 12
    # cols_q = Jy.shape[1]            # = nq-7
    # p      = DyDtheta.shape[1]      # = len(theta)

    # # ---------- FD for Jy ----------
    # FD_Jy = np.zeros_like(Jy)
    # for i in range(cols_q):
    #     q_pert = q_joint_base.copy()
    #     q_pert[i] += eps_q
    #     y_plus = np.array(f_pf(q=q_pert, theta=theta_base)["y"].full()).reshape(-1)
    #     FD_Jy[:, i] = (y_plus - y) / eps_q

    # err_Jy = FD_Jy - Jy
    # fro_Jy = np.linalg.norm(err_Jy)
    # rel_Jy = fro_Jy / max(1.0, np.linalg.norm(Jy))
    # print("\n[FD] Jy check:")
    # print("  ||FD - Jy||_F =", fro_Jy, "   rel =", rel_Jy)
    # print("  max |FD - Jy| =", np.max(np.abs(err_Jy)))
    # print("  FD_Jy[0:3,0:3]:\n", FD_Jy[0:3, 0:3])
    # print("   Ana[0:3,0:3]:\n", Jy[0:3, 0:3])

    # # ---------- FD for DyDtheta ----------
    # FD_DyDtheta = np.zeros_like(DyDtheta)
    # for k in range(p):
    #     th_pert = theta_base.copy()
    #     th_pert[k] += eps_t
    #     y_plus = np.array(f_pf(q=q_joint_base, theta=th_pert)["y"].full()).reshape(-1)
    #     FD_DyDtheta[:, k] = (y_plus - y) / eps_t

    # err_D = FD_DyDtheta - DyDtheta
    # fro_D = np.linalg.norm(err_D)
    # rel_D = fro_D / max(1.0, np.linalg.norm(DyDtheta))
    # print("\n[FD] DyDtheta check:")
    # print("  ||FD - DyDtheta||_F =", fro_D, "   rel =", rel_D)
    # print("  max |FD - DyDtheta| =", np.max(np.abs(err_D)))
    # print("  FD_DyDtheta[0:3,0:3]:\n", FD_DyDtheta[0:3, 0:3])
    # print("   Ana[0:3,0:3]:\n", DyDtheta[0:3, 0:3])
