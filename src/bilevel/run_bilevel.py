# src/bilevel/run_bilevel.py

import pinocchio as pin

from .config import BilevelConfig
from .data_io import load_dataset
from .codegen import load_codegen_functions
from .fw_solver import run_frank_wolfe
from .dynamics.srb_dynamics import SrbDynamics
from .estimator.srb_mhe import MHE

def main():
    cfg = BilevelConfig()
    data = load_dataset(cfg)

    # build pinocchio model
    model = pin.buildModelFromUrdf(cfg.urdf_path, pin.JointModelFreeFlyer())
    data_pin = model.createData()
    foot_names = ["FR_foot","FL_foot","RR_foot","RL_foot"]
    fids = [model.getFrameId(nm) for nm in foot_names]

    # srb dynamics
    srb = SrbDynamics(cfg.dt_mhe)
    srb.model()

    # MHE
    mhe = MHE(cfg.horizon, cfg.dt_mhe)
    mhe.SetStateVariable(srb.xa)
    mhe.SetOutputVariable(srb.y)
    mhe.SetControlVariable(srb.u)
    mhe.SetNoiseVariable(srb.w)
    mhe.SetModelDyn(srb.dymh)
    mhe.SetCostDyn()

    # external casadi functions
    f_yv, f_pf = load_codegen_functions(cfg)

    # run outer loop
    run_frank_wolfe(
        cfg=cfg,
        srbMHE=mhe,
        srbDyn=srb,
        pin_model=model,
        pin_data=data_pin,
        fids=fids,
        dataset=data,
        f_yv=f_yv,
        f_pf=f_pf
    )

if __name__ == "__main__":
    main()

