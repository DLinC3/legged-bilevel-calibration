# src/bilevel/codegen.py

import os
from casadi import external
from .config import BilevelConfig

def _pick_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of these codegen libs exist: {paths}")

def load_codegen_functions(cfg: BilevelConfig):
    """
    Load external CasADi functions for
      - foot velocity and jacobian wrt tip
      - foot position and jacobian wrt tip
    """
    lib_v_candidates = [
        os.path.join(cfg.external_lib_dir, "libyv_and_J_codegen.so"),
        os.path.join(cfg.external_lib_dir, "libyv_and_J_codegen.dylib"),
        os.path.join(cfg.external_lib_dir, "libyv_and_J_codegen.dll"),
    ]
    lib_p_candidates = [
        os.path.join(cfg.external_lib_dir, "libpf_and_J_codegen.so"),
        os.path.join(cfg.external_lib_dir, "libpf_and_J_codegen.dylib"),
        os.path.join(cfg.external_lib_dir, "libpf_and_J_codegen.dll"),
    ]
    lib_v = _pick_first_existing(lib_v_candidates)
    lib_p = _pick_first_existing(lib_p_candidates)

    f_yv = external("yv_and_J", lib_v)
    f_pf = external("pf_and_J", lib_p)
    return f_yv, f_pf

