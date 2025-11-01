from casadi import *
import numpy as np
from ..dynamics.srb_dynamics import SrbDynamics
import os
import json
from casadi import Function
def covariance(v):
    """put v=[dxx,dyy,dzz,dxy,dxz,dyz] into 3x3 symmetric matrix"""
    return vertcat(
        horzcat(v[0], v[3], v[4]),
        horzcat(v[3], v[1], v[5]),
        horzcat(v[4], v[5], v[2]))

class MHE:
    
    def __init__(self, horizon, dt_sample):
        self.N = horizon
        self.DT = dt_sample

    def SetStateVariable(self, xa):
        self.state = xa
        self.n_state = xa.numel()

    def SetOutputVariable(self, y):
        assert hasattr(self, 'state'), "Define the state variable first!"
        self.output = y
        self.y_fn   = Function('y',[self.state], [self.output], ['x0'], ['yf'])
        self.n_output = self.output.numel()

    def SetControlVariable(self, u):
        self.ctrl = u
        self.n_ctrl = u.numel()

    def SetNoiseVariable(self, eta):
        self.noise = eta
        self.n_noise = eta.numel()

    def SetModelDyn(self, dymh):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'ctrl'), "Define the control variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        # discrete-time dynamic model based on 4th-order Runge-Kutta method
        self.ModelDyn = self.state + self.DT*dymh
        self.MDyn_fn  = Function('MDyn', [self.state, self.ctrl, self.noise], [self.ModelDyn],
                                 ['s', 'c', 'n'], ['MDynf'])

    def SetArrivalCost(self, x_hat):
        assert hasattr(self, 'state'), "Define the state variable first!"
        self.P0        = diag(self.weight_para[0, 0:self.n_state])
        # Define filter priori
        error_a        = self.state - x_hat
        self.cost_a    = 1/2 * mtimes(mtimes(transpose(error_a), self.P0), error_a)
        self.cost_a_fn = Function('cost_a', [self.state, self.weight_para], [self.cost_a], ['s','tp'], ['cost_af'])

    def SetCostDyn(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'output'), "Define the output variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"

        # ---------- dimensions ----------
        MEAS_LEN  = 12          # R_q(6) + R_qdot(6)
        NOISE_LEN = 6 + 6 + 3 + 3 + 3 + 3   

        # weights parameter vector
        self.weight_para = SX.sym('t_para', 1, self.n_state + MEAS_LEN + NOISE_LEN)

        self.horizon1 = SX.sym('h1')
        self.index    = SX.sym('ki')

        # measurements and contact
        # y = [yv_foot^B(12); pf_mea(12)]
        self.measurement = SX.sym('y', 24, 1)
        self.contact     = SX.sym('c', 4, 1)

        # G for every step(24x9)= 8x3x9 vertically stack
        self.Gmeas       = SX.sym('G', 24, 9)

        # ---------- unpack measurement weights ----------
        idx = self.n_state
        R_q    = covariance(self.weight_para[0, idx:idx+6]); idx += 6   # 3x3
        R_qdot = covariance(self.weight_para[0, idx:idx+6]); idx += 6   # 3x3

        # outputs from dynamics (Body frame)
        y_state = self.output
        vB      = y_state[0:3]
        pfB_FR  = y_state[3:6]
        pfB_FL  = y_state[6:9]
        pfB_RR  = y_state[9:12]
        pfB_RL  = y_state[12:15]

        # measurement vectors
        yv_foot = self.measurement[0:12]    # (12x1)
        pf_mea  = self.measurement[12:24]   # (12x1)

        # residuals per leg
        r_v_FR = vB - yv_foot[0:3]
        r_v_FL = vB - yv_foot[3:6]
        r_v_RR = vB - yv_foot[6:9]
        r_v_RL = vB - yv_foot[9:12]

        r_pf_FR = pfB_FR - pf_mea[0:3]
        r_pf_FL = pfB_FL - pf_mea[3:6]
        r_pf_RR = pfB_RR - pf_mea[6:9]
        r_pf_RL = pfB_RL - pf_mea[9:12]

        # ---------- process weights (for R9's omega block and Q) ----------
        idx = self.n_state + MEAS_LEN
        Qa_blk = covariance(self.weight_para[0, idx:idx+6]); idx += 6
        Qw_blk = covariance(self.weight_para[0, idx:idx+6]); idx += 6   # 3x3 as R9 omega block
        Qa_walk = self.weight_para[0, idx:idx+3]; idx += 3
        Qw_walk = self.weight_para[0, idx:idx+3]; idx += 3
        Qswing  = self.weight_para[0, idx:idx+3]; idx += 3
        Qstance = self.weight_para[0, idx:idx+3]; idx += 3

        # R9 = diag(R_q, R_qdot, Qw_blk)  (9x9)
        R9 = vertcat(
                horzcat(R_q,              SX.zeros(3,3), SX.zeros(3,3)),
                horzcat(SX.zeros(3,3),    R_qdot,       SX.zeros(3,3)),
                horzcat(SX.zeros(3,3),    SX.zeros(3,3), Qw_blk)
            )
        C9   = inv(R9)
        # contact gates
        cFR, cFL, cRR, cRL = self.contact[0], self.contact[1], self.contact[2], self.contact[3]
        
        # ---------- split 8 blocks from G (24x9) ----------
        Gv_FR = self.Gmeas[0:3,   :]
        Gv_FL = self.Gmeas[3:6,   :]
        Gv_RR = self.Gmeas[6:9,   :]
        Gv_RL = self.Gmeas[9:12,  :]
        Gp_FR = self.Gmeas[12:15, :]
        Gp_FL = self.Gmeas[15:18, :]
        Gp_RR = self.Gmeas[18:21, :]
        Gp_RL = self.Gmeas[21:24, :]

        # velovity block
        S_v_FR = mtimes(mtimes(Gv_FR, C9), Gv_FR.T)
        S_v_FL = mtimes(mtimes(Gv_FL, C9), Gv_FL.T)
        S_v_RR = mtimes(mtimes(Gv_RR, C9), Gv_RR.T)
        S_v_RL = mtimes(mtimes(Gv_RL, C9), Gv_RL.T)

        # position block（Gp only get [J_leg,0,0]）
        S_p_FR = mtimes(mtimes(Gp_FR, C9), Gp_FR.T)
        S_p_FL = mtimes(mtimes(Gp_FL, C9), Gp_FL.T)
        S_p_RR = mtimes(mtimes(Gp_RR, C9), Gp_RR.T)
        S_p_RL = mtimes(mtimes(Gp_RL, C9), Gp_RL.T)

        # use solve as inv
        J_pf = 0.5*mtimes(r_pf_FR.T, solve(S_p_FR, r_pf_FR)) \
            + 0.5*mtimes(r_pf_FL.T, solve(S_p_FL, r_pf_FL)) \
            + 0.5*mtimes(r_pf_RR.T, solve(S_p_RR, r_pf_RR)) \
            + 0.5*mtimes(r_pf_RL.T, solve(S_p_RL, r_pf_RL))

        J_v  = 0.5*cFR*mtimes(r_v_FR.T, solve(S_v_FR, r_v_FR)) \
            + 0.5*cFL*mtimes(r_v_FL.T, solve(S_v_FL, r_v_FL)) \
            + 0.5*cRR*mtimes(r_v_RR.T, solve(S_v_RR, r_v_RR)) \
            + 0.5*cRL*mtimes(r_v_RL.T, solve(S_v_RL, r_v_RL))

        # ---------- process noise Q ----------
        Q_pf_FR = if_else(cFR >= 0.5, Qstance.T, Qswing.T)
        Q_pf_FL = if_else(cFL >= 0.5, Qstance.T, Qswing.T)
        Q_pf_RR = if_else(cRR >= 0.5, Qstance.T, Qswing.T)
        Q_pf_RL = if_else(cRL >= 0.5, Qstance.T, Qswing.T)

        Q_tail_diag = diag(vertcat(Qa_walk.T, Qw_walk.T, Q_pf_FR, Q_pf_FL, Q_pf_RR, Q_pf_RL))  # 18×18
        Z3_18 = SX.zeros(3,18); Z18_6 = SX.zeros(18,6)

        Q = vertcat(
            horzcat(Qa_blk, SX.zeros(3,3), Z3_18),
            horzcat(SX.zeros(3,3), Qw_blk, Z3_18),
            horzcat(Z18_6, Q_tail_diag)
        )

        self.dJ_running = J_pf + J_v + 0.5*mtimes(mtimes(self.noise.T, Q), self.noise)
        self.dJ_fn = Function('dJ_running',
                            [self.state, self.measurement, self.Gmeas, self.contact, self.noise, self.weight_para, self.horizon1, self.index],
                            [self.dJ_running],
                            ['s','m','G','c','n','tp','h1','ind'], ['dJrunf'])

        self.dJ_T = J_pf + J_v
        self.dJ_T_fn = Function('dJ_T',
                                [self.state, self.measurement, self.Gmeas, self.contact, self.weight_para, self.horizon1, self.index],
                                [self.dJ_T],
                                ['s','m','G','c','tp','h1','ind'], ['dJ_Tf'])

    def MHEsolver(self, Y, ctrl, x_hat, weight_para, time, contact_seq,G_meas):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        assert hasattr(self, 'MDyn_fn'), "Define the model dynamics function first!"
        assert hasattr(self, 'dJ_fn'), "Define the cost dynamics function first!"
        # arrival cost setting
        """
        Formulate MHE as a nonlinear programming problem solved by CasADi nlpsol() function
        """
        self.SetArrivalCost(x_hat) # x_hat: MHE estimate at t-N, obtained by the previous MHE
        
        # Start with an empty NLP
        w   = [] # optimal trajectory list
        w0  = [] # initial guess of optimal trajectory
        lbw = [] # lower boundary of optimal variables
        ubw = [] # upper boundary of optimal variables
        g   = [] # equality or inequality constraints
        lbg = [] # lower boundary of constraints
        ubg = [] # upper boundary of constraints

        # Initial state for the arrival cost
        Xk  = SX.sym('X0', self.n_state, 1)
        w  += [Xk]
        X_hatmh = []
        for i in range(len(x_hat)): # convert an array to a list
            X_hatmh += [x_hat[i,0]]
        w0 += X_hatmh
        lbw+= self.n_state*[-1e20] # value less than or equal to -1e19 stands for no lower bound
        ubw+= self.n_state*[1e20] # value greater than or equal to 1e19 stands for no upper bound
        
        # Initial constraints for FIE
        x0_val = DM(x_hat).reshape((self.n_state, 1))
        # g += [Xk - x0_val]
        # lbg  += self.n_state*[0]
        # ubg  += self.n_state*[0]
        # lbw+= x0_val 
        # ubw+= x0_val 
        
        # Formulate the NLP
        if time < self.N:
            # Full-information estimator
            self.horizon = time + 1
        else:
            # Moving horizon estimation
            self.horizon = self.N + 1 # note that we start from t-N, so there are N+1 data points

        J = self.cost_a_fn(s=Xk, tp=weight_para)['cost_af']
        # J  = 0
        
        for k in range(self.horizon-1):
            # Process noise as NLP variables
            Nk   = SX.sym('N_' + str(k), self.n_noise, 1)
            w   += [Nk]
            lbw += self.n_noise*[-1e20]
            ubw += self.n_noise*[1e20]
            W_guess = []
            W_guess += self.n_noise*[0]

            # if self.horizon <=3:
            #     W_guess += self.n_noise*[0]
            # else:
            #     if k<self.horizon-3:
            #         for iw in range(self.n_noise):
            #             W_guess += [noise_traj[k+1,iw]]
            #     else:
            #         for iw in range(self.n_noise):
            #             W_guess += [noise_traj[-1,iw]]
            
            w0  += W_guess 
            
            Gk = DM(G_meas[len(Y)-self.horizon+k, :, :])   # (24,9)
            J  += self.dJ_fn(s=Xk,
                             m=DM(Y[len(Y)-self.horizon+k, :]).reshape((24, 1)),
                             c=DM(contact_seq[len(contact_seq)-self.horizon+k, :]).reshape((4, 1)),
                             n=Nk, tp=weight_para, G=Gk, h1=self.horizon-1, ind=k)['dJrunf']
            Xnext = self.MDyn_fn(s=Xk, c=ctrl[len(ctrl)-self.horizon+k], n=Nk)['MDynf']
           
            # State as NLP variables
            Xk    = SX.sym('X_' + str(k + 1), self.n_state, 1)
            w    += [Xk]
            lbw  += self.n_state*[-1e20]
            ubw  += self.n_state*[1e20]
            X_guess = []
            X_guess = self.n_state*[0] # initialize the guess with zeros
            
            # if k<self.horizon-3:
            #     for ix in range(self.n_state):
            #         X_guess += [xmhe_traj[k+2, ix]]
            # else:
            #     for ix in range(self.n_state):
            #         X_guess += [xmhe_traj[-1, ix]]
            
            w0 += X_guess
            
            # Add equality constraint
            g    += [Xk - Xnext] # pay attention to this order! The order should be the same as that defined in the paper!
            lbg  += self.n_state*[0]
            ubg  += self.n_state*[0]

        # Add the final cost
        GH = DM(G_meas[-1, :, :])
        J += self.dJ_T_fn(s=Xk,
                          m=DM(Y[-1, :]).reshape((24, 1)),
                          c=DM(contact_seq[-1, :]).reshape((4, 1)),
                          tp=weight_para, G=GH, h1=self.horizon-1, ind=self.horizon-1)['dJ_Tf']

        # Create an NLP solver
        opts = {}
        opts['ipopt.tol'] = 1e-12
        opts['ipopt.print_level'] = 0
        opts['print_time'] = 0
        opts['ipopt.warm_start_init_point']='yes'
        opts['ipopt.max_iter']=1e3
        opts['ipopt.acceptable_tol']=1e-12
        opts['ipopt.mu_strategy']='adaptive'
        # opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten() # convert to a row array
        lam_g = sol['lam_g'].full().flatten() # row vector of Lagrange multipilers for bounds on g

        # Take the optimal noise, state, and costate
        sol_traj1 = np.concatenate((w_opt, self.n_noise * [0])) # sol_traj1 = [x0,w0,x1,w1,...,xk,wk,...xn-1,wn-1,xn,wn] note that we added a wn
        sol_traj = np.reshape(sol_traj1, (-1, self.n_state + self.n_noise)) # sol_traj = [[x0,w0],[x1,w1],...[xk,wk],...[xn-1,wn-1],[xn,wn]] 
        state_traj_opt = sol_traj[:, 0:self.n_state] # each xk is a row vector
        noise_traj_opt = np.delete(sol_traj[:, self.n_state:], -1, 0) # delete the last one as we have added it to make the dimensions of x and w equal
        costate_traj_ipopt = np.reshape(lam_g, (-1,self.n_state))
        
        # for k in range(state_traj_opt.shape[0]):
        #     q = state_traj_opt[k, 9:13]   # quaternion normalization [9:13]
        #     norm_q = np.linalg.norm(q)
        #     state_traj_opt[k, 9:13] = q / norm_q

        # Output
        opt_sol = {"state_traj_opt": state_traj_opt,
                   "noise_traj_opt": noise_traj_opt,
                   "costate_ipopt": costate_traj_ipopt}
        return opt_sol
    
    def diffKKT(self):
        assert hasattr(self, 'MDyn_fn'), "Define the model dynamics function first!"
        assert hasattr(self, 'output'),  "Define the output variable first!"
        assert hasattr(self, 'dJ_fn'),   "Define the cost dynamics function first!"
        assert hasattr(self, 'dJ_T_fn'), "Define the terminal cost function first!"
        assert hasattr(self, 'n_state') and hasattr(self, 'n_noise') and hasattr(self, 'n_ctrl')

        # Window and dimensions
        H  = self.N             # number of transitions
        nx = self.n_state
        nw = self.n_noise
        nu = self.n_ctrl
        
        MEAS_LEN, NOISE_LEN = 12, 24
        tp = SX.sym('tp', 1, nx + MEAS_LEN + NOISE_LEN)     # weight params
        
        Xhat = SX.sym('Xhat', nx, 1)                         # arrival estimate for x0
        Y    = [SX.sym(f"Y_{k}", 24, 1) for k in range(H+1)] # measurements (24x1)
        C    = [SX.sym(f"C_{k}",  4, 1) for k in range(H+1)] # contacts (4x1)
        U    = [SX.sym(f"U_{k}", nu, 1) for k in range(H)]   # controls (nux1)
        X    = [SX.sym(f"X_{k}", nx, 1) for k in range(H+1)] # state (nxx1)
        W    = [SX.sym(f"W_{k}", nw, 1) for k in range(H)]   # noise (nwx1)
        Lambda = [SX.sym(f"lambda_{k}", nx, 1) for k in range(H)]   # lagragian multipiler (nxx1)

        # full horizon G as a single vector，then slice back into 24×9
        Gvec = SX.sym('Gvec', (H+1)*24*9, 1)
        # Gvec: ((H+1)*24*9) x 1, reshape into 216*1 as row first order
        def G_slice(k):
            start = k*24*9
            v = Gvec[start:start+24*9]          # 216x1
            return transpose(reshape(v, 9, 24)) # back into-> (24x9)，row first order

        
        J = 0
        g = []
        # L = mtimes(Lambda[0].T, (X[0] - Xhat))
        L = 0        
        L += 1/2 * mtimes(mtimes(transpose(X[0] - Xhat), diag(tp[0, 0:nx])), (X[0] - Xhat))
        J += 1/2 * mtimes(mtimes(transpose(X[0] - Xhat), diag(tp[0, 0:nx])), (X[0] - Xhat))
        # g += [(X[0] - Xhat)]
        for k in range(H):

                # running cost
                Gk = G_slice(k)
                L += self.dJ_fn(s=X[k], m=Y[k], c=C[k], n=W[k], tp=tp, G=Gk, h1=H, ind=k)['dJrunf']
                J += self.dJ_fn(s=X[k], m=Y[k], c=C[k], n=W[k], tp=tp, G=Gk, h1=H, ind=k)['dJrunf']

                # dynamics g_k = X_{k+1} - f(X_k, U_k, N_k)
                Xnext = self.MDyn_fn(s=X[k], c=U[k], n=W[k])['MDynf']

                L    += mtimes(Lambda[k].T, (X[k+1] - Xnext))
                g += [(X[k+1] - Xnext)]

        GH = G_slice(H)
        L += self.dJ_T_fn(s=X[H], m=Y[H], c=C[H], tp=tp, G=GH, h1=H, ind=H)['dJ_Tf']
        J += self.dJ_T_fn(s=X[H], m=Y[H], c=C[H], tp=tp, G=GH, h1=H, ind=H)['dJ_Tf']

        Xvec = vertcat(*X)            # shape nx*(H+1) x 1
        Wvec = vertcat(*W)            # shape nw*H x 1 
        Lamvec = vertcat(*Lambda)  # shape nx*(H+1) x 1
        Z_vec = vertcat(Xvec, Wvec, Lamvec)
        
        Y_vec = vertcat(*Y)
        U_vec = vertcat(*U)
        C_vec = vertcat(*C)
        g_vec = vertcat(*g)
        G_list = [SX.sym(f"G_{k}", 24, 9) for k in range(H+1)]
        G_vec  = vertcat(*[reshape(Gk, 24*9, 1) for Gk in G_list])  # ( (H+1)*216, 1 )

        self.KKT = gradient(L, Z_vec) 
        self.dKKT_Z = jacobian(self.KKT, Z_vec) 
        self.dKKT_tp = jacobian(self.KKT, tp) 
        self.dKKT_Y_fn  = jacobian(self.KKT, Y_vec)

        self.Cost_fn = Function('J',    [Xvec, Wvec, Lamvec, Y_vec, U_vec, C_vec, Xhat, tp, Gvec], [J],
                                ['s','n','costate','y','u','c','prior','tp','G'], ['Cost_fn'])
        self.g_fn    = Function('g_vec',[Xvec, Wvec, Lamvec, Y_vec, U_vec, C_vec, Xhat, tp, Gvec], [g_vec],
                                ['s','n','costate','y','u','c','prior','tp','G'], ['g_fn'])
        self.KKT_fn  = Function('KKT',  [Xvec, Wvec, Lamvec, Y_vec, U_vec, C_vec, Xhat, tp, Gvec], [self.KKT],
                                ['s','n','costate','y','u','c','prior','tp','G'], ['KKT_fn'])
        self.dKKT_Z_fn  = Function('dKKT_Z',[Xvec, Wvec, Lamvec, Y_vec, U_vec, C_vec, Xhat, tp, Gvec], [self.dKKT_Z],
                                ['s','n','costate','y','u','c','prior','tp','G'], ['dKKT_Z_fn'])
        self.dKKT_tp_fn = Function('dKKT_tp',[Xvec, Wvec, Lamvec, Y_vec, U_vec, C_vec, Xhat, tp, Gvec], [self.dKKT_tp],
                                ['s','n','costate','y','u','c','prior','tp','G'], ['dKKT_tp_fn'])
        self.dKKT_Y_fn     = Function('dKKT_Y', [Xvec, Wvec, Lamvec, Y_vec, U_vec, C_vec, Xhat, tp, Gvec], [self.dKKT_Y_fn],
                        ['s','n','costate','y','u','c','prior','tp','G'], ['dKKT_Y_fn'])

        self.dKKT_G   = jacobian(self.KKT, G_vec)

        self.dKKT_G_fn = Function('dKKT_G',
            [Xvec, Wvec, Lamvec, Y_vec, U_vec, C_vec, Xhat, tp, G_vec],
            [self.dKKT_G],
            ['s','n','costate','y','u','c','prior','tp','G'],
            ['dKKT_G'])
    @staticmethod
    def quaternion_to_rotation_matrix(q):
        # q: quaternion [qx, qy, qz, qw]

        norm_q = sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
        qx = q[0] / norm_q
        qy = q[1] / norm_q
        qz = q[2] / norm_q
        qw = q[3] / norm_q

        return vertcat(
            horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)),
            horzcat(2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)),
            horzcat(2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2))
        )


    @staticmethod
    def rotation_matrix_log(R):
        # Compute trace and rotation angle
        eps=1e-6
        trace_R = R[0,0] + R[1,1] + R[2,2]
        theta = acos((trace_R - 1) / 2)
        # Compute the skew-symmetric matrix and map to vector
        factor = if_else(theta > eps, theta / (2 * sin(theta)), 0.5 + (3 - trace_R) / 12)
        log_R_matrix =  factor * (R - R.T)
        return vertcat(
            log_R_matrix[2,1],
            log_R_matrix[0,2],
            log_R_matrix[1,0]
        )

    @staticmethod
    def norm3(v):
        # v: 3x1 SX
        return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    
    def diffquat(self):
            H = self.N  # number of transitions; there are (H+1) states/poses

            # Quaternions for MHE estimate and mocap (symbolic)
            Q  = [SX.sym(f"q_{k}",  4, 1) for k in range(H+1)]   # q_mhe(k)
            Qm = [SX.sym(f"qm_{k}", 4, 1) for k in range(H+1)]   # q_mocap(k)

            # Build the scalar loss L
            L = 0
            for k in range(H+1):  # sum k=0..H
                R_mhe   = MHE.quaternion_to_rotation_matrix(Q[k])
                R_mocap = MHE.quaternion_to_rotation_matrix(Qm[k])
                R_rel   = mtimes(R_mhe, R_mocap.T)                 # R_mhe * R_mocap^T
                w_log   = MHE.rotation_matrix_log(R_rel)           # 3x1
                L       = L + MHE.norm3(w_log)

            # dL/d q_k, stack vertically -> (4*(H+1)) x 1
            grad_blocks = [ gradient(L, Q[k]) for k in range(H+1) ]  # each 4x1
            dLdQ = vertcat(*grad_blocks)

            q_stack  = vertcat(*Q)   # (4*(H+1))x1
            qm_stack = vertcat(*Qm)  # (4*(H+1))x1
            self.dL_dQ_fn = Function('dL_dQ',
                                    [q_stack, qm_stack],
                                    [dLdQ],
                                    ['q', 'qm'],
                                    ['dL_dQ'])
            # L 
            self.L_att_fn = Function('L_att',
                                    [q_stack, qm_stack],
                                    [L],
                                    ['q', 'qm'],
                                    ['L'])
            return self.dL_dQ_fn, self.L_att_fn
    

    # —— diffKKT() / diffquat() —— 
    def save_derivative_bundle(self, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        # save meta for sanity check
        meta = {
            "N": self.N, "n_state": self.n_state, "n_noise": self.n_noise, "n_ctrl": self.n_ctrl,
            "note": "Change any of these? Rebuild and overwrite the cache."
        }
        with open(os.path.join(cache_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # save every Function
        def _save(name):
            fn = getattr(self, name, None)
            if fn is not None: fn.save(os.path.join(cache_dir, f"{name}.casadi"))
        for name in ["Cost_fn","g_fn","KKT_fn","dKKT_Z_fn","dKKT_tp_fn","dKKT_Y_fn","dKKT_G_fn",
                     "dL_dQ_fn","L_att_fn"]:
            _save(name)

    # —— load saved casadi function —— 
    def load_or_build_derivatives(self, cache_dir):
        try:
            with open(os.path.join(cache_dir, "meta.json"), "r") as f:
                meta = json.load(f)
            assert meta["N"] == self.N and meta["n_state"] == self.n_state \
                and meta["n_noise"] == self.n_noise and meta["n_ctrl"] == self.n_ctrl, \
                "Cache dims mismatch; rebuild needed."

            def _load(name):
                path = os.path.join(cache_dir, f"{name}.casadi")
                if os.path.exists(path):
                    setattr(self, name, Function.load(path))
            for name in ["Cost_fn","g_fn","KKT_fn","dKKT_Z_fn","dKKT_tp_fn","dKKT_Y_fn","dKKT_G_fn",
                         "dL_dQ_fn","L_att_fn"]:
                _load(name)

            #
            assert hasattr(self, "KKT_fn") and hasattr(self, "dKKT_Z_fn"), "Partial cache; rebuild."
            return True  # successfully reloaded from casadi function file
        except Exception as e:
            print(f"[cache miss] {e}\nRebuilding derivatives...")
            self.diffKKT()
            self.diffquat()
            self.save_derivative_bundle(cache_dir)
            return False
