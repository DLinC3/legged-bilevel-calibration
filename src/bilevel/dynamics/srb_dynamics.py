
from casadi import *
import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class SrbDynamics:
    def __init__(self, dt_sample):
        # Position in inertial frame
        self.p    = SX.sym('p',3,1)
        # Velocity in inertial frame
        self.v    = SX.sym('v',3,1)
        # Quaternion from inertial frame to body frame
        self.qx, self.qy, self.qz, self.qw = SX.sym('qx'), SX.sym('qy'), SX.sym('qz'), SX.sym('qw')
        self.q = vertcat(self.qx, self.qy, self.qz, self.qw)
        # Foot position in inertial frame for quadrapedal
        self.pf = SX.sym('pf', 12,1)
        # Angular velocity in body frame
        self.omegax, self.omegay, self.omegaz    = SX.sym('omegax'), SX.sym('omegay'), SX.sym('omegaz')
        self.omega    = vertcat(self.omegax, self.omegay, self.omegaz)
        self.omega_walk = SX.sym('omega_walk',3,1)  # Angular velocity white noise bias
        # Acceleration in body frame   
        self.a       = SX.sym('a',3,1)  
        self.a_walk  = SX.sym('a_walk',3,1)  # Acceleration white noise bias
        # self.pf_walk = SX.sym('pf_walk',12,1)
        # Control input u: 6-by-1 vecter
        self.u     = vertcat(self.a, self.omega)
        # Process noise for acceleration
        self.wa    = SX.sym('wa',3,1)
        # Acceleration white noise bias
        self.wa_walk = SX.sym('wa_walk',3,1)  
        # Process noise for angular velocity
        self.womega    = SX.sym('womega',3,1)
        # Angular velocity white noise bias
        self.womega_walk = SX.sym('womega_walk',3,1)  
        # Process noise for foot position change
        self.wpf = SX.sym('wpf',12,1)
        # Process noise
        self.w     = vertcat(self.wa, self.womega, self.wa_walk , self.womega_walk , self.wpf)
        # Discretization step in MHE and geometric controller
        self.dt    = dt_sample
        # Unit direction vectors free of coordinate
        self.ex    = vertcat(1, 0, 0)
        self.ey    = vertcat(0, 1, 0)
        self.ez    = vertcat(0, 0, 1)
        # Gravitational acceleration 
        self.g     = 9.81

    # Augmented SRB dynamics, state:x, control:u, disturbance:d
    def noise_dywb(self,x,u,noise):
            # Position
            p      = vertcat(x[0,0], x[1,0], x[2,0])
            # Velocity
            v      = vertcat(x[3,0], x[4,0], x[5,0])
            # Acceleration white noise bias
            a_walk = vertcat(x[6,0], x[7,0], x[8,0])
            # Quaternion
            q      = vertcat(x[9,0], x[10,0], x[11,0], x[12,0])
            omega_walk = vertcat(x[13,0], x[14,0], x[15,0])
            # Foot postion
            pf      = vertcat(x[16:28,0])
            # Process noise for acceleration
            wa     = vertcat(noise[0,0], noise[1,0], noise[2,0])
            # Process noise for angular velocity
            womega     = vertcat(noise[3,0], noise[4,0], noise[5,0])
            wa_walk     = vertcat(noise[6,0], noise[7,0], noise[8,0])
            womega_walk     = vertcat(noise[9,0], noise[10,0], noise[11,0])
            # Rotation matrix
            R_B = self.quaternion_to_rotation_matrix(q)
            # Acceleration in body frame
            a      = vertcat(u[0,0], u[1,0], u[2,0]) + wa + a_walk
            # Angular velocity in body frame
            omegax, omegay, omegaz     = u[3,0], u[4,0], u[5,0]
            omega  = vertcat(omegax, omegay, omegaz) + womega + omega_walk
            # Process noise for foot position
            wpf    = vertcat(noise[12:24,0])
            # Dynamics model augmented by the random walk model of the disturbance
            dp     = v
            dv     = -self.g*self.ez + mtimes(R_B, a)
            da    = wa_walk
            # dq = 0.5 * self.omega_matrix(omega) @ q
            dq       = 0.5 * self.quat_multiply_elementwise(q, vertcat(omega, DM(0.)))
            domega  = womega_walk
            dpf   = wpf
            dyn_wb = vertcat(dp, dv, da, dq, domega, dpf)
            return dyn_wb

    def model(self):
            # Single Rigid Body (SRB) dynamics model
            self.xa = vertcat(self.p, self.v, self.a_walk, self.q, self.omega_walk, self.pf)
            # self.xa  = vertcat(self.p, self.v, self.q, self.pf)
            self.x   = vertcat(self.p, self.v, self.q, self.pf)
            # Output
            R_WB = self.quaternion_to_rotation_matrix(self.q)
            v_B = mtimes(R_WB.T, self.v)
            pf_FR_W = self.pf[0:3]
            pf_FL_W = self.pf[3:6]
            pf_RR_W = self.pf[6:9]
            pf_RL_W = self.pf[9:12]
            pf_FR_B = mtimes(R_WB.T, (pf_FR_W - self.p))
            pf_FL_B = mtimes(R_WB.T, (pf_FL_W - self.p))
            pf_RR_B = mtimes(R_WB.T, (pf_RR_W - self.p))
            pf_RL_B = mtimes(R_WB.T, (pf_RL_W - self.p))

            self.y  = vertcat(v_B, pf_FR_B, pf_FL_B, pf_RR_B, pf_RL_B)  # 15*1
            
            # Dynamics model
            dp        = self.v
            R_B     = self.quaternion_to_rotation_matrix(self.q)
            dv        = -self.g*self.ez + mtimes(R_B, self.a)
            dq        = 0.5 * self.omega_matrix(self.omega) @ self.q
            dq       = 0.5 * self.quat_multiply_elementwise(self.q, vertcat(self.omega, DM(0.)))
            dpf      = vertcat(DM(0.), DM(0.), DM(0.), 
                               DM(0.), DM(0.), DM(0.), 
                               DM(0.), DM(0.), DM(0.), 
                               DM(0.), DM(0.), DM(0.)
                               )
            
            xdot      = vertcat(dp, dv, dq, dpf)
            self.dywb = Function('Dywb', [self.x, self.u], [xdot], ['x0', 'u0'], ['xdot'])
            
            # 4-order Runge-Kutta discretization of the augmented model used in MHE (symbolic computation)
            k1        = self.noise_dywb(self.xa, self.u, self.w)
            k2        = self.noise_dywb(self.xa + self.dt/2*k1, self.u, self.w)
            k3        = self.noise_dywb(self.xa + self.dt/2*k2, self.u, self.w)
            k4        = self.noise_dywb(self.xa + self.dt*k3, self.u, self.w)
            self.dymh = (k1 + 2*k2 + 2*k3 + k4)/6

    def step(self, x, u, dt):
            # self.model()
            # define discrete-time dynamics using 4-th order Runge-Kutta
            k1    = self.dywb(x0=x, u0=u)['xdot'].full()
            k2    = self.dywb(x0=x+dt/2*k1, u0=u)['xdot'].full()
            k3    = self.dywb(x0=x+dt/2*k2, u0=u)['xdot'].full()
            k4    = self.dywb(x0=x+dt*k3, u0=u)['xdot'].full()
            xdot  = (k1 + 2*k2 + 2*k3 + k4)/6
            x_new = x + dt*xdot
            # components
            p_new     = np.array([[x_new[0,0], x_new[1,0], x_new[2,0]]]).T
            v_new     = np.array([[x_new[3,0], x_new[4,0], x_new[5,0]]]).T
            q_new   = np.array([[x_new[6,0], x_new[7,0], x_new[8,0], x_new[9,0]]]).T
            # q_new = q_new / np.linalg.norm(q_new)
            R_B_new   = self.quaternion_to_rotation_matrix(q_new)
            pf_new = np.array([x_new[10:22,0]]).T
            
            # Y->Z->X rotation from {b} to {I}
            gamma   = np.arctan(R_B_new[2, 1]/R_B_new[1, 1])
            theta   = np.arctan(R_B_new[0, 2]/R_B_new[0, 0])
            psi     = np.arcsin(-R_B_new[0, 1])
            Euler_new = np.array([[gamma, theta, psi]]).T
            output = {"p_new":p_new,
                    "v_new":v_new,
                    "q_new":q_new,
                    "pf_new":pf_new,
                    "Euler":Euler_new
                    }
            return output

    def skew_sym(self, v):
            v_cross = vertcat(
                horzcat(0, -v[2,0], v[1,0]),
                horzcat(v[2,0], 0, -v[0,0]),
                horzcat(-v[1,0], v[0,0], 0)
            )
            return v_cross
        
    def omega_matrix(self, w):
        # w: body frame angular velocity
        wx, wy, wz = w[0], w[1], w[2]
        return vertcat(
            horzcat(0, -wx, -wy, -wz),
            horzcat(wx, 0, wz, -wy),
            horzcat(wy, -wz, 0, wx),
            horzcat(wz, wy, -wx, 0)
        )
        
    def quat_multiply_elementwise(self, q1, q2):
        # q1, q2: [x, y, z, w] format
        x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
        x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]

        x = w1 * x2 + w2 * x1 + (y1 * z2 - z1 * y2)
        y = w1 * y2 + w2 * y1 + (z1 * x2 - x1 * z2)
        z = w1 * z2 + w2 * z1 + (x1 * y2 - y1 * x2)
        w = w1 * w2 - (x1 * x2 + y1 * y2 + z1 * z2)

        return vertcat(x, y, z, w)
    
    def quaternion_to_rotation_matrix(self, q):
        # q: quaternion [qx, qy, qz, qw]
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        return vertcat(
            horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)),
            horzcat(2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)),
            horzcat(2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2))
        )
        
    def quaternion_discrete_update(self, q, omega, dt):
        qx, qy, qz, qw = q
        wx, wy, wz = omega
        dq = np.array([
            qx + 0.5 * dt * (wx * qw - wy * qz + wz * qy),
            qy + 0.5 * dt * (wx * qz + wy * qw - wz * qx),
            qz + 0.5 * dt * (-wx * qy + wy * qx + wz * qw),
            qw - 0.5 * dt * (wx * qx + wy * qy + wz * qz)
        ])
        # Normalize to unit quaternion
        return dq / np.linalg.norm(dq)
