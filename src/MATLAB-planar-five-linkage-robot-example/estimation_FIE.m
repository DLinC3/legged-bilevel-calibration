classdef estimation_FIE < handle

    %%%%% 
    properties (Access = public)
        
        dt = 0;
        % N = 4; % MHE window
        %% states
        %%% [p; v; p_lf; p_rf;]  
        %%% kinematics joint index order: rhip; rknee; lhip; lknee
        nstate = 2 + 2 + 2*2 % [p; v; p_lf; p_rf;] in 2D
        nmeas = 2*2 + 2*2;
        %% MHE Optimization
        init = 0;
        % x_MHE;
        x_FIE;
        x_vec;

        A_dyn;
        b_dyn;

        A_meas;
        b_meas;
        % q,dq,ddq,contact information for covariance update
        q_hist       = [];   % 7×K
        dq_hist      = [];   % 7×K
        ddq_hist     = [];   % 7×K
        contact_hist = [];   % 1×K
       
        A_dyn_stack = {}; % xk+1=Ak xk + bk + δxk
        b_dyn_stack = {};
        Q_dyn_stack = {}; % information
        C_dyn_stack = {}; % covariance

        A_meas_stack = {};
        b_meas_stack = {};
        Q_meas_stack = {};
        C_meas_stack = {};

        p_std; % position noise
        accel_std; % accel noise
        p_f_stand_std; % stand feet position noise
        p_f_swing_std; % swing feet position noise
        
        joint_p_std;% joint angle noise
        joint_v_std;% joint velo noise
        omega_std; % base frame angular velo std
        
        % Prior Covariance Std
        p_init_std;
        v_init_std;
        p_f_init_std;

        H_arrival;
        h_arrival;
        
        % Used when deriving prior from KF
        C_arrival;
        x_arrival;

        %% Measurements
        %%% IMU
        accel = zeros(2,1); % accelerometer
        omega = 0; % gyroscope
        Rot = 0; % AHRS (some direct orientation measurement)
            
        %%% Joint encoders
        q = zeros(7,1);
        dq= zeros(7,1);
        
        %%% Contact sensor
        contact;

        link_error;
    end
    
    %% Kinematics functions
    methods 
        function p_lf = p_lf_b(obj,q)
            p_lf = pLeftToe_d([0;0;0;q(4:7)], obj.link_error);
        end

        function p_rf = p_rf_b(obj,q)
            p_rf = pRightToe_d([0;0;0;q(4:7)], obj.link_error);
        end

        function J_lf = J_lf_b(obj,q)
            J_lf = J_leftToe_d([0;0;0;q(4:7)], obj.link_error);
            J_lf = J_lf(1:2,6:7);
        end

        function J_rf = J_rf_b(obj,q)
            J_rf = J_rightToe_d([0;0;0;q(4:7)], obj.link_error);
            J_rf = J_rf(1:2,4:5);
        end
        function J_lf = J_lf_lwa(obj,q)
            J_lf = J_leftToe_d([0;0;0;q(4:7)], obj.link_error);
            J_lf = J_lf(1:2,3:7);
        end

        function J_rf = J_rf_lwa(obj,q)
            J_rf = J_rightToe_d([0;0;0;q(4:7)], obj.link_error);
            J_rf = J_rf(1:2,3:7);
        end
     
        function Rot = theta_2_Rot(obj,theta) 
            Rot = [cos(theta) -sin(theta);
                   sin(theta) cos(theta)];
        end
        function Skew = so2_skew(obj,omega)
            Skew = omega *  [0 -1;1 0];
        end
    end
    

    methods 
        function obj = estimation_FIE(dt,theta) % take frequency and covariance information as input
            obj.dt = dt; 
            obj.setter(theta); % modify std parameter for solver(not the covaraiance matrix we stack in)
        end
        function setter(obj,theta) % function for update std parameter 
            obj.p_std           = theta(1:3);
            obj.accel_std       = theta(4:6);
            obj.p_f_stand_std   = theta(7:8);
            obj.p_f_swing_std   = theta(9:10);
            obj.joint_p_std     = theta(11:13);
            obj.joint_v_std     = theta(14:16);
            obj.omega_std       = theta(17);
            obj.p_init_std      = theta(18:19);
            obj.v_init_std      = theta(20:21);
            obj.p_f_init_std    = theta(22:23);
            obj.link_error       = theta(24);
        end

        function reset_arrival(obj)
            % use the new link_bias computed in last iteration 
            % to compute the coordinates in world frame
            if isempty(obj.q_hist);  return;  end
            p_lf_0 = obj.p_lf_b(obj.q_hist(:,1));     % already use the new obj.link_bias in p_lf_b function
            p_rf_0 = obj.p_rf_b(obj.q_hist(:,1));     
            obj.x_arrival(5:6)  = obj.x_arrival(1:2) + p_lf_0;
            obj.x_arrival(7:8) = obj.x_arrival(1:2) + p_rf_0;
            obj.h_arrival = -obj.H_arrival * obj.x_arrival;
        end

        function x_vec = run_FIE(est, theta) 
            est.setter(theta); % set std parameter
            est.reset_arrival();
            est.rebuild_cov(); % once get the parameter, rebuild covariance matrix and stack into solver
            % tic
            est.solve_FIE();
            % toc
            x_vec = est.x_vec; % get the (Time step*state dimension)*1 vector
        end
        function J = jacobian(est, std, e)
            if nargin<3, e = 1e-8; end % set 1e-6 if no finite difference input recieve
            n_std = numel(std);
            base = run_FIE(est, std);
            S    = numel(base);
            J    = zeros(S, n_std); % initial jacobian matrix(TD*N)
        
            parfor j = 1:n_std % compute for in parallel
                warning('off', 'MATLAB:singularMatrix');
                warning('off', 'MATLAB:nearlySingularMatrix');
                theta_p      = std;
                theta_p(j)   = theta_p(j) + e;
                f_p          = run_FIE(est, theta_p);
                J(:,j)       = (f_p - base) / e;
            end
        end

        function rebuild_cov(obj)

            obj.A_dyn_stack = {};  
            obj.b_dyn_stack = {};
            obj.Q_dyn_stack = {};  
            obj.C_dyn_stack = {};
            obj.A_meas_stack= {};  
            obj.b_meas_stack= {};
            obj.Q_meas_stack= {};  
            obj.C_meas_stack= {};
        
            K = size(obj.q_hist, 2);
        
            % Since length(Q_meas_stack) == length(A_dyn_stack) + 1
            q0   = obj.q_hist(:,1);
            dq0  = obj.dq_hist(:,1);
            ddq0 = obj.ddq_hist(:,1);
            c0   = obj.contact_hist(1);
        
            obj.get_measments(q0, dq0, ddq0, c0);
            obj.update_meas_constraints(0);         % only meas
        
            % for 1 to K-1 dyn(k-1) + meas(k)
            for k = 2:K
                % dyn use last time step
                q_prev   = obj.q_hist(:,k-1);
                dq_prev  = obj.dq_hist(:,k-1);
                ddq_prev = obj.ddq_hist(:,k-1);
                c_prev   = obj.contact_hist(k-1);
        
                obj.get_measments(q_prev, dq_prev, ddq_prev, c_prev);
                obj.update_dyn_constraints(k-1, q_prev, dq_prev, ddq_prev, c_prev);
        
                % meas use current time step
                q_cur   = obj.q_hist(:,k);
                dq_cur  = obj.dq_hist(:,k);
                ddq_cur = obj.ddq_hist(:,k);
                c_cur   = obj.contact_hist(k);
        
                obj.get_measments(q_cur, dq_cur, ddq_cur, c_cur);
                obj.update_meas_constraints(k-1);       % index doesn't matter
            end
    
            % arrival cost stay the same
            obj.C_arrival = blkdiag(diag(obj.p_init_std.^2), ...
                                    diag(obj.v_init_std.^2), ...
                                    diag(obj.p_f_init_std.^2), ...
                                    diag(obj.p_f_init_std.^2));
            obj.H_arrival = inv(obj.C_arrival);
            obj.h_arrival = -obj.H_arrival * obj.x_arrival;
        end



        function get_measments(obj, q, dq, ddq, contact)
            % simulate the sensors
            % Note add the covariance
            % sim imu
            obj.accel = ddq(1:2); % in base frame
            obj.omega = dq(3);
            % sim AHRS(some direct orientation measurement)
            obj.Rot = obj.theta_2_Rot(q(3)); % World to body

            % sim joint encoders 
            obj.q = q;
            obj.dq = dq;
            
            % sim contact sensors
            obj.contact = contact;
        end


        function update_dyn_constraints(obj,T,q,dq,ddq,contact)
            % ---- dynamic constraints --------            
            %%% x_next = A_dyn x - b_dyn + delta_dyn
            %%% G is the jacobian of dynamics model wrt noise (input level)
            G_dyn = zeros(obj.nstate,obj.nstate);
            G_dyn(1:2,1:2) = obj.Rot * obj.dt; % p , p_std
            G_dyn(1:2,3:4) = - 0.5 * obj.Rot * obj.dt^2; % p, accel_std
            G_dyn(3:4,3:4) = - obj.Rot * obj.dt; % v， accel_std
            G_dyn(5:6,5:6) = obj.Rot * obj.dt; % p_f_lf
            G_dyn(7:8,7:8) = obj.Rot * obj.dt; % p_f_rf

            switch obj.contact
                case AmberConstants.RightFootContact % 'RightFootContact'
                    C_p_lf_process = diag(obj.p_f_swing_std.^2);
                    C_p_rf_process = diag(obj.p_f_stand_std.^2);
                case AmberConstants.LeftFootContact  %'LeftFootContact'
                    C_p_lf_process = diag(obj.p_f_stand_std.^2);
                    C_p_rf_process = diag(obj.p_f_swing_std.^2);
                case AmberConstants.DoubleSupport % 'DoubleSupport'
                    C_p_lf_process = diag(obj.p_f_stand_std.^2);
                    C_p_rf_process = diag(obj.p_f_stand_std.^2);
            end            
            C_p     = [obj.p_std(1)  obj.p_std(2);
                      obj.p_std(2)  obj.p_std(3)];
           
           C_accel = [obj.accel_std(1)  obj.accel_std(2);
                      obj.accel_std(2)  obj.accel_std(3)];

           C_input = blkdiag(C_p, ...
               C_accel, ...
               C_p_lf_process, ...
               C_p_rf_process);
           
           C_dyn = G_dyn * C_input * G_dyn';
           Q_dyn = inv(C_dyn);
           
           obj.A_dyn = eye(obj.nstate); % equation 24
           obj.A_dyn(1:2,3:4) = obj.dt * eye(2); 

           obj.b_dyn = zeros(obj.nstate,1); % equation 24 constant term
           obj.b_dyn(1:2) = -0.5 * obj.Rot * obj.dt^2 * obj.accel;
           obj.b_dyn(3:4) = - obj.dt * obj.Rot * obj.accel;

           %% Stack linear constraints
           obj.A_dyn_stack{end+1} = obj.A_dyn;
           obj.b_dyn_stack{end+1} = obj.b_dyn;
           obj.Q_dyn_stack{end+1} = Q_dyn;
           obj.C_dyn_stack{end+1} = C_dyn;

           % %% Keep fixed-size stack (sliding window of size obj.N+1)
          % if T >= obj.N + 1
          %     obj.A_dyn_stack(1)     = [];  % remove first element
          %     obj.b_dyn_stack(1)     = [];
          %     obj.Q_dyn_stack(1)     = [];
          %     obj.C_dyn_stack(1)     = [];
          % end
      end

      function update_meas_constraints(obj,T) % LO
          C_joint_p = [obj.joint_p_std(1)  obj.joint_p_std(2);
             obj.joint_p_std(2)  obj.joint_p_std(3)];
             
          C_joint_v = [obj.joint_v_std(1)  obj.joint_v_std(2);
                       obj.joint_v_std(2)  obj.joint_v_std(3)];
          C_omega = obj.omega_std^2;

          %%% 0 = A_meas x - b_meas + delta_meas
          b_p_lf = obj.Rot * obj.p_lf_b(obj.q); % equation 15
          b_p_rf = obj.Rot * obj.p_rf_b(obj.q);
          
          J_lf = obj.J_lf_b(obj.q);
          J_rf = obj.J_rf_b(obj.q); % equation 17
          % b_v_lf =  obj.Rot * (J_lf * obj.dq(6:7) + obj.so2_skew(obj.omega) * obj.p_lf_b(obj.q)); % v_foot_left - v_base
          % b_v_rf =  obj.Rot * (J_rf * obj.dq(4:5) + obj.so2_skew(obj.omega) * obj.p_rf_b(obj.q)); % v_foot_right - v_base

          J_lf_lwa = obj.J_lf_lwa(obj.q);
          J_rf_lwa = obj.J_rf_lwa(obj.q); % equation 17
      
          b_v_lf =  J_lf_lwa * obj.dq(3:7);
          b_v_rf =  J_rf_lwa * obj.dq(3:7);


          obj.A_meas = zeros(obj.nmeas,obj.nstate);
          obj.A_meas(1:2,1:2) = -eye(2); % p
          obj.A_meas(1:2,5:6) = eye(2);  % p_lf
          obj.A_meas(3:4,1:2) = -eye(2); % p
          obj.A_meas(3:4,7:8) = eye(2); % p_rf

          obj.A_meas(5:6,3:4) = -eye(2); % v
          obj.A_meas(7:8,3:4) = -eye(2); % v

          obj.b_meas = [b_p_lf;b_p_rf;b_v_lf;b_v_rf];

          Q_p_lf = obj.Rot * inv(J_lf *  C_joint_p * J_lf') * obj.Rot'; % ?
          Q_p_rf = obj.Rot * inv(J_rf *  C_joint_p * J_rf') * obj.Rot';
          C_p_lf = obj.Rot * J_lf *  C_joint_p * J_lf' * obj.Rot';
          C_p_rf = obj.Rot * J_rf *  C_joint_p * J_rf' * obj.Rot';  

          %%% G is the jacobian of dynamics model wrt noise (input level)
          % G_v_lf = [J_lf, obj.so2_skew(obj.omega)*J_lf, [0 1;-1 0]* obj.p_lf_b(obj.q)];
          % G_v_rf = [J_rf, obj.so2_skew(obj.omega)*J_rf, [0 1;-1 0]* obj.p_rf_b(obj.q)];
          G_v_lf = [obj.so2_skew(obj.omega)*J_lf, J_lf, [0 1;-1 0]* obj.p_lf_b(obj.q)];
          G_v_rf = [obj.so2_skew(obj.omega)*J_rf, J_rf, [0 1;-1 0]* obj.p_rf_b(obj.q)];
          switch obj.contact
              case AmberConstants.RightFootContact % 'RightFootContact'
                  C_v_lf = diag(obj.p_f_swing_std.^2);
                  C_v_rf = obj.Rot * G_v_rf * blkdiag(C_joint_p,C_joint_v,C_omega) * G_v_rf' * obj.Rot';

                  Q_v_lf = inv(diag(obj.p_f_swing_std.^2));
                  Q_v_rf = obj.Rot * inv(G_v_rf * blkdiag(C_joint_p,C_joint_v,C_omega) * G_v_rf') * obj.Rot';
              case AmberConstants.LeftFootContact  %'LeftFootContact'
                  C_v_lf = obj.Rot * G_v_lf * blkdiag(C_joint_p,C_joint_v,C_omega) * G_v_lf' * obj.Rot';
                  C_v_rf = diag(obj.p_f_swing_std.^2);

                  Q_v_lf = obj.Rot * inv(G_v_lf * blkdiag(C_joint_p,C_joint_v,C_omega) * G_v_lf') * obj.Rot';
                  Q_v_rf = inv(diag(obj.p_f_swing_std.^2));
              case AmberConstants.DoubleSupport % 'DoubleSupport'
                  C_v_lf = obj.Rot * G_v_lf * blkdiag(C_joint_p,C_joint_v,C_omega) * G_v_lf' * obj.Rot';
                  C_v_rf = obj.Rot * G_v_rf * blkdiag(C_joint_p,C_joint_v,C_omega) * G_v_rf' * obj.Rot';

                  Q_v_lf = obj.Rot * inv(G_v_lf * blkdiag(C_joint_p,C_joint_v,C_omega) * G_v_lf') * obj.Rot';
                  Q_v_rf = obj.Rot * inv(G_v_rf * blkdiag(C_joint_p,C_joint_v,C_omega) * G_v_rf') * obj.Rot';
          end       
          
          Q_meas = blkdiag(Q_p_lf,Q_p_rf,Q_v_lf,Q_v_rf);
          C_meas = blkdiag(C_p_lf,C_p_rf,C_v_lf,C_v_rf);

          %% Stack linear constraints
          obj.A_meas_stack{end+1} = obj.A_meas;
          obj.b_meas_stack{end+1} = obj.b_meas;
          obj.Q_meas_stack{end+1} = Q_meas;
          obj.C_meas_stack{end+1} = C_meas;

          % %% Keep fixed-size stack (sliding window of size obj.N+1)
          % if T >= obj.N + 1
          %     T
          %     obj.A_meas_stack(1)     = [];  % remove first element
          %     obj.b_meas_stack(1)     = [];
          %     obj.Q_meas_stack(1)     = [];
          %     obj.C_meas_stack(1)     = [];
          % end
      end
  end

  methods
      function update_estimation(obj, T, q, dq, ddq, contact)
          if (obj.init == 0)
              obj.init_FIE(T,q, dq, ddq, contact);
              obj.init = 1;
          else
              % tic
              % obj.update_MHE(T, q, dq, ddq, contact);
              % toc
              obj.update_dyn_constraints(T,q,dq,ddq,contact);
              obj.get_measments(q,dq,ddq,contact);
              obj.update_meas_constraints(T);
              obj.q_hist       = [obj.q_hist,       q];
              obj.dq_hist      = [obj.dq_hist,      dq];
              obj.ddq_hist     = [obj.ddq_hist,     ddq];
              obj.contact_hist = [obj.contact_hist, contact];

          end
      end

      function init_FIE(obj, T, q, dq, ddq, contact)
          obj.get_measments(q, dq, ddq, contact);

          obj.C_arrival = blkdiag(diag(obj.p_init_std.^2), ...
              diag(obj.v_init_std.^2), ...
              diag(obj.p_f_init_std.^2), ...
              diag(obj.p_f_init_std.^2));
          
          obj.H_arrival = inv(obj.C_arrival); 
          obj.x_arrival = zeros(obj.nstate,1);
          obj.x_arrival(1) = -0.0231924;
          obj.x_arrival(2) = 0.404844;
          p_lf = obj.x_arrival(1:2) + obj.p_lf_b(obj.q);
          p_rf = obj.x_arrival(1:2) + obj.p_rf_b(obj.q);
          obj.x_arrival(5:6) = p_lf;
          obj.x_arrival(7:8) = p_rf;
          obj.h_arrival = -obj.H_arrival*obj.x_arrival;
          obj.update_meas_constraints(T);

          obj.q_hist       = q;
          obj.dq_hist      = dq;
          obj.ddq_hist     = ddq;
          obj.contact_hist = contact;

          % obj.x_MHE = zeros(obj.nstate, 1);
      end

      function solve_FIE(obj)
          %% FIE optimization (QP problem, while solved directly using ipopt)
          D = numel(obj.A_dyn_stack);
          opts = struct();
          opts.ipopt.print_level = 0;   % suppress IPOPT iteration log
          opts.ipopt.sb = 'yes';        % silent banner
          opts.print_time = false;      % suppress timing output
          opti = casadi.Opti(); % Optimization problem

          % ---- decision variables ---------
          X = opti.variable(obj.nstate,D+1); 
          delta_dyn = opti.variable(obj.nstate,D);  
          delta_meas = opti.variable(obj.nmeas,D+1);   

          % ---- objective & constraints ---------
          objective = 0;
          % objective = objective +  X(:,1)' * obj.H_arrival * X(:,1) + 2 * obj.h_arrival * X(:,1);
          objective = objective +  X(:,1)' * obj.H_arrival * X(:,1) + 2 * obj.h_arrival' * X(:,1); 
          for i = 1:D
              objective = objective + delta_meas(:,i)' * obj.Q_meas_stack{i} * delta_meas(:,i);
              opti.subject_to(zeros(obj.nmeas,1)==obj.A_meas_stack{i} * X(:,i) -  obj.b_meas_stack{i} + delta_meas(:,i)); 

              objective =  objective + delta_dyn(:,i)' * obj.Q_dyn_stack{i} * delta_dyn(:,i);
              opti.subject_to(X(:,i+1)==obj.A_dyn_stack{i} * X(:,i) - obj.b_dyn_stack{i} + delta_dyn(:,i));  
          end
          objective = objective + delta_meas(:,D+1)' * obj.Q_meas_stack{D+1} * delta_meas(:,D+1);
          opti.subject_to(zeros(obj.nmeas,1)==obj.A_meas_stack{D+1} * X(:,(D+1)) -  obj.b_meas_stack{D+1} + delta_meas(:,D+1)); 
          
          opti.minimize(objective);
          opti.solver('ipopt', opts); % set numerical backend

          % opti.debug;
          sol = opti.solve();   % actual solve
          solX = sol.value(X);
          obj.x_FIE = solX;
          [n, S] = size(solX);
          obj.x_vec = reshape(solX, n*S, 1);

      end
      

  end
end