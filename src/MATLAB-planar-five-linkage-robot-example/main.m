clc, clear
close all
format long
%% Path 
addpath(genpath('Expression'));
addpath(genpath('library'));
addpath(genpath('controller'));
addpath(genpath('C:\Users\dcheng32\qpOASES')); % QP solver path
addpath('C:\Users\dcheng32\casadi-3.7.0-windows64-matlab2018b') % Casadi path
% addpath('C:\Users\dcheng32\Documents\MATLAB\toolbox\Mosek') % Mosek path
% addpath('C:\Users\dcheng32\Documents\MATLAB\toolbox\yalmip') % yalmip path
% addpath('C:\Users\dcheng32\Documents\MATLAB\toolbox\sedumi') % sedumi path
% addpath('C:\Users\dcheng32\Documents\MATLAB\toolbox\spotless') % spotless path
%%% To install casadi, follow the guide in: https://web.casadi.org/get/
%%% To install qpOASES, follow the guide in: https://github.com/coin-or/qpOASES/wiki/QpoasesInstallation
%%% To install MOSEK, follow the guide in: https://docs.mosek.com/11.0/toolbox/install-interface.html
%%% To install yalmip, follow the guide in: https://yalmip.github.io/tutorial/installation/
%% Supress some matlab warning
warning('off', 'MATLAB:singularMatrix');
warning('off', 'MATLAB:nearlySingularMatrix');
% pctRunOnAll warning('off', 'MATLAB:singularMatrix');
% pctRunOnAll warning('off','MATLAB:nearlySingularMatrix');
%% Simulation frequency 
Frequency = 400; %1e+3
period = 1/Frequency; 
Ncycles = 8;  % 8
%% lower and upper bound for optimization parameter
lb = 1e-4 * ones(24,1);
ub = 1      * ones(24,1);
% off-diag can be negative
lb(2)    = -1; 
lb(5)    = -1; 
lb(12)    = -1; 
lb(15)    = -1;
% swing foot position term
lb(9:10) = 1e4;  
ub(9:10) = 1e6;
% link error term
ub(24)  = 1e-2;
%% some lousy noise std settings as initial value
% All process noise modeled in base frame
% Process noise Covariance:
p_std = [1, 0.001, 1]; % position noise
accel_std = [0.001, 0.001, 1]; % accel noise
p_f_stand_std = [0.001, 1]; % stand feet position noise
p_f_swing_std = [100000,100000]; % swing feet position noise
% All measurement noise modeled in configuration space
% Measurement noise Covariance:
joint_p_std = [0.001,0.001,1];% joint angle noise
joint_v_std = [0.001,0.001,1];% joint velo noise
omega_std = 0.8; % base frame angular velo std
% Prior Covariance Std
p_init_std = [0.001, 1];
v_init_std = [0.001, 1];
p_f_init_std = [0.001, 1];
% link bias initial value, true value is 0.0475
link_error_init = 0.000;  
% stack into one optimization parameter vector
std0 = [ ...
    p_std(:); 
    accel_std(:); 
    p_f_stand_std(:); 
    p_f_swing_std(:); 
    joint_p_std(:); 
    joint_v_std(:); 
    omega_std; 
    p_init_std(:); 
    v_init_std(:); 
    p_f_init_std(:)
    link_error_init(:) 
];
stdi = std0;
%% Data logging (Don't touch)
log = struct(); 
log.finishingIdx = []; 
log.flow = struct('t', [], 'q', [], 'dq',[], 'ddq',[],  'u', [], 'uDM', [], 'comX', []);
log.QP = struct('F', [], 'delta' , [], 'exitflag', [], 'numiter',[], 'fval', [], 'GRF_lf', [],'GRF_rf', []);
log.DMC = struct('u', [], 'cost', [], 'F', [], 'exitflag', []);
log.outputs = struct('y2', [], 'dy2', [], 'y2des', [], 'dy2des', [], 'yNext', []);
logOpt = struct('alpha', [], 'p',[]);
%% Data logging for estimator
log.estimate = struct('t',[],'v_lf_gt',[],'v_rf_gt',[],'x_MHE',[],'contact',[],'x_KF',[]);
%% Controller params (Don't touch)
ifubound = 1;  % 1: use torque bounds
qsize = 7; 
miu = 0.6;  % friction? 0.6
vxdes = 0.1; % x velocity desired? previously 0.5
COMheight = 0.33;
tSSP = 0.4;  % previously 0.4
swingHeight = 0.02;  % previously 0.08 
impactVel = -0.1; % 
torsoAngle = 0; %   
gaitType = 'P1';  %%%% 'P1', 'P2' 
stepL_left = -0.05;  %%%% one step size for P2 orbits originally -0.2 
q0 = [0; 0; torsoAngle;  
      pi/2; -pi;  %%% right leg: stance leg [pi-0.5;1;pi-0.5;1] for last four digits
      -pi/2; pi]; %%% left leg: swing leg
stanceFoot = pLeftToe(q0); 
q0(2) = -stanceFoot(2); 
dq0 = zeros(7,1); 
x0 = [q0; dq0]; 
leftFootOffset = [0; 0]; 
q0 = InverseKinematicsWalker(COMheight, q0, torsoAngle, leftFootOffset);

comPosZ = COMPosition(q0)*[0;0;1]; 
param = struct('COMz', comPosZ, 'tSSP', tSSP,  'vxdes', vxdes,  'swingHeight', swingHeight,...
                'torsoAngle', 0, 'impactVel', impactVel, 'gaitType', gaitType, 'stepL_left', stepL_left);
contact = AmberConstants.RightFootContact;
QPvariableType  =  'OSC';  %%% 'onlyU' 'U-Fhol', 'U-Fhol-ddq', 'OSC'
QP_p = 1e+5; % parameters in CLF relaxation cost
log.param = param; 
%% Simulation init (Don't touch)
outputs = GenOutputsFiveLinkWalker(logOpt, tSSP, comPosZ, swingHeight, impactVel, vxdes, gaitType, stepL_left, period); 
outputs.qPelvis0 = q0(3);
eomType = struct('QPvariableType', QPvariableType);
eom = EOM_walker(7, eomType); 
eventSim = customSimWalking2D(contact); 
qp = QP_Walking(miu, QPvariableType);
q = q0; 
dq = dq0; 
d = 1; 
Nperiods = round(tSSP/period)*Ncycles;
t0 = 0; 
lastContact = contact;
ddq = zeros(7,1); 
QPfail = false; 
%% Estimation class
% est = estimation_MHE(1/Frequency);
est = estimation_FIE(1/Frequency, stdi);
%% Run simulation once
u0 = []; 
odeopts = odeset('MaxStep', 1e-3, 'RelTol', 1e-4, 'AbsTol',1e-4);
tIdx = 0;
tIdx0 = 1;
T = 0;
while (d<= Ncycles && tIdx <Nperiods)
    for tIdx = tIdx0:Nperiods
        %% Update the sim & control (Don't touch)
        if eventSim.contact ~= lastContact
            outputs.SSPnsf0 = [];
            contact = eventSim.contact;
            lastContact = eventSim.contact;
            log.finishingIdx = [log.finishingIdx, tIdx];
            d = d + 1;
             tIdx0 = tIdx + 1;
            break;
        end
        if q(2) < 0 %%% falling
            break
        end
        eom.updateWalking(q, dq, contact);
        outputs = outputs.getOutputs(q, dq, ddq, tIdx - tIdx0+1, contact);
 
        [u, exitflag, F_GRF, delta, fval, numiter, QPfail] = qp.constructAndSolve(contact, u0, eom, outputs);
        [tt, x, ddq] = eventSim.sim(period, [q; dq], eom, u, odeopts, t0, QPfail);
        xnext = x(end, :);
        q  = xnext(1:qsize)'; 
        dq = xnext(qsize+1:end)';
        tt = tt + t0*ones(size(tt));
        t0 = tt(end);
        
        %% Update the estimator
        est.update_estimation(T, q, dq, ddq, eventSim.contact)
        T = T+1;
        %% Visualization data logging (Don't touch)
        log.flow.t      =  [log.flow.t;     t0];
        log.flow.q      =  [log.flow.q;     q'];
        log.flow.dq     =  [log.flow.dq;   dq'];
        % add ddq flow
        log.flow.ddq = [log.flow.ddq;   ddq'];

        log.flow.u      =  [log.flow.u;     u'];
        log.flow.comX   =  [log.flow.comX, [outputs.comPos; outputs.comVel]]; 
        if(contact == -1) % left contact
            log.QP.GRF_lf = [log.QP.GRF_lf, F_GRF]; 
            log.QP.GRF_rf = [log.QP.GRF_rf, 0 *F_GRF];             
        else
            log.QP.GRF_lf = [log.QP.GRF_lf, 0 * F_GRF]; 
            log.QP.GRF_rf = [log.QP.GRF_rf, F_GRF]; 
        end
        log.outputs.yNext = [log.outputs.yNext;   outputs.y_kp1'];
        log.outputs.y2 = [log.outputs.y2;         outputs.RD2.y_unscaled'];
        log.outputs.dy2 = [log.outputs.dy2;       outputs.RD2.dy_unscaled'];
        log.outputs.y2des = [log.outputs.y2des;   outputs.RD2.y_des_unscaled'];
        log.outputs.dy2des = [log.outputs.dy2des; outputs.RD2.dy_des_unscaled'];
        %% Estimation data logging
        log.estimate.t = [log.estimate.t; t0];
        % log.estimate.x_MHE = [log.estimate.x_MHE est.x_MHE];
        log.estimate.contact = [log.estimate.contact eventSim.contact];
    end
end

% compute ground-truth x 
K  = size(log.flow.q,1); % total time step
xGT = zeros(8,K); % ground truth for all time step

for k = 1:K
    qk  = log.flow.q(k,:)'; % 7×1
    dqk = log.flow.dq(k,:)';
    p     = qk(1:2);
    v     = dqk(1:2);
    theta = qk(3);
    Rwb   = [cos(theta) -sin(theta);
             sin(theta)  cos(theta)]; % rotation matrix for computing ground truth foot position
    p_lf_b = pLeftToe ([0;0;0;qk(4:7)]);  p_lf_b = p_lf_b(1:2); % forward kinematics, use true fk functoin without error
    p_rf_b = pRightToe([0;0;0;qk(4:7)]);  p_rf_b = p_rf_b(1:2); % foot position in base frame
    p_lf_w = p + Rwb * p_lf_b;
    p_rf_w = p + Rwb * p_rf_b; % Ground truth foot position in world frame

    xGT(:,k) = [p; v; p_lf_w; p_rf_w]; % append to ground truth
end

log.groundtruth.x = xGT; % add to log
[n, S] = size(log.groundtruth.x);
x_GT_vec = reshape(log.groundtruth.x, n*S, 1);
%% Visualization of 5LinkWalker
% anasim = analysisWalker(log, outputs, Ncycles, eventSim, Frequency);
% anasim.animate();
%% set noise to measurement
K = size(log.flow.q,1);
% IMU noise
Cov_IMU = [ 0.05^2,  0.8*0.05*0.09 ;
             0.8*0.05*0.09, 0.09^2 ];
% joint angle noise
sigma_enc_p = 0.002;
% joint velocity noise
sigma_enc_v = 1e-4;
% rng(2025); % set random seed
% record data before corrupt
ddq_true = log.flow.ddq(:,1:2);
q_true = log.flow.q(:,4:7);
for k = 1:K
    % IMU 
    imu_noise = mvnrnd([0 0], Cov_IMU, 1)';      % 2×1
    log.flow.ddq(k,1:2) = log.flow.ddq(k,1:2) + imu_noise';
    est.ddq_hist(1:2,k) = est.ddq_hist(1:2,k)   + imu_noise;   % corrupt
    % joint encoder angle q(4:7) 
    p_noise = sigma_enc_p * randn(1,4);
    log.flow.q (k,4:7) = log.flow.q (k,4:7) + p_noise;
    est.q_hist  (4:7,k) = est.q_hist  (4:7,k) + p_noise';
    % joint velocity
    v_noise = sigma_enc_v * randn(1,4);
    log.flow.dq(k,4:7)  = log.flow.dq(k,4:7) + v_noise;
    est.dq_hist(4:7,k)  = est.dq_hist(4:7,k) + v_noise';
end
%% Store the initial simulation data that will be reused
est.rebuild_cov();
est.solve_FIE() % solve FIE once to establish the structure
log.estimate.x_FIE = est.x_FIE;
%% ===== Output directory for CSVs =====
out_dir = fullfile('C:\Users\dcheng32\Pictures', 'FW_CSV_EXPORTS');  % 
if ~exist(out_dir,'dir'), mkdir(out_dir); end

maxIter = 75;
xFIE_iters = cell(maxIter,1);   % 8×K trajectory matrix
 
%% Frank-Wolfe optimization loop

lossHist    = nan(maxIter,1);   % loss histroy for each iteration
stdHist = nan(maxIter+1,24);  % extra line
stdHist(1,:) = std0;          % initial std
gradHist   = nan(maxIter,1);   % 2-norm of gradient at each iter
dL_exp_hist  = nan(maxIter,1); % expected loss computed by frank-wolfe after linearization 
dL_act_hist  = nan(maxIter,1); % true loss computed by loss function
% solve FIE first
x_fie   = run_FIE(est, stdi);                 % θ^0
log.estimate.x_FIE = est.x_FIE;
log.estimate.x_FIE_init = est.x_FIE;      %
loss    = 0.5*(x_fie - x_GT_vec)'*(x_fie - x_GT_vec);
warm=5;
for Iter = 1: maxIter
    fprintf('=== Outer Loop iter %d ===\n', Iter);
    fprintf('loss = %.6g\n', loss);
    lossHist(Iter) = loss;% record loss for plot
    %% Gradient
    J        = jacobian(est, stdi, 1e-8);
    g        = J' * (est.x_vec - x_GT_vec);    % column vector
    gradHist(Iter) = norm(g);
    %% YALMIP
    alpha = 2/(warm+2*Iter);
    x_opt = sdpvar(24,1);
    % PSD + x_new \in PSD constraint (x_new may not in PSD constraint)
    % solved
    % PSD constraint and box constraint
    Constraint = [ ...
          [x_opt(1)  x_opt(2);  x_opt(2)  x_opt(3) ]   >= 1e-5*eye(2), ...   % Σ_p
          [x_opt(4)  x_opt(5);  x_opt(5)  x_opt(6) ]   >= 1e-5*eye(2), ...   % Σ_a
          [x_opt(11) x_opt(12); x_opt(12) x_opt(13)]   >= 1e-5*eye(2), ...   % Σ_joint_p
          [x_opt(14) x_opt(15); x_opt(15) x_opt(16)]   >= 1e-5*eye(2), ...   % Σ_joint_v
          lb    <= x_opt    <= ub                  ];  % lb ub for x_new(all vector)
    
    opt = sdpsettings('solver','mosek','verbose',0);
    optimize(Constraint, g'*x_opt, opt);

    s = value(x_opt);% Frank-Wolfe vertex

    std_new=stdi+alpha*(s-stdi); % update parameters θ^{t+1}
    % fprintf('e_calf = %f\n', stdi(24));  % link_error
    stdi = std_new;
    stdHist(Iter+1,:) = stdi;
    %% expected ΔL and ΔL
    delta          = s - stdi;                  % use new stdi(θ^{t})
    dL_expected    = alpha * (g' * delta);      % expected ΔL by linearization
    dL_exp_hist(Iter) = dL_expected;
    % state update
    x_fie   = run_FIE(est, stdi);               % only solve one FIE in one loop and put here
    log.estimate.x_FIE = est.x_FIE;
    xFIE_iters{Iter} = est.x_FIE;

    loss_new = 0.5*(x_fie - x_GT_vec)'*(x_fie - x_GT_vec);
    dL_actual = loss_new - loss;                % sign
    dL_act_hist(Iter) = dL_actual;
    %% print
    fprintf('   dL_exp  = %.6g\n', dL_expected);
    fprintf('   dL      = %.6g\n', dL_actual);
    fprintf('   ||grad|| = %.6g\n', gradHist(Iter));
    % plot_FIE(log);
    fprintf('---------------------------\n\n');
    %%
    loss = loss_new; % θ^{t+1}
end
log.estimate.x_FIE_final = est.x_FIE;
plot_vel_compare(log, 'C:\Users\dcheng32\Pictures', 'svg')
iters = 1:maxIter;
% Loss plot
figure; plot(iters, lossHist);
xlabel('Iteration'); ylabel('Loss'); title('Loss vs. iteration');
%% Visualization of 5LinkWalker after corrupt data
% anasim = analysisWalker(log, outputs, Ncycles, eventSim, Frequency);
% anasim.animate()

out_csv = fullfile('C:\Users\dcheng32\Pictures', 'std_history.csv');
writematrix(stdHist, out_csv);

%% ====== CSV EXPORTS ======
iters = (1:maxIter).';           
iters0 = (0:maxIter).';          
t = log.estimate.t(:);    
K = numel(t);

% ---------- (A) Loss vs iteration ----------
T_loss = table(iters, lossHist(:), ...
    'VariableNames', {'iter','loss'});
writetable(T_loss, fullfile(out_dir,'loss_history.csv'));

% ---------- (B) ||grad|| vs iteration ----------
T_grad = table(iters, gradHist(:), ...
    'VariableNames', {'iter','grad_norm'});
writetable(T_grad, fullfile(out_dir,'grad_norm_history.csv'));

% ---------- (C) intial/final/ground truth
vx_init  = log.estimate.x_FIE_init(3,1:K).';
vx_final = log.estimate.x_FIE_final(3,1:K).';
vx_gt    = log.groundtruth.x(3,1:K).';
T_vx = table(t, vx_gt, vx_init, vx_final, ...
    'VariableNames', {'t','vx_gt','vx_init','vx_cal'});
writetable(T_vx, fullfile(out_dir,'traj_vx_gt_init_cal.csv'));

vz_init  = log.estimate.x_FIE_init(4,1:K).';
vz_final = log.estimate.x_FIE_final(4,1:K).';
vz_gt    = log.groundtruth.x(4,1:K).';
T_vz = table(t, vz_gt, vz_init, vz_final, ...
    'VariableNames', {'t','vz_gt','vz_init','vz_cal'});
writetable(T_vz, fullfile(out_dir,'traj_vz_gt_init_cal.csv'));

% ---------- (D) Frank–Wolfe trajectory at iteration 1-5 ----------
nSave = min(5, maxIter);
for k = 1:nSave
    if isempty(xFIE_iters{k}), continue; end
    vx_k = xFIE_iters{k}(3,1:K).';
    vz_k = xFIE_iters{k}(4,1:K).';
    T_k  = table(t, vx_k, vz_k, 'VariableNames', {'t','vx','vz'});
    fname = sprintf('traj_iter%02d_est_vx_vz.csv', k);
    writetable(T_k, fullfile(out_dir, fname));
end

% (E) theta_(delta l_shin) history
% stdHist (maxIter+1)×24，column is 24 link_error（）；
theta_shin_cm = 100 * stdHist(:,24);   % 0..T
T_shin = table(iters0, theta_shin_cm, ...
    'VariableNames', {'iter','theta_delta_l_shin_cm'});
writetable(T_shin, fullfile(out_dir,'theta_shin_cm_history.csv'));
writematrix(theta_shin_cm, fullfile(out_dir,'offset_singlecol_cm.csv'));

