function plot_FIE(log)
% Figure 1 : base position and velocity GT vs FIE
figure(1); clf
t = log.estimate.t;

subplot(2,2,1)
plot(t , log.estimate.x_FIE(1,:), 'LineWidth',1.2); hold on
plot(t , log.groundtruth.x(1,:) , 'LineWidth',1.0);
legend('FIE','GT'); ylabel('x (m)')

subplot(2,2,2)
plot(t , log.estimate.x_FIE(2,:), 'LineWidth',1.2); hold on
plot(t , log.groundtruth.x(2,:) , 'LineWidth',1.0);
legend('FIE','GT'); ylabel('z (m)')

subplot(2,2,3)
plot(t , log.estimate.x_FIE(3,:), 'LineWidth',1.2); hold on
plot(t , log.groundtruth.x(3,:) , 'LineWidth',1.0);
legend('FIE','GT'); ylabel('v_x (m/s)')

subplot(2,2,4)
plot(t , log.estimate.x_FIE(4,:), 'LineWidth',1.2); hold on
plot(t , log.groundtruth.x(4,:) , 'LineWidth',1.0);
legend('FIE','GT'); ylabel('v_z (m/s)')

% Figure 2: absolute error for base position and velocity
figure(2); clf
subplot(2,2,1)
plot(t , abs(log.estimate.x_FIE(1,:) - log.groundtruth.x(1,:)));
ylabel('|x−x_{GT}| (m)'); title('X Pos Error')

subplot(2,2,2)
plot(t , abs(log.estimate.x_FIE(2,:) - log.groundtruth.x(2,:)));
ylabel('|z−z_{GT}| (m)'); title('Z Pos Error')

subplot(2,2,3)
plot(t , abs(log.estimate.x_FIE(3,:) - log.groundtruth.x(3,:)));
ylabel('|v_x−v_{x,GT}|');  title('X Vel Error')

subplot(2,2,4)
plot(t , abs(log.estimate.x_FIE(4,:) - log.groundtruth.x(4,:)));
ylabel('|v_z−v_{z,GT}|');  title('Z Vel Error')

% Figure 3: foot position GT vs FIE
figure(3); clf
subplot(2,2,1)
plot(t , log.estimate.x_FIE(5,:), 'LineWidth',1.2); hold on
plot(t , log.groundtruth.x(5,:) , 'LineWidth',1.0);
legend('FIE','GT'); ylabel('p_{lf,x} (m)')
title('Left Foot X')

subplot(2,2,2)
plot(t , log.estimate.x_FIE(6,:), 'LineWidth',1.2); hold on
plot(t , log.groundtruth.x(6,:) , 'LineWidth',1.0);
legend('FIE','GT'); ylabel('p_{lf,z} (m)')
title('Left Foot Z')

subplot(2,2,3)
plot(t , log.estimate.x_FIE(7,:), 'LineWidth',1.2); hold on
plot(t , log.groundtruth.x(7,:) , 'LineWidth',1.0);
legend('FIE','GT'); ylabel('p_{rf,x} (m)')
title('Right Foot X')

subplot(2,2,4)
plot(t , log.estimate.x_FIE(8,:), 'LineWidth',1.2); hold on
plot(t , log.groundtruth.x(8,:) , 'LineWidth',1.0);
legend('FIE','GT'); ylabel('p_{rf,z} (m)')
title('Right Foot Z')

% Figure 4 : absolute error for foot position
figure(4); clf
subplot(2,2,1)
plot(t , abs(log.estimate.x_FIE(5,:) - log.groundtruth.x(5,:)));
ylabel('|p_{lf,x}−GT|'); title('LF X Error')

subplot(2,2,2)
plot(t , abs(log.estimate.x_FIE(6,:) - log.groundtruth.x(6,:)));
ylabel('|p_{lf,z}−GT|'); title('LF Z Error')

subplot(2,2,3)
plot(t , abs(log.estimate.x_FIE(7,:) - log.groundtruth.x(7,:)));
ylabel('|p_{rf,x}−GT|'); title('RF X Error')

subplot(2,2,4)
plot(t , abs(log.estimate.x_FIE(8,:) - log.groundtruth.x(8,:)));
ylabel('|p_{rf,z}−GT|'); title('RF Z Error')
end