clc; clear; close all;

%% Parameters AND Condi
L = 40; % Length of the domain
T = 200; % Total simulation time
r1 = 0.8; % Intrinsic growth rate for phytoplankton
r2 = 0.7; % Intrinsic growth rate for zooplankton
K1 = 1.0; % Carrying capacity for phytoplankton
K2 = 0.38; % Carrying capacity for zooplankton
beta1 = 0.8; % Predation rate of zooplankton on phytoplankton
beta2 = 0.7; % Conversion efficiency of ingested biomass to zooplankton
m = 0.5; % Half-saturation constant
D1 = 0.02; % Diffusion coefficient for phytoplankton
D2 = 0.1; % Diffusion coefficient for zooplankton
 
% Resolutions for RK2 and RK4
Nx = 100; % Number of spatial points for RK2 and RK4
Nt = 1000; % Number of time points for RK2 and RK4
dx = L / Nx; % Spatial step size for RK2 and RK4
dt = T / Nt; % Time step size for RK2 and RK4
x = (0:dx:L-dx)'; % Spatial grid for RK2 and RK4
 
%% Boundary Conditions (for simplicity, assuming Neumann boundary conditions)
P0 = @(x) 1/( exp(D1*sqrt(2)*(x-L/3)/2)); % Initial distribution of Phytoplankton
Z0 = @(x) 1/( exp(D2*sqrt(2)*(x-L/3)/2)); % Initial distribution of Zooplankton
gP0 = @(t) 0; % Left boundary condition for phytoplankton
gPL = @(t) 0; % Right boundary condition for phytoplankton
gZ0 = @(t) 0; % Left boundary condition for zooplankton
gZL = @(t) 0; % Right boundary condition for zooplankton
 
%% Initialize solution matrices
% RK2
P_RK2 = zeros(Nx, Nt); % Phytoplankton density using RK2
Z_RK2 = zeros(Nx, Nt); % Zooplankton density using RK2
P_RK2(:, 1) = P0(0);
Z_RK2(:, 1) = Z0(0);
 
% RK4
P_RK4 = zeros(Nx, Nt); % Phytoplankton density using RK4
Z_RK4 = zeros(Nx, Nt); % Zooplankton density using RK4
P_RK4(:, 1) = P0(0);
Z_RK4(:, 1) = Z0(0);
 
% RK5
P_RK5 = zeros(Nx, Nt); % Phytoplankton density using RK5
Z_RK5 = zeros(Nx, Nt); % Zooplankton density using RK5
P_RK5(:, 1) = P0(0);
Z_RK5(:, 1) = Z0(0);
 
%% Time-stepping using Runge-Kutta 2nd order method (RK2)
tic; % Start timing
for n = 1:Nt-1
    % Compute the RK2 coefficients
    [k1P, k1Z] = reaction_diffusion(P_RK2(:, n), Z_RK2(:, n), r1, r2, K1, K2, beta1, beta2, m, D1, D2, dx, 2, gP0, gPL, gZ0, gZL, n*dt);
    [k2P, k2Z] = reaction_diffusion(P_RK2(:, n) + dt * k1P, Z_RK2(:, n) + dt * k1Z, r1, r2, K1, K2, beta1, beta2, m, D1, D2, dx, 2, gP0, gPL, gZ0, gZL, n*dt + dt);
 
    % Update densities using Runge-Kutta 2nd order method
    P_RK2(:, n+1) = P_RK2(:, n) + dt / 2 * (k1P + k2P);
    Z_RK2(:, n+1) = Z_RK2(:, n) + dt / 2 * (k1Z + k2Z);
 
    % Applying boundary conditions
    P_RK2(1, n+1) = gP0(n*dt);
    P_RK2(end, n+1) = gPL(n*dt);
    Z_RK2(1, n+1) = gZ0(n*dt);
    Z_RK2(end, n+1) = gZL(n*dt);
end
time_RK2 = toc; % End timing and store time
 
%% Time-stepping using Runge-Kutta 4th order method (RK4)
tic; % Start timing
for n = 1:Nt-1
    % Compute the RK4 coefficients
    [k1P, k1Z] = reaction_diffusion(P_RK4(:, n), Z_RK4(:, n), r1, r2, K1, K2, beta1, beta2, m, D1, D2, dx, 4, gP0, gPL, gZ0, gZL, n*dt);
    [k2P, k2Z] = reaction_diffusion(P_RK4(:, n) + dt * k1P / 2, Z_RK4(:, n) + dt * k1Z / 2, r1, r2, K1, K2, beta1, beta2, m, D1, D2, dx, 4, gP0, gPL, gZ0, gZL, n*dt + dt/2);
    [k3P, k3Z] = reaction_diffusion(P_RK4(:, n) + dt * k2P / 2, Z_RK4(:, n) + dt * k2Z / 2, r1, r2, K1, K2, beta1, beta2, m, D1, D2, dx, 4, gP0, gPL, gZ0, gZL, n*dt + dt/2);
    [k4P, k4Z] = reaction_diffusion(P_RK4(:, n) + dt * k3P, Z_RK4(:, n) + dt * k3Z, r1, r2, K1, K2, beta1, beta2, m, D1, D2, dx, 4, gP0, gPL, gZ0, gZL, n*dt + dt);
 
    % Update densities using Runge-Kutta 4th order method
    P_RK4(:, n+1) = P_RK4(:, n) + dt / 6 * (k1P + 2*k2P + 2*k3P + k4P);
    Z_RK4(:, n+1) = Z_RK4(:, n) + dt / 6 * (k1Z + 2*k2Z + 2*k3Z + k4Z);
 
    % Applying boundary conditions
    P_RK4(1, n+1) = gP0(n * dt);
    P_RK4(end, n+1) = gPL(n * dt);
    Z_RK4(1, n+1) = gZ0(n * dt);
    Z_RK4(end, n+1) = gZL(n * dt);
end
time_RK4 = toc; % End timing and store time
 
%% Time-stepping using Runge-Kutta 5th order method (RK5)
tic; % Start timing
for n = 1:Nt-1
    [k1P, k1Z] = reaction_diffusion(P_RK5(:, n), Z_RK5(:, n), r1, r2, K1, K2, beta1, beta2, m, D1, D2, dx, 4, gP0, gPL, gZ0, gZL, n*dt);
    [k2P, k2Z] = reaction_diffusion(P_RK5(:, n) + dt * k1P / 2, Z_RK5(:, n) + dt * k1Z / 2, r1, r2, K1, K2, beta1, beta2, m, D1, D2, dx, 4, gP0, gPL, gZ0, gZL, n*dt + dt/2);
    [k3P, k3Z] = reaction_diffusion(P_RK5(:, n) + dt * (2 * k1P + (3 - sqrt(5)) * k2P)/10, Z_RK5(:, n) + dt * (2 * k1Z + (3 - sqrt(5)) * k2Z)/10, r1, r2, K1, K2, beta1, beta2, m, D1, D2, dx, 4, gP0, gPL, gZ0, gZL, n*dt + dt/4);
    [k4P, k4Z] = reaction_diffusion(P_RK5(:, n) + dt * ((k1P + k2P) / 4), Z_RK5(:, n) + dt * ((k1Z + k2Z) / 4), r1, r2, K1, K2, beta1, beta2, m, D1, D2, dx, 4, gP0, gPL, gZ0, gZL, n*dt + dt/2);
    [k5P, k5Z] = reaction_diffusion(P_RK5(:, n) + dt * ((1 - sqrt(5)) * k1P - 4 * k2P + (5 + 3 * sqrt(5)) * k3P + 8 * k4P)/20, Z_RK5(:, n) + dt * ((1 - sqrt(5)) * k1Z - 4 * k2Z + (5 + 3 * sqrt(5)) * k3Z + 8 * k4Z)/20, r1, r2, K1, K2, beta1, beta2, m, D1, D2, dx, 4, gP0, gPL, gZ0, gZL, n*dt + dt*3/4);
    [k6P, k6Z] = reaction_diffusion(P_RK5(:, n) + dt * (((sqrt(5) - 1) / 4 * k1P + (2 * sqrt(5) - 2) / 4 * k2P + (5 - sqrt(5)) / 4 * k3P - 8 / 4 * k4P + (10 - 2 * sqrt(5)) / 4 * k5P)), Z_RK5(:, n) + dt * (((sqrt(5) - 1) / 4 * k1Z + (2 * sqrt(5) - 2) / 4 * k2Z + (5 - sqrt(5)) / 4 * k3Z - 8 / 4 * k4Z + (10 - 2 * sqrt(5)) / 4 * k5Z)), r1, r2, K1, K2, beta1, beta2, m, D1, D2, dx, 4, gP0, gPL, gZ0, gZL, n*dt + dt);
 
    % Update densities using Runge-Kutta 5th order method
    P_RK5(:, n+1) = P_RK5(:, n) + dt / 12 * (k1P + 5 * k3P + 5 * k5P + k6P);
    Z_RK5(:, n+1) = Z_RK5(:, n) + dt / 12 * (k1Z + 5 * k3Z + 5 * k5Z + k6Z);
    
    % Applying boundary conditions
    P_RK5(1, n+1) = gP0(n*dt);
    P_RK5(end, n+1) = gPL(n*dt);
    Z_RK5(1, n+1) = gZ0(n*dt);
    Z_RK5(end, n+1) = gZL(n*dt);
end
time_RK5 = toc; % End timing and store time
 
%% Output the computation times
fprintf('Computation time using RK2: %.2f seconds\n', time_RK2);
fprintf('Computation time using RK4: %.2f seconds\n', time_RK4);
fprintf('Computation time using RK5: %.2f seconds\n', time_RK5);

% Compute relative errors
Relative_Error_RK2 = norm(abs(P_RK2(:,end) - P_RK5(:,end)), inf) / norm(P_RK5(:,end), inf)
Relative_Error_RK4 = norm(abs(P_RK4(:,end) - P_RK5(:,end)), inf) / norm(P_RK5(:,end), inf)

 
%% Results
 
% Plot comparison of final states RK4, RK5, and RK2
figure;
subplot(2, 1, 1);
plot(x, P_RK2(:, end), 'ro', x, P_RK4(:, end), 'b', x, P_RK5(:,end), 'g');
title('Phytoplankton Density');
legend('RK2', 'RK4', 'RK5');
xlabel('Space');
ylabel('Density');
 
subplot(2, 1, 2);
plot(x, Z_RK2(:, end), 'ro', x, Z_RK4(:, end), 'b', x, Z_RK5(:,end), 'g');
title('Zooplankton Density');
legend('RK2', 'RK4', 'RK5');
xlabel('Space');
ylabel('Density');
 
% Plot results for RK4
figure;
a1 = linspace(0, T, Nt);
[a4] = a1(101:900);
a2 = x(2:99);
a3 = P_RK4(2:99,101:900);
[aa, bb] = meshgrid(a4, a2);
mesh(aa,bb,a3);
zlim([0.382 0.385]);
xlabel('Time');
ylabel('Position');
title('Phytoplankton Density (RK4)');
 
 
function [dPdt, dZdt] = reaction_diffusion(P, Z, r1, r2, K1, K2, beta1, beta2, m, D1, D2, dx, order, gP0, gPL, gZ0, gZL, t)
    % Reaction terms
    fP = r1 * P .* (1 - P / K1) - beta1 * P .* Z ./ (m + P);
    fZ = r2 * Z .* (1 - Z / K2) + beta2 * P .* Z ./ (m + P);
 
    % Diffusion terms
    if order == 2
        % Second-order central difference
        d2Pdx2 = ([P(2:end); P(end)] - 2 * P + [P(1); P(1:end-1)]) / dx^2;
        d2Zdx2 = ([Z(2:end); Z(end)] - 2 * Z + [Z(1); Z(1:end-1)]) / dx^2;
    elseif order == 4
        % Fourth-order central difference
        d2Pdx2 = (-[P(3:end); P(end); P(end)] + 16*[P(2:end); P(end)] - 30*P + 16*[P(1); P(1:end-1)] - [P(1); P(1); P(1:end-2)]) / (12*dx^2);
        d2Zdx2 = (-[Z(3:end); Z(end); Z(end)] + 16*[Z(2:end); Z(end)] - 30*Z + 16*[Z(1); Z(1:end-1)] - [Z(1); Z(1); Z(1:end-2)]) / (12*dx^2);
        
        % Apply boundary conditions
        d2Pdx2(1) = (P(2) - 2 * P(1) + gP0(t)) / dx^2;
        d2Pdx2(2) = (P(3) - 2 * P(2) + P(1)) / dx^2;
        d2Pdx2(end-1) = (P(end) - 2 * P(end-1) + P(end-2)) / dx^2;
        d2Pdx2(end) = (gPL(t) - 2 * P(end) + P(end-1)) / dx^2;
 
        d2Zdx2(1) = (Z(2) - 2 * Z(1) + gZ0(t)) / dx^2;
        d2Zdx2(2) = (Z(3) - 2 * Z(2) + Z(1)) / dx^2;
        d2Zdx2(end-1) = (Z(end) - 2 * Z(end-1) + Z(end-2)) / dx^2;
        d2Zdx2(end) = (gZL(t) - 2 * Z(end) + Z(end-1)) / dx^2;
    end
 
    % Total derivatives
    dPdt = fP + D1 * d2Pdx2;
    dZdt = fZ + D2 * d2Zdx2;
end
 
