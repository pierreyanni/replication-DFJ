clc
clear all


N = 1000; % number of simulations
g = 0; % gender
cohort = 1;

params.nu      = 3.81; % curvature on period utility function
params.beta    = 0.97; % discount factor
params.c_ubar  = 2663; % consumption floor
age_min        = 70;   % starting age
age_max        = 101;  % max age
params.T       = age_max-age_min+1; % number of periods 
params.r       = 0.02;     % interest rate
params.rho     = 0.922;   % rho medical shock; zeta(t)=rho*zeta(t-1)+eps(t)
params.sig_z   = sqrt(0.05); % sd persistent med shock; eps~N(0,sig_zeta^2)
params.sig_eps = sqrt(0.665); % sd transitory shock medical expenses
params.N_h     = 2;  % number of health states
params.N_x     = 100;  % number of points on grid cash on hand (coh)
params.N_z     = 9;  % number grid points medical expenses permanent shock
params.N_eps   = 8;  % number grid points medical expenses transitory shock
params.path_data = '../data/'; % path for data
params.path_output = '../output/'; % path for data



data = readtable('../data/cohort1.csv');

figure(1)
for quintile = 1:5
    
    [c_sim, x_sim, s_sim] = simulation(g, quintile, cohort, N, params);
    %subplot(5,1,quintile)
    plot(70:101, median((x_sim-c_sim).*s_sim, 'omitnan'))
    %  , ... data{:,'age'}, data{:,sprintf('q%d',quintile)}, '--')

    hold on
end

    title('assets')
    xlabel('age')
    xlim([74, 102])
hold off
