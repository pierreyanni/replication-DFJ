% computes decision functions for both genders and all quintiles

clc
clear all

M_C = cell(1,10); 

params.nu      = 3.81; % curvature on period utility function
params.beta    = 0.97; % discount factor
params.c_ubar  = 2663; % consumption floor
age_min        = 70;   % starting age
age_max        = 100;  % max age
params.T       = age_max-age_min+1; % number of periods 
params.r       = 0.02;     % interest rate
params.rho     = 0.922;   % rho medical shock; zeta(t)=rho*zeta(t-1)+eps(t)
params.sig_z   = sqrt(0.05); % sd persistent med shock; eps~N(0,sig_zeta^2)
params.sig_eps = sqrt(0.665); % sd transitory shock medical expenses
params.N_h     = 2;  % number of health states
params.N_x     = 500;  % number of points on grid cash on hand (coh)
params.upper_x = 10500000; % upper bound on x grid adjusted to max wealth
params.N_z     = 9;  % number grid points medical expenses permanent shock
params.N_eps   = 8;  % number grid points medical expenses transitory shock
params.path_data = '../data/'; % path for data

for g = 0:1
    
    g
    
    for quintile = 1:5
        
        quintile
        
        ind = g*5 + quintile;

        M_C{ind} = DFJ_cons(g, quintile, params);
        
    end
end

save('../output/decision_fns.mat', 'M_C')
