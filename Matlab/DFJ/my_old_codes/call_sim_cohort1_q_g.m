% simulations for cohort 1, all quintiles and genders

clc
clear all

N = 1000; % number of simulations
cohort = 1;

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
params.N_x     = 100;  % number of points on grid cash on hand (coh)
params.upper_x = 10500000; % upper bound on x grid adjusted to max wealth
params.N_z     = 9;  % number grid points medical expenses permanent shock
params.N_eps   = 8;  % number grid points medical expenses transitory shock
params.path_data = '../data/'; % path for data
params.path_output = '../output/'; % path for data

data = readtable(strcat(params.path_data,'cohort1.csv'));
% median assets position in data
M_C_sim = cell(1, 10); % store sim cons
M_X_sim = cell(1, 10); % store sim coh next period
M_A_sim = cell(1, 10); % store assets beginning of period
M_S_sim = cell(1, 10); % store survival (1: survival, NaN: dead)

for g = 0:1
    
    g
    
    for quintile = 1:5
        
        quintile
        
        ind = g*5 + quintile;        
        [c_sim, x_sim, a_sim, s_sim] = simulation(g, quintile, cohort, N, params);
        M_C_sim{ind} = c_sim;
        M_X_sim{ind} = x_sim;
        M_A_sim{ind} = a_sim;
        M_S_sim{ind} = s_sim;        
    end
end


% FIGURE 5 IN PAPER (only for cohort 1)

figure(1)
for quintile = 1:5
    
    [c_sim, x_sim, a_sim, s_sim] = simulation(g, quintile, cohort, N, params);
    plot(70:100, median(a_sim, 'omitnan'), ...
    data{:,'age'}, data{:,sprintf('q%d',quintile)}, '--')

    hold on
end

    title('assets')
    xlabel('age')
    xlim([74, 100])
hold off

%%%% tbc %%%%

    
%{
    figure(11)
    subplot(311)
    plot(70:100, median(M_C_sim{ind}.*M_S_sim{ind}, 'omitnan'))
    title('consumption')
    xlabel('age')
    xlim([74, 100])
    subplot(312)
    plot(70:100, median(M_X_sim{ind}.*M_S_sim{ind}, 'omitnan'))
    title('cash on hand')
    xlabel('age')
    xlim([74, 100])
    subplot(313)
    plot(70:100, median((M_X_sim{ind} - M_C_sim{ind}).*M_S_sim{ind}, 'omitnan'), ...
         data{:,'age'}, data{:,sprintf('q%d',quintile)}, '--')
    title('assets level')
    xlabel('age')
    xlim([74, 100])

%}