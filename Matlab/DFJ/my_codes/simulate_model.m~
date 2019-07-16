% computes decision functions for both genders and all quintiles

clc
clear all

path_graphs = '../graphs/';
M_C = cell(1,5); % store decision functions
M_C_no_med = cell(1,5);
M_C_mean_med = cell(1,5);

params.g       = 0;     % gender: female
params.N_x     = 500;  % number of points on grid cash on hand (coh)
params.upper_x = 10500000; % upper bound on x grid adjusted to max wealth
params.N_z     = 9;  % number grid points medical expenses permanent shock
params.N_eps   = 8;  % number grid points medical expenses transitory shock

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

params.path_data = '../data/'; % path for data
params.path_output = '../output/'; % path for output

% CREATE DECISION FUNCTIONS


for quintile = 1:5

    quintile

    ind = quintile;

    M_C{ind} = DFJ_cons(quintile, params);
    M_C_no_med{ind} = DFJ_cons_no_med(quintile, params); 
    M_C_mean_med{ind} = DFJ_cons_mean_med(quintile, params);

end


save('../output/decision_fns.mat', 'M_C', 'M_C_no_med', 'M_C_mean_med')


% SIMULATIONS USING DECISION FUNCTIONS ABOVE

% simulations for cohort 1, all quintiles and female
N = 2000; % number of simulations
cohort = 1;

% load data
cohort1_female = readtable(strcat(params.path_data,'cohort1_female.csv'));

% FEMALE, COHORT 1 AND QUINTILE 3

figure(1)
quintile=3
[c_sim, x_sim, a_sim, s_sim] = simulation(quintile, cohort, N, params);
plot(70:100, median(a_sim, 'omitnan'), ...
cohort1_female{:,'age'}, cohort1_female{:,sprintf('q%d',quintile)},'--')

title('median assets level')
xlabel('age')
xlim([74, 84])

% FIGURE 5 IN PAPER (only for females of cohort 1)

figure(2)
for quintile = 1:5
    
    [c_sim, x_sim, a_sim, s_sim] = simulation(quintile, cohort, N, params);
    plot(70:100, median(a_sim, 'omitnan'), ...
    cohort1_female{:,'age'}, cohort1_female{:,sprintf('q%d',quintile)},'--')

    hold on
end

    title('median assets level')
    xlabel('age')
    xlim([74, 84])
hold off

% FIGURE 9 IN PAPER (only for females of cohort 1)

figure(3)

for quintile = 1:5
    
    [c_sim, x_sim, a_sim, s_sim] = simulation(quintile, cohort, N, params);
    plot(70:100, median(a_sim, 'omitnan'), '--')
    hold on
    [c_sim, x_sim, a_sim, s_sim] = simulation_no_med(quintile, cohort, ...
                                                    N, params);
    plot(70:100, median(a_sim, 'omitnan'))
    hold on
end

    title('median assets level')
    xlabel('age')
    xlim([74, 100])
hold off

% FIGURE 10 IN PAPER (only for females of cohort 1)

figure(4)

for quintile = 1:5
    
    [c_sim, x_sim, a_sim, s_sim] = simulation(quintile, cohort, N, params);
    plot(70:100, median(a_sim, 'omitnan'), '--')
    hold on
    [c_sim, x_sim, a_sim, s_sim] = simulation_mean_med(quintile, ...
                                                        cohort, N, params);
    plot(70:100, median(a_sim, 'omitnan'))
    hold on
end

    title('median assets level')
    xlabel('age')
    xlim([74, 100])
hold off

% save figures (with right size)

h = figure(1);
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize', ...
    [pos(3), pos(4)])
print(h, strcat(path_graphs, 'simul_q3.pdf'), '-dpdf', '-r0')

h = figure(2);
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize', ...
    [pos(3), pos(4)])
print(h, strcat(path_graphs, 'assets.pdf'), '-dpdf', '-r0')

h = figure(3);
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize', ...
    [pos(3), pos(4)])
print(h, strcat(path_graphs, 'no_med.pdf'), '-dpdf', '-r0')

% save figures (with right size)
h = figure(4);
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[...
    pos(3), pos(4)])
print(h, strcat(path_graphs, 'mean_med.pdf'), '-dpdf', '-r0')
