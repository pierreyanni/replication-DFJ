clc
clear all


N = 100; % number of simulations
g = 0; % gender
cohort = 1;

data = readtable('cohort1.csv');

figure(1)
for q = 1:5 %quintile
    
    [c_sim, x_sim, s_sim] = simulation(g, q, cohort, N);
    [c_sim_nomed, x_sim_nomed, s_sim_nomed] = simulation_nomed(g, q, cohort, N);
    
    plot(70:100, median(x_sim-c_sim, 'omitnan'), ...
        70:100, median(x_sim_nomed-c_sim_nomed, 'omitnan'));
        

    hold on
end

    title('assets')
    xlabel('age')
    xlim([74, 100])
hold off
