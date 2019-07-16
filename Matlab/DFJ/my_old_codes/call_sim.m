clc
clear all


N = 5000; % number of simulations
g = 0; % gender
cohort = 1;

data = readtable('cohort1.csv');


for quintile = 1:5
    
    [c_sim, x_sim, s_sim] = simulation(g, quintile, cohort, N, params);

    figure(quintile)
    subplot(311)
    plot(70:100, median(c_sim.*s_sim, 'omitnan'))
    title('consumption')
    xlabel('age')
    xlim([74, 84])
    subplot(312)
    plot(70:100, median(x_sim.*s_sim, 'omitnan'))
    title('cash on hand')
    xlabel('age')
    xlim([74, 84])
    subplot(313)
    plot(70:100, median((x_sim - c_sim).*s_sim, 'omitnan'), ...
         data{:,'age'}, data{:,sprintf('q%d',quintile)}, '--')
    title('assets level')
    xlabel('age')
    xlim([74, 84])
end

