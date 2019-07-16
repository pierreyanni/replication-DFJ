%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODEL WITH HEALTH SHOCK, %
%   SIMPLE MEDICAL SHOCK   %
%  AND CONSUMPTION FLOOR   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;

% PARAMETERS

% for all agents
nu          = 3.81; % curvature on period utility function
beta        = 0.97; % discount factor
c_ubar      = 2663; % consumption floor
age_max     = 100; % max age
age_min     = 70; % starting age
r           = 0.02; % real interest rate
N_grid      = 100; % number of points on grid
N_periods   = age_max - age_min + 1; % number of periods
rho         = 0.922; % rho medical shock; zeta(t) = rho*zeta(t-1)+eps(t)
sig_z       = sqrt(0.05); % sd medical shock; eps ~ N(0,sig_zeta^2)
N_h         = 2;    % number of health states
N_z         = 11;    % number of grid points medical expenses shock approx

% tax schedule
brackets    = [0, 6250, 40200, 68400, 93950, 148250, 284700, 1e6]; % income brackets (upper bound 1e6)
tau         = [0.0765, 0.2616, 0.4119, 0.3499, 0.3834, 0.4360, 0.4761]; % marginal rates
tax         = zeros(8, 1);
for i = 1:7
    tax(i+1)    =  tax(i) + (brackets(i+1)-brackets(i)) *  tau(i);
end

% agent specific
g           = 0;    % gender: 1 is male
I           = 0.5;  % income percentile (in the paper, quintiles: 0.2 to 1)
m_agent     = [1 0 g I I^2;     % agent specific characteristics 
               1 1 g I I^2];    % for good and bad health (second row)


% upload coefficient matrices for survival, health shock, income and
% medical expenses and convert them into vector indexed by age 70 to 100

fileID      = fopen('deathprof.out','r');
s_coef      = fscanf(fileID, '%f', [6 33]);     % survival logit coefficients
Xb_s        = (m_agent * s_coef(2:6, 3:33))';   % using age 72 to 102 for probs
s           = sqrt(exp(Xb_s) ./ (1+exp(Xb_s))); % survival probabilities
                                                % sqrt() because 2 years prob
fileID      = fopen('healthprof.out','r');
h_coef      = fscanf(fileID, '%f', [6 33]);     % health logit coefficients
Xb_h        = (m_agent * h_coef(2:6, 3:33))';   % using age 72 to 102 for probs
prob_h      = exp(Xb_h) ./ (1+exp(Xb_h));       % health transition probabilities

fileID      = fopen('incprof.out','r');
inc_coef    = fscanf(fileID, '%f', [6 33]);         % income coefficients
Xb_inc      = (m_agent * inc_coef(2:6, 1:31))';     % using age 70 to 100
inc         = exp(Xb_inc);                          % income

fileID      = fopen('medexprof_adj.out','r');
med_coef    = fscanf(fileID, '%f', [11 33]);        % average medical expenses coefficients
Xb_med      = (m_agent * med_coef(2:6, 1:31))';     % average (age 70-100)
Xb_var_med  = (m_agent * med_coef(7:11, 1:31))';    % volatility (age 70-100)
% med         = exp(Xb_med + 1/2 * Xb_var_med);      % medical expenses

% SOLVING MODEL

utility = @(c) c.^(1-nu) / (1-nu);  % period utility function
u_bar   = utility(c_ubar); % utility at consumption floor

% create grid on cash in hand x
lower_x     = 0;
upper_x     = 200000;
v_x         = linspace(sqrt(lower_x), sqrt(upper_x), N_grid).^2; % tighter grid for smaller values

% approximate shock on medical expenses

[Pi_z, eps, v_z] = tauchen(N_z, 0, rho, sig_z);
% Pi_z: transition matrix for z and v_z: vector of discretized shocks

% create matrices to store value functions, consumption and others
% index: periods in good health, periods in bad health

m_V_c       = zeros(N_h * N_z, N_grid);             % current value 
m_V_f       = zeros(N_h * N_z, N_grid);             % future value
m_V         = zeros(N_h * N_z, N_grid, N_periods);  % value function
m_c         = zeros(N_h * N_z, N_grid, N_periods);  % consumption choice
m_x_prime   = zeros(N_h * N_z, N_grid, N_periods);  % end-of-period cash-in-hand
year_prob_h = zeros(N_periods, N_h);                % prob of good health (for graph)
m_med       = zeros(N_periods, N_h * N_z);          % med expenses (graph)

options     = optimset('TolX', 1e-6);

% iterations on value function for good and bad health (0/1)

for period = N_periods:-1:1
    
    period
    
    med = exp(kron(Xb_med(period,:), ones(N_z, 1)) + kron(Xb_var_med(period,:), v_z));
    % dim (N_z, N_h)
    m_med(period, :) = [med(:, 1)', med(:, 2)'];  % med expenses in graph below
    
    m_V_f   = m_V_c;  % switch future and current value function 
                      %(necessary because non-stationary; see below)                  
                                          
    Pi_h    = [1 - prob_h(period,1) prob_h(period,1);
               1 - prob_h(period,2) prob_h(period,2)]^(1/2); 
               % transition matrix for health
           
    Pi      =  kron(Pi_h, Pi_z);  % transitions matrix all shocks
        
    year_prob_h(period,:) = Pi_h(:,2); % prob of bad health (graph below)
    
    for n_h = 1:N_h
        
        for n_z = 1:N_z

            index = N_z * (n_h-1) + n_z;
            
            y           = r * v_x + inc(period, n_h); % earnings
            net_cash    = v_x + y - interp1(brackets, tax, y) - med(n_z, n_h);
            ind_bar        = net_cash < c_ubar; % consumption floor index
            x_prime      = 0; % cash-in-hand end of period
            w_new       = u_bar + beta * s(period, n_h) ...
                           * interp1q(v_x, m_V_f', x_prime) ...
                           * Pi(index, :)';
            
            m_c(index, ind_bar, period)       = c_ubar;
            m_x_prime(index, ind_bar, period) = x_prime;
            m_V(index, ind_bar, period)       = w_new;
            m_V_c(index, ind_bar)             = w_new;           
                       
            parfor i = sum(ind_bar)+1:N_grid
                
                w               = @(c) -utility(c) - beta * s(period, n_h) ...
                                  * interp1q(v_x, m_V_f', net_cash(i) - c) ...
                                  * Pi(index,:)';
                [cons, w_new]   = fminbnd(w, 0, net_cash(i), options);

                m_c(index, i, period)       = cons;
                m_x_prime(index, i, period) = net_cash(i) - cons;
                m_V(index, i, period)       = -w_new;
                m_V_c(index, i)             = -w_new;
            end
            
        end     
    end    
end

% graphs
figure(1)

periods = [16 30];
N_per = size(periods,2);
txt = strings(N_h * N_per,1);
for h = 0:1
    for p = 1:size(periods, 2)
        index = h * N_per + p;
        txt(index) = sprintf('p=%d, h=%d', [periods(p), h]);
    end
end

ax1 = subplot(3, 3, 1);
for per =  periods
    plot(v_x, m_c(:, :, per));
    hold on;
end
plot(v_x, c_ubar*ones(N_grid,1));
hold off;
title('consumption')
xlabel('cash-in-hand')
legend(ax1, txt, 'location', 'best')

ax2 = subplot(3, 3, 4);
for per =  periods
    plot(v_x, m_x_prime(:, :, per));
    hold on;
end
hold off;
title('next period cash-in-hand')
xlabel('cash-in-hand')
legend(ax2, txt, 'location', 'best')

ax3 = subplot(3, 3, 7);
for per =  periods
    plot(v_x, m_V(:, :, per));
    hold on;
end
hold off;
title('value function')
xlabel('cash-in-hand')
legend(ax3, txt, 'location', 'best')

ax4 = subplot(3, 3, 2);
plot(1:N_periods, m_med(:, 1:N_z), '-', 1:N_periods, m_med(:, N_z+1:2*N_z), '--');
title('medical spendings')
xlabel('period')
legend(ax4, {'good health', 'bad health'}, 'location', 'best')

ax5 = subplot(3, 3, 5);
plot(1:N_periods, s(:,1), '-', 1:N_periods, s(:,2), '--');
title('survival probability')
xlabel('period')
legend(ax5, {'good health', 'bad health'}, 'location', 'best')

ax6 = subplot(3, 3, 8);
plot(1:N_periods, year_prob_h(:,1), '-', 1:N_periods, year_prob_h(:,2), '--');
title('prob of bad health next period')
xlabel('period')
legend(ax6, {'good health', 'bad health'}, 'location', 'best')

subplot(3, 3, 3);
plot(1:N_periods, inc(:,1), '-', 1:N_periods, inc(:,2), '--');
title('gross income (not from assets)')
xlabel('period')

figure(2)
for per =  periods
    plot(v_x, m_c(:, :, per));
    hold on;
end
hold off;
title('consumption')
xlabel('cash-in-hand')
legend(txt, 'location', 'best')