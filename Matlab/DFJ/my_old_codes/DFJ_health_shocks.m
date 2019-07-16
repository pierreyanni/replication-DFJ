%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODEL WITH HEALTH SHOCK  %
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
               1 1 g I I^2];    % for good and bad health (second column)
N_health    = 2;    % number of health states

% upload coefficient matrices for survival, health shock, income and
% medical expenses and convert them into vector indexed by age 70 to 102

fileID  = fopen('deathprof.out','r');
s_coef  = fscanf(fileID, '%f', [6 33]);     % survival logit coefficients
Xb_s    = (m_agent * s_coef(2:6, 3:33))';   % using age 72 to 102 for probs
s       = sqrt(exp(Xb_s) ./ (1+exp(Xb_s))); % survival probabilities
                                            % sqrt() because 2 years prob

fileID  = fopen('healthprof.out','r');
h_coef  = fscanf(fileID, '%f', [6 33]);     % health logit coefficients
Xb_h    = (m_agent * h_coef(2:6, 3:33))';   % using age 72 to 102 for probs
prob_h  = exp(Xb_h) ./ (1+exp(Xb_h));       % health transition probabilities

fileID      = fopen('incprof.out','r');
inc_coef    = fscanf(fileID, '%f', [6 33]);         % income coefficients
Xb_inc      = (m_agent * inc_coef(2:6, 1:31))';     % using age 70 to 100
inc         = exp(Xb_inc);                          % income

fileID      = fopen('medexprof_adj.out','r');
med_coef    = fscanf(fileID, '%f', [11 33]);        % average medical expenses coefficients
Xb_med      = (m_agent * med_coef(2:6, 1:31))';     % constant term (age 70-100)
Xb_var_med  = (m_agent * med_coef(7:11, 1:31))';    % constant term (age 70-100)
med         = exp(Xb_med + 1/2 * Xb_var_med);       % medical expenses

% SOLVING MODEL

utility = @(x) x.^(1-nu) / (1-nu);  % period utility function

% create grid on cash in hand x
lower_x     = c_ubar;
upper_x     = 200000;
v_x         = linspace(sqrt(lower_x), sqrt(upper_x), N_grid).^2; % tighter grid for smaller values

% create matrix to store value functions and consumption
m_V         = zeros(N_periods, N_grid, N_health);   % value function
m_V_c       = zeros(N_grid, N_health);
m_V_f       = zeros(N_grid, N_health);
m_c         = zeros(N_periods, N_grid, N_health);   % consumption choice
m_x_prime   = zeros(N_periods, N_grid, N_health);   % future cash-in-hand

% initial iteration (final value)
m_V_c                   = utility(v_x)' * ones(1,2);
m_c(N_periods, :, 1)    = v_x;
m_c(N_periods, :, 2)    = v_x;

% iterations on value function for good and bad health (0/1)

for k = 1:N_periods-1
    
    period = N_periods - k
    
    m_V_f    = m_V_c;  % switch future and current value function 
                       %(necessary because non-stationary; see below
    trans_prob_h = [1 - prob_h(period,1) prob_h(period,1);
                    1 - prob_h(period,2) prob_h(period,2)]^(1/2);
    
    for h = 1:N_health
        
        for i = 1:N_grid
            
            income_prime            = @(c) r*(v_x(i)-c) + inc(period+1, h);
            x_prime                 = @(c) v_x(i) - c  + income_prime(c) ...
                                        - interp1(brackets, tax, income_prime(c), 'linear', 'extrap') ...
                                        - med(period+1, h);
            x_prime_ubar            = @(c) max(x_prime(c), c_ubar);

            w                       = @(c) -utility(c) - beta * s(period) ...
                                        * interp1(v_x, m_V_f, x_prime_ubar(c), 'linear', 'extrap') ...
                                        * trans_prob_h(h,:)';

            [c, w_new]              = fminbnd(w, c_ubar, v_x(i));

            m_c(period, i, h)       = c;
            m_x_prime(period, i, h) = x_prime_ubar(c);
            m_V(period, i, h)       = -w_new;

            m_V_c(i,h)              = -w_new;
            
        end
        
    end
    
end

% graphs of output tbc



% simulation of policy functions (x_prime and c)

x_start     = upper_x/2; %starting cash at hand
sim_c       = zeros(N_periods,1);
sim_x       = [x_start; zeros(N_periods,1)];

figure(2)
for h = 1:2

    for period = 1:N_periods
        sim_c(period)       = interp1(v_x, m_c(period, :, h), sim_x(period), 'linear', 'extrap');
        sim_x(period+1)     = interp1(v_x, m_x_prime(period, :, h), sim_x(period), 'linear', 'extrap');
    end
    h=1
    subplot(2, 3, 3*(h-1)+1)
    plot(1:size(sim_c), sim_c)
    title('consumption')
    xlabel('period')
    subplot(2, 3, 3*(h-1)+2)
    plot(1:size(sim_x), sim_x)
    title('x prime')
    xlabel('period')
    subplot(2, 3, 3*(h-1)+3)
    plot(1:size(med,1), med(:,h))
    title('medical spending')
    xlabel('period')
    
end