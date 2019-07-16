%%%%%%%%%%%%%%%%%%%%%%%%%%
% DETERMINISTIC MODEL    %
% WITH CONSUMPTION FLOOR %
%%%%%%%%%%%%%%%%%%%%%%%%%%

% rem: survival probabilities should be converted from 2 years into 1 year

clc;
clear all;

% PARAMETERS

% for all agents
nu          = 3.81; % curvature on period utility function
beta        = 0.97; % discount factor
c_ubar      = 2663; % consumption floor
age_max     = 102; % max age
age_min     = 70; % starting age
r           = 0.02; % real interest rate
N_grid      = 200; % number of points on grid
N_periods   = age_max - age_min + 1; % number of periods

% tax schedule
brackets    = [0, 6250, 40200, 68400, 93950, 148250, 284700, 1e6]; % income brackets (upper bound 1e6)
tau         = [0.0765, 0.2616, 0.4119, 0.3499, 0.3834, 0.4360, 0.4761]; % marginal rates
tax         = zeros(8, 1);
for i = 1:7
    tax(i+1)    =  tax(i) + (brackets(i+1)-brackets(i)) *  tau(i);
end

% agent specific
g   = 0;    % gender: 1 is male
I   = 0.5;  % income percentile (in the paper, quintiles: 0.2 to 1)
h   = 0;    % health: 0 is good; stochastic in non-deterministic model

% upload coefficient matrices for survival, income and medical expenses and
% convert them into vector indexed by age 70 to 102
fileID  = fopen('deathprof.out','r');
s_coef  = fscanf(fileID, '%f', [6 33]);     % survival logit coefficients
Xb_s    = s_coef(2:6,:)' * [1 h g I I^2]';
s       = exp(Xb_s) ./ (1+exp(Xb_s));       % survival probabilities

fileID      = fopen('incprof.out','r');
inc_coef    = fscanf(fileID, '%f', [6 33]);         % income coefficients
Xb_inc      = inc_coef(2:6,:)' * [1 h g I I^2]'; 
inc         = exp(Xb_inc);                          % income

fileID      = fopen('medexprof_adj.out','r');
med_coef    = fscanf(fileID, '%f', [11 33]);        % average medical expenses coefficients
Xb_med      = med_coef(2:6,:)' * [1 h g I I^2 ]';   % constant term
Xb_var_med  = med_coef(7:11,:)' * [1 h g I I^2 ]';  % constant term
med         = exp(Xb_med + 1/2 * Xb_var_med);       % medical expenses

% SOLVING MODEL

utility = @(x) x.^(1-nu) / (1-nu); % period utility function

% create grid on cash in hand x
lower_x     = c_ubar;
upper_x     = 2e5;
v_x         = linspace(sqrt(lower_x), sqrt(upper_x), N_grid).^2; % tighter grid for smaller values

% create matrix to store value functions and consumption
m_V = zeros(2, N_grid);
m_c = zeros(N_periods, N_grid);
m_x_prime = zeros(N_periods, N_grid);

% initial iteration (final value)
m_V(1,:)            = utility(v_x);
m_c(N_periods,:)    = v_x;
period              = age_max - age_min; % start at second period from the end

% iterations on value function
while period > 0
    
    m_V(2,:)    = m_V(1,:);  % switch future and current value function 
                             %(necessary because non-stationary; see below)
    
    for i = 1:N_grid
        
        income_prime        = @(c) r*(v_x(i)-c) + inc(period+1);
        x_prime             = @(c) v_x(i) - c  + income_prime(c) ...
                                 - interp1(brackets, tax, income_prime(c), 'linear', 'extrap') ...
                                 - med(period+1);
        x_prime_ubar        = @(c) max(x_prime(c), c_ubar);
        
        w                   = @(c) -utility(c) - beta * s(period)...
                                        * interp1(v_x, m_V(2,:), x_prime_ubar(c), 'linear', 'extrap');
        
        [c, w_new]          = fminbnd(w, c_ubar, v_x(i));
        
        m_c(period, i)      = max(c,c_ubar);
        m_x_prime(period,i) = x_prime_ubar(c);

        m_V(1,i)            = -w_new;
    end
    
    period      = period - 1
    
end


% simulation of policy functions (x_prime and c)

x_start     = 100000; %starting cash at hand
sim_c       = zeros(N_periods,1);
sim_x       = [x_start; zeros(N_periods,1)];

for period = 1:N_periods
    sim_c(period)       = interp1(v_x, m_c(period,:), sim_x(period), 'linear', 'extrap');
    sim_x(period+1)     = interp1(v_x, m_x_prime(period,:), sim_x(period), 'linear', 'extrap');
end

subplot(2,2,1)
plot(1:size(sim_c), sim_c)
title('consumption')
xlabel('period')
subplot(2,2,2)
plot(1:size(sim_x), sim_x)
title('x prime')
xlabel('period')
subplot(2,2,3)
plot(1:size(med), med)
title('medical spending')
xlabel('period')