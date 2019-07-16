%%%%%%%%%%%%%%
% SIMULATION %
%%%%%%%%%%%%%%

% simulation for N agents given g, quintile, cohort 
% uses consumption functions computed w/ DFJ_cons_mean_med.m
% creates artificial data for c, x and s

function [c_sim, x_sim, a_sim, s_sim] = simulation_mean_med(quintile, ...
                                    cohort, N , params)

g           = params.g; % gender, female
c_ubar      = params.c_ubar; % consumption floor
r           = params.r;     % real interest rate
T           = params.T; % number of periods 
rho         = params.rho;    % rho med shock; zeta(t)=rho*zeta(t-1)+eps(t)
sig_z       = params.sig_z;   % sd persist med shock; eps ~ N(0,sig_zeta^2)
sig_eps     = params.sig_eps;  % sd transitory shock medical expenses
N_x         = params.N_x;  % number of points on grid cash on hand (coh)
upper_x     = params.upper_x; % upper bound on x grid adjusted to max wealth
N_h         = params.N_h;    % number of health states
N_z         = params.N_z;   % numb grid points med expenses permanent shock
N_eps       = params.N_eps; % numb grid pts med expenses transitory shock
path_data   = params.path_data; % path for data
path_output = params.path_output; % path for output

% open decision function
ind = g*5 + quintile; % M_C: g=0,1 and quintile=1,...,5
decision_fns = load(strcat(path_output,'decision_fns.mat'));
m_c = decision_fns.M_C_mean_med{ind};
           
% upload coefficient matrices for survival, health shock, income and
% medical expenses and convert them into vector indexed by age 70 to 100

I = 0.1 + 0.2*(quintile-1);     % middle range of quintile
m_agent     = [1 0 g I I^2;     % agent specific characteristics 
               1 1 g I I^2];    % for good and bad health (second row)
fileID      = fopen(strcat(path_data,'deathprof.out'),'r');
s_coef      = fscanf(fileID, '%f', [6 33]);     % survival logit coeffs
Xb_s        = (m_agent * s_coef(2:6, 3:33))';   % age 72 to 102 for probs
p_s           = sqrt(exp(Xb_s) ./ (1+exp(Xb_s))); % survival probabilities
                                                % sqrt() b/c 2 years prob
fileID      = fopen(strcat(path_data,'healthprof.out'),'r');
h_coef      = fscanf(fileID, '%f', [6 33]);     % health logit coefficients
Xb_h        = (m_agent * h_coef(2:6, 3:33))';   % age 72 to 102 for probs
p_h         = exp(Xb_h) ./ (1+exp(Xb_h));       % health transition prob

fileID      = fopen(strcat(path_data,'incprof.out'),'r');
inc_coef    = fscanf(fileID, '%f', [6 33]);         % income coefficients
Xb_inc      = (m_agent * inc_coef(2:6, 1:32))';     % using age 70 to 100
inc         = exp(Xb_inc(:,1));                     % income indep of h

fileID      = fopen(strcat(path_data,'medexprof_adj.out'),'r');
med_coef    = fscanf(fileID, '%f', [11 33]);      % medical expenses coeffs
Xb_med      = (m_agent * med_coef(2:6, 1:32))';   % average (age 70-100)
Xb_var_med  = (m_agent * med_coef(7:11, 1:32))';  % volatility (age 70-100) 

% markov chains med expenses shock
[Pi_z, eps, z_grid] = tauchen(N_z, 0, rho, sig_z); % zeta shock
[Pi_eps, eps, eps_grid] = tauchen(N_eps, 0, 0, sig_eps); % eps shock
med_grid   = kron(z_grid, ones(N_eps, 1)) + kron(ones(N_z,1), eps_grid);

% tax schedule
brackets    = [0, 6250, 40200, 68400, 93950, 148250, 284700, 1e10]; 
                % income brackets (upper bound 1e6)
tau         = [0.0765, 0.2616, 0.4119, 0.3499, 0.3834, 0.4360, 0.4761]; 
                % marginal rates
tax         = zeros(8, 1);
for i = 1:7
    tax(i+1)    =  tax(i) + (brackets(i+1)-brackets(i)) * tau(i);
end

% create grid on cash in hand x
lower_x     = c_ubar;
v_x         = linspace(sqrt(lower_x), sqrt(upper_x), N_x)'.^2; 
                        % tighter grid for smaller values
d           = (sqrt(upper_x) - sqrt(lower_x)) / (N_x-1); % dist b/w gridpts

% initial distributions
tab_init = readtable(strcat(path_data,'data_init.csv'));
tab_gI = tab_init((tab_init.g == g) & (tab_init.quintile == quintile) ...
                    & (tab_init.cohort == cohort), ...
                  {'h', 'a', 'med', 'inc', 'age'});

% random draws in inital distribution              
vec_n = randi(size(tab_gI,1), N,1);

% empty matrices for survival paths, consumption paths, cash on hand paths
s_sim = NaN(N, T);
c_sim = NaN(N, T);
x_sim = NaN(N, T);
a_sim = NaN(N, T);

% loop on simulations 
for i = 1:N

n = vec_n(i); % pick initial conditions for an agent g, I

a0 = tab_gI{n, 'a'}; % assets
inc0 = tab_gI{n, 'inc'}; % income
med0 = tab_gI{n, 'med'}; % med expenses
t0 = tab_gI{n, 'age'} - 69; % initial period (age 70: 1)
h0 = tab_gI{n, 'h'} + 1; % good health: 1, bad health: 2

% initial cash on hand x0:
y0          = r*a0  + inc0; % earnings next period
net_y0      = y0 - interp1(brackets, tax, y0); % earnings net of taxes
x0          = max(a0 + net_y0, c_ubar); % initial cash on hand

% simulating medical shock
z0_est = (log(med0) - Xb_med(t0, h0))/Xb_var_med(t0, h0); %estimates z0
[z0, n_z0] = min(abs(z_grid - z0_est)); % closest value in z grid

theta_z = simul(n_z0, Pi_z, T-t0); % path for persistent shock index
theta_eps = simul(1, Pi_eps, T-t0); % path for transitory shock index
v_z = z_grid(theta_z); % persistent shock
v_eps = eps_grid(theta_eps); % transitory shock
v_psi = v_z + v_eps; % combined shocks 

h = h0; % initial health
ind_z = n_z0; % initial persistent shock index
x = x0; % initial coh
x_sim(i,t0) = x; 
s_sim(i,t0) = 1; % initial survival state (1: alive)
a_sim(i,t0) = a0; % initial asset position
s = 1; % survival in first period



% simulate path for every draw
for t = t0:T-1
    
    if (s == 1) & (rand < p_s(t, h))  % s=1 if survival, s=NaN otherwise
        s = 1;
    else
        s = NaN; % dead next period
    end
    
    s_sim(i, t+1) = s;
     
    ind = (h-1)*N_z + ind_z; % index agent w/ health h and pers shock n_z
    v_c = m_c(:, ind, t); % decision fn for agent fn of h, n_z
    c = interpy1(v_x, v_c, x, d, lower_x); % consumption choice
    c_sim(i, t) = c;
    a_sim(i, t+1) = x - c; % assets at beginning of t+1 (and end of t)
    
    h       = 1 + (rand < p_h(t, h)); % h next period
    ind_z   = theta_z(t-t0+1); % z ind next period
    med     = mean(exp(Xb_med(t+1, h) + ...
              Xb_var_med(t+1, h).^(1/2) * med_grid)); %mean med next period    
    y       = r*(x-c)  + inc(t+1); % earnings next period
    net_y   = y - interp1(brackets, tax, y); % earnings net of taxes
    x       = max(x-c + net_y - med, c_ubar); % cash on hand next period
    x_sim(i, t+1) = x;
    a_sim(i, t+1) = x - c; % assets at beginning of t+1 (and end of t)
end

c_sim(i, t+1) = x;

end

% fast interpolation function (vectors x, y and point x0)
function y0 = interpy1(x, y, x0, d, lower_x) % interpolation
N = length(x);
ind = min(floor((sqrt(x0)-sqrt(lower_x))/d)+1, N-1);
y0 = y(ind) + (x0-x(ind)) .* (y(ind+1)-y(ind)) ./ (x(ind+1)-x(ind));

% simulation of markov chain
function theta = simul(theta0, Pi, T)

Pi_cum = cumsum(Pi, 2);
theta = [theta0; zeros(T-1, 1)];

for t = 1:T-1

    theta(t+1) = find(rand < Pi_cum(theta(t), :), 1, 'first');
end

