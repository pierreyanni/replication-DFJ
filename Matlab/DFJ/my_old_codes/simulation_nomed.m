%%%%%%%%%%%%%%
% SIMULATION %
%%%%%%%%%%%%%%

% model without medical expenses

function [c_sim, x_sim, s_sim] = simulation_nomed(g, quintile, cohort, N)


c_ubar      = 2663; % consumption floor
age_max     = 100;  % max age
age_min     = 70;   % starting age
r           = 0.02;     % real interest rate
T           = age_max-age_min+1; % number of periods 
rho         = 0.922;    % rho medical shock; zeta(t) = rho*zeta(t-1)+eps(t)
sig_z       = sqrt(0.05);   % sd persistent medical shock; eps ~ N(0,sig_zeta^2)
sig_eps     = sqrt(0.665);  % sd transitory shock medical expenses
N_x         = 1000;  % number of points on grid cash-in-hand (cih)
N_h         = 2;    % number of health states
N_z         = 8;   % number of grid points medical expenses shock persistent
N_eps       = 8;    % number of grid points medical expenses shock transitory


I = 0.1 + 0.2*(quintile-1); % middle range of quintile

% open decision function
ind = g*5 + quintile; % M_C: g=0,1 and quintile=1,...,5
decision_fns = load('decision_fns.mat');
m_c = decision_fns.M_C{ind};
           
% upload coefficient matrices for survival, health shock, income and
% medical expenses and convert them into vector indexed by age 70 to 100

m_agent     = [1 0 g I I^2;     % agent specific characteristics 
               1 1 g I I^2];    % for good and bad health (second row)
fileID      = fopen('deathprof.out','r');
s_coef      = fscanf(fileID, '%f', [6 33]);     % survival logit coefficients
Xb_s        = (m_agent * s_coef(2:6, 3:33))';   % using age 72 to 102 for probs
p_s           = sqrt(exp(Xb_s) ./ (1+exp(Xb_s))); % survival probabilities
                                                % sqrt() because 2 years prob
fileID      = fopen('healthprof.out','r');
h_coef      = fscanf(fileID, '%f', [6 33]);     % health logit coefficients
Xb_h        = (m_agent * h_coef(2:6, 3:33))';   % using age 72 to 102 for probs
p_h         = exp(Xb_h) ./ (1+exp(Xb_h));       % health transition probabilities

fileID      = fopen('incprof.out','r');
inc_coef    = fscanf(fileID, '%f', [6 33]);         % income coefficients
Xb_inc      = (m_agent * inc_coef(2:6, 1:31))';     % using age 70 to 100
inc         = exp(Xb_inc(:,1));                     % income indep of h

% markov chains med expenses shock
[Pi_z, eps, z_grid] = tauchen(N_z, 0, rho, sig_z); % zeta shock
[Pi_eps, eps, eps_grid] = tauchen(N_eps, 0, 0, sig_eps); % eps shock

% tax schedule
brackets    = [0, 6250, 40200, 68400, 93950, 148250, 284700, 1e10]; % income brackets (upper bound 1e6)
tau         = [0.0765, 0.2616, 0.4119, 0.3499, 0.3834, 0.4360, 0.4761]; % marginal rates
tax         = zeros(8, 1);
for i = 1:7
    tax(i+1)    =  tax(i) + (brackets(i+1)-brackets(i)) * tau(i);
end

% create grid on cash in hand x
lower_x     = c_ubar;
upper_x     = 10500000;
v_x         = linspace(sqrt(lower_x), sqrt(upper_x), N_x)'.^2; % tighter grid for smaller values
d           = (sqrt(upper_x) - sqrt(lower_x)) / (N_x-1); % distance between gridpoints

% initial distributions

tab_init = readtable('data_init.csv');
tab_gI = tab_init((tab_init.g == g) & (tab_init.quintile == quintile) ...
                    & (tab_init.cohort == cohort), ...
                  {'h', 'a', 'med', 'inc', 'age'});

% loop on simulations              
vec_n = randi(size(tab_gI,1), N,1);
% simulate path for h, survival and medical expenses (and income)
s_sim = NaN(N, T);
c_sim = NaN(N, T);
x_sim = NaN(N, T);

for i = 1:N

n = vec_n(i);

a0 = tab_gI{n, 'a'};
inc0 = tab_gI{n, 'inc'};
med0 = 0;
t0 = tab_gI{n, 'age'} - 70;
h0 = tab_gI{n, 'h'} + 1; % good health: 1, bad health: 2

% initial cash in hand x0:
y0          = r*a0  + inc0; % earnings next period
net_y0      = y0 - interp1(brackets, tax, y0); % earnings net of taxes
x0          = max(a0 + net_y0 - med0, c_ubar);

% simulating medical shock
z0_est = (log(med0) - Xb_med(t0, h0))/Xb_var_med(t0, h0); %estimate z0
[z0, n_z0] = min(abs(z_grid - z0_est)); % closest value in z grid

theta_z = simul(n_z0, Pi_z, T-t0);
theta_eps = simul(1, Pi_eps, T-t0);
v_z = z_grid(theta_z);
v_eps = eps_grid(theta_eps);
v_psi = v_z + v_eps;

h = h0;
n_z = n_z0;
x = x0;
x_sim(i,t0) = x;
s_sim(i,t0) = 1;
s = 1; % survival in first period

for t = t0:T-1
    
    if (s == 1) & (rand < p_s(t+1, h))  % s=1 if survival, s=0 otherwise
        s = 1;
    else
        s = NaN;
    end
    
    s_sim(i, t+1) = s;
     
    ind = (h-1)*N_z + n_z;
    v_c = m_c(:, ind, t);
    c = interpy1(v_x, v_c, x, d, lower_x);
    c_sim(i, t) = c;
    
    h = 1 + (rand < p_h(t, h)); % h next period
    z = v_z(t-t0+1); % z next period
    med = 0; % med next period      
    y       = r*(x-c)  + inc(t+1); % earnings next period
    net_y   = y - interp1(brackets, tax, y); % earnings net of taxes
    x       = max(x-c + net_y - med, c_ubar); % cash in hand next period
    x_sim(i, t+1) = x;
end

c_sim(i, t+1) = x;

end

% fast interpolation function (vectors x, y and point x0)
function y0 = interpy1(x, y, x0, d, lower_x) % interpolation
N = length(x);
ind = min(floor((sqrt(x0)-sqrt(lower_x))/d)+1, N-1);
y0 = y(ind) + (x0-x(ind)) .* (y(ind+1)-y(ind)) ./ (x(ind+1)-x(ind));

function theta = simul(theta0, Pi, T)

Pi_cum = cumsum(Pi, 2);
theta = [theta0; zeros(T-1, 1)];

for t = 1:T-1

    theta(t+1) = find(rand < Pi_cum(theta(t), :), 1, 'first');
end

