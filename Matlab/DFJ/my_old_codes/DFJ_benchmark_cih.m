%%%%%%%%%%%%%%%%%%%%%%%
% DFJ BENCHMARK MODEL %
%%%%%%%%%%%%%%%%%%%%%%%

% gss function performs golden section search

clc;
clear all;

% PARAMETERS

% for all agents
nu          = 3.81; % curvature on period utility function
beta        = 0.97; % discount factor
c_ubar      = 2663; % consumption floor
age_max     = 100; % max age
age_min     = 70; % starting age
T           = age_max-age_min+1; % number of periods 
r           = 0.02; % real interest rate
rho         = 0.922; % rho medical shock; zeta(t) = rho*zeta(t-1)+eps(t)
sig_z       = sqrt(0.05); % sd medical shock; eps ~ N(0,sig_zeta^2)
sig_eps     = sqrt(0.665); % sd transitory shock medical expenses
N_x         = 50; % number of points on grid cash-in-hand (cih)
N_h         = 2;    % number of health states
N_z         = 8;    % number of grid points medical expenses shock permanent
N_eps       = 8;    % number of grid points medical expenses shock transitory

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
p_h         = exp(Xb_h) ./ (1+exp(Xb_h));       % health transition probabilities

fileID      = fopen('incprof.out','r');
inc_coef    = fscanf(fileID, '%f', [6 33]);         % income coefficients
Xb_inc      = (m_agent * inc_coef(2:6, 1:31))';     % using age 70 to 100
inc         = exp(Xb_inc(:,1));                     % income indep of h

fileID      = fopen('medexprof_adj.out','r');
med_coef    = fscanf(fileID, '%f', [11 33]);        % average medical expenses coefficients
Xb_med      = (m_agent * med_coef(2:6, 1:31))';     % average (age 70-100)
Xb_var_med  = (m_agent * med_coef(7:11, 1:31))';    % volatility (age 70-100)

% SOLVING MODEL

utility = @(c) c.^(1-nu) / (1-nu);  % period utility function
u_bar   = utility(c_ubar); % utility at consumption floor

% create grid on cash in hand x
lower_x     = c_ubar;
upper_x     = 300000;
v_x         = linspace(sqrt(lower_x), sqrt(upper_x), N_x)'.^2; % tighter grid for smaller values
d           = (sqrt(upper_x) - sqrt(lower_x))/ (N_x-1);

% approximate shocks on medical expenses

[Pi_z, eps, v_z] = tauchen(N_z, 0, rho, sig_z);
% Pi_z: transition matrix for z and v_z: vector of discretized shocks
[Pi_eps, eps, v_eps] = tauchen(N_eps, 0, 0, sig_eps);
% Pi_eps: transition matrix for eps and v_eps: vector of discretized shocks

v_med   = kron(v_z, ones(N_eps, 1)) + repmat(v_eps, N_z, 1);
Pi_med  = kron(Pi_z, Pi_eps(1,:));
N_med   = N_z * N_eps;
% grid and transition matrix for medical expenses shock

% create matrices to store value functions, consumption and others
% index: periods in good health, periods in bad health

m_V_c       = zeros(N_x, N_h * N_z);        % current value 
m_V_f       = zeros(N_x, N_h * N_z);        % future value
m_V         = zeros(N_x, N_h * N_z, T);     % value function
m_c         = zeros(N_x, N_h * N_z, T);     % consumption choice
m_x_prime   = zeros(N_x, N_h * N_z, T);     % end-of-period cash-in-hand
p_bh        = zeros(T, N_h);                % prob of bad health (graph)
m_med       = zeros(T, N_h * N_z);          % med expenses (graph)

% consumption floor

i_bar = v_x < c_ubar;

% last period (T)

m_c(:,:,T)      = repmat(v_x, 1, N_h * N_z);
m_x_p(:,:,T)    = zeros(N_x, N_h * N_z);
m_V(:,:,T)      = repmat(utility(v_x), 1, N_h * N_z);;

% iterations on value function

for t = T-1:-1:1
    
    disp(sprintf('t: %d', t))
    
    m_V_f = m_V(:, :, t+1); % parallel computation does not work with m_V
    
    Pi_h        = [1-p_h(t,1) p_h(t,1); 1-p_h(t,2) p_h(t,2)]^(1/2); % Pi(h)
    p_bh(t,:)   = Pi_h(:,2)'; % prob of bad health (graph below)
    
    y           = r * v_x + inc(t+1); % earnings next period
    net_y       = y - interp1(brackets, tax, y); % earnings net of taxes
    
    for n_h = 1:N_h

        for n_z = sum(i_bar)+1:N_z
            
            ind = (n_h-1) * N_h + n_z;
            
            med     = exp(repmat(Xb_med(t+1,:), 1, N_med) ...
                          + kron(Xb_var_med(t+1,:), v_med'));
            prob_tr = kron(Pi_h(n_h,:), Pi_med(n_z,:));
            
            % when consumption floor is reached (wealth = 0)
            
            cih_bar = max(inc(t+1) - interp1(brackets, tax, inc(t+1)) - med,...
                      c_ubar); % cash-in-hand next period
            EV_bar  = prob_tr * interpy(v_x, m_V_f(:, ind), cih_bar', d, lower_x); 
            V_bar   = u_bar + beta * s(t, n_h) * EV_bar;
            
            m_V_c(1,ind)    = V_bar;
            
            m_c(1,ind,t)    = c_ubar;
            m_x_p(1,ind,t)  = 0;
            m_V(1,ind,t)    = V_bar;
            
        
            parfor n_x = 2:N_x
                
                cih = @(c) max(v_x(n_x) + net_y(n_x) - med - c, c_ubar);
                
                EV  = @(c) prob_tr * interpy(v_x, m_V_f(:, ind), ...
                           cih(c)', d, lower_x);
                       
                V   = @(c) utility(c) + beta * s(t, n_h) * EV(c);
                
                [cons, V_s] = gss(V, c_ubar, v_x(n_x), 1);
                
                m_V_c(n_x,ind)    = V_s;
                
                m_c(n_x,ind,t)      = cons;
                m_x_p(n_x,ind,t)    = v_x(n_x) - cons;
                m_V(n_x,ind,t)      = V_s;
                
            end
        end
    end
end
 

%%% functions %%%

% fast interpolation function
function y0 = interpy(x, y, x0, d, lower_x) % interpolation
N = length(x);
ind = min(floor((sqrt(x0)-sqrt(lower_x))/d)+1, N-1);
y0 = y(ind) + (x0-x(ind)) .* (y(ind+1)-y(ind)) ./ (x(ind+1)-x(ind));
end

% fast golden section search function; searches for a MAX
function [argmax_gss, max_gss] = gss(f, a, b, tol)

psi = (1+sqrt(5))/2;
c   = b - (b-a)/psi;
d   = a + (b-a)/psi;
f_c = f(c);
f_d = f(d);

while b - a > tol
    
    if f_c > f_d
        
        b   = d;
        d   = c;
        c   = b - (b-a)/psi;
        f_d = f_c;
        f_c = f(c);    
    else
        
        a   = c;
        c   = d;
        d   = a + (b-a)/psi;
        f_c = f_d;
        f_d = f(d);
    end   
end

argmax_gss  = (a+b)/2;
max_gss     = f(argmax_gss);
end 