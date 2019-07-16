%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DFJ BENCHMARK MODEL ONLY CONS %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% used in simulations

function m_c = benchmark_decision(quintile, params)


% PARAMETERS

% for all agents
g           = params.g;         % gender female
nu          = params.nu;        % curvature on period utility function
beta        = params.beta;      % discount factor
c_ubar      = params.c_ubar;    % consumption floor
r           = params.r;         % real interest rate
T           = params.T;         % number of periods 
rho         = params.rho;     % rho med shock; zeta(t)=rho*zeta(t-1)+eps(t)
sig_z       = params.sig_z;   % sd persist med shock; eps ~ N(0,sig_zeta^2)
sig_eps     = params.sig_eps;   % sd transitory shock medical expenses
N_x         = params.N_x;     % number of points on grid cash-in-hand (cih)
upper_x     = params.upper_x; % upper bound on grid;
N_h         = params.N_h;    % number of health states
N_z         = params.N_z;   % numb grid points med expenses permanent shock
N_eps       = params.N_eps; % numb grid pts med expenses transitory shock
path_data   = params.path_data; % path for data

% gender: 1 is male, income quintile 1 to 5
I = quintile*0.2-0.1; % middle percentile for each quintile (0.1-0.9)
m_agent     = [1 0 g I I^2;     % agent specific characteristics 
               1 1 g I I^2];    % for good and bad health (second row)

tol         = 1;            % tol on golden section search algorithm

% tax schedule (brackets and marginal rates tau
brackets    = [0, 6250, 40200, 68400, 93950, 148250, 284700, 1e10]; 
tau         = [0.0765, 0.2616, 0.4119, 0.3499, 0.3834, 0.4360, 0.4761]; 
tax         = zeros(8, 1);
for i = 1:7
    tax(i+1)    =  tax(i) + (brackets(i+1)-brackets(i)) * tau(i);
end

% upload coefficient matrices for survival, health shock, income and
% medical expenses and convert them into vector indexed by age 70 to 100

fileID      = fopen(strcat(path_data,'deathprof.out'),'r');
s_coef      = fscanf(fileID, '%f', [6 33]);       % survival logit coefs
Xb_s        = (m_agent * s_coef(2:6, 3:32))';     % age 72 to 102 for probs
s           = sqrt(exp(Xb_s) ./ (1+exp(Xb_s)));   % survival probabilities
                                                  % sqrt() b/c 2 years prob
fileID      = fopen(strcat(path_data,'healthprof.out'),'r');
h_coef      = fscanf(fileID, '%f', [6 33]);       % health logit coefs
Xb_h        = (m_agent * h_coef(2:6, 3:32))';     % age 72 to 102 for probs
p_h         = exp(Xb_h) ./ (1+exp(Xb_h));         % health transition probs

fileID      = fopen(strcat(path_data,'incprof.out'),'r');
inc_coef    = fscanf(fileID, '%f', [6 33]);       % income coefs
Xb_inc      = (m_agent * inc_coef(2:6, 1:31))';   % using age 70 to 100
inc         = exp(Xb_inc(:,1));                   % income indep of h

fileID      = fopen(strcat(path_data,'medexprof_adj.out'),'r');
med_coef    = fscanf(fileID, '%f', [11 33]);      % medical expenses coefs
Xb_med      = (m_agent * med_coef(2:6, 1:31))';   % average (age 70-100)
Xb_var_med  = (m_agent * med_coef(7:11, 1:31))';  % volatility (age 70-100)

% SOLVING MODEL
% grid on cash in hand x
lower_x     = c_ubar;
v_x         = linspace(sqrt(lower_x), sqrt(upper_x), N_x)'.^2; 
                % tighter grid for smaller values
d           = (sqrt(upper_x) - sqrt(lower_x)) / (N_x-1); 
                % distance between gridpoints

% approximate shocks on medical expenses
[Pi_z, eps, v_z] = tauchen(N_z, 0, rho, sig_z);
                    % Pi_z: transition matrix: v_z: vector of shocks
[Pi_eps, eps, v_eps] = tauchen(N_eps, 0, 0, sig_eps);
                       % Pi_eps: transition matrix; v_eps: vector of shocks
% grid on combined (persistent and transitory) med shocks
v_med   = kron(v_z, ones(N_eps, 1)) + kron(ones(N_z,1), v_eps);
% transition matrix for combined med shocks 
Pi_med  = kron(Pi_z, Pi_eps(1,:));
N_med   = N_z * N_eps;

% create matrices to store value functions, consumption and others
% index: periods in good health, periods in bad health

m_V_f       = zeros(N_x, N_h * N_z);     % future value
m_V         = zeros(N_x, N_h * N_z, T);  % value function
m_c         = zeros(N_x, N_h * N_z, T);  % consumption choice

% LAST PERIOD (T)
utility = @(c) c.^(1-nu) / (1-nu);  % period utility function
m_c(:,:,T)      = repmat(v_x, 1, N_h * N_z);
m_V(:,:,T)      = repmat(utility(v_x), 1, N_h * N_z);

% ITERATIONS ON VALUE FUNCTION
for t = T-1:-1:1
    
    disp(sprintf('t: %d', t))
    m_V_f       = m_V(:, :, t+1); % parallel comput does not work with m_V
    Pi_h        = [1-p_h(t,1) p_h(t,1); 1-p_h(t,2) p_h(t,2)]^(1/2); % Pi(h)
    prob_tr     = kron(Pi_h, Pi_med);
    med_        = exp(repmat(Xb_med(t+1,:), 1, N_med) ...
                          + kron(Xb_var_med(t+1,:).^(1/2), v_med')); 
    med         = kron([mean(med_(1:N_med)), mean(med_(N_med+1:2*N_med))],...
                    ones(1,N_med)); % mean med expenses by health status
    
    parfor n_x = 1:N_x
        
        v_cons = zeros(N_h * N_z, 1);
        v_V = zeros(N_h * N_z, 1);
                
        for n_h = 1:N_h

            for n_z = 1:N_z
            
                ind = (n_h-1) * N_z + n_z;
          
            if n_x == 1 % consumption floor is reached

                V_s = objective(c_ubar, r, v_x(n_x), inc(t+1), ...
                      brackets, tax, med, c_ubar, prob_tr(ind,:), v_x, ... 
                      m_V_f, d, lower_x, nu, beta, s(t, n_h));
                cons = c_ubar;
            else
                
                f = @(c) objective(c, r, v_x(n_x), inc(t+1), ...
                         brackets, tax, med, c_ubar, prob_tr(ind,:), v_x, ... 
                         m_V_f, d, lower_x, nu, beta, s(t, n_h));

                [cons, V_s] = gss(f, c_ubar, v_x(n_x), tol);
            end

                v_cons(ind, 1)      = cons;
                v_V(ind,1)          = V_s;                 
            end            
        end
        
        m_c(n_x,:,t)      = v_cons;
        m_V(n_x,:,t)      = v_V;        
    end
end


% FUNCTIONS

% objective function
function V = objective(c, r, x, inc, brackets, tax, med, c_ubar, ...
                       prob_tr, v_x, m_V_f, d, lower_x, nu, beta, s)
                   
y           = r * (x - c)  + inc; % earnings next period
net_y       = y - interp1(brackets, tax, y); % earnings net of taxes
cih         = max(x - c + net_y - med, c_ubar)'; % cih next period
EV          = prob_tr * interpy(v_x, m_V_f, cih, d, lower_x); %E[value(t+1)]
V           = c.^(1-nu) / (1-nu) + beta * s * EV; % value(t)

% fast interpolation function (updated to work with matrix m_V
function y0 = interpy(x, m_V, x0, d, lower_x) % interpolation

[M, N] = size(m_V);
ind1 = min(floor((sqrt(x0)-sqrt(lower_x))/d)+1, M-1); % index for x
ind2 = kron([0:N-1]', ones(length(x0)/N,1)*M); % ind to adj for col of m_V
ind = ind1 + ind2; % ind to use linear index in m_V

y0 = m_V(ind) + (x0-x(ind1))./(x(ind1+1)-x(ind1)) .* (m_V(ind+1)-m_V(ind));


% golden section search function; searches for a MAX
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