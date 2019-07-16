% checking markov chain approximation of stochastic process
% zeta(t) = rho * zeta(t-1) + eps(t), where eps(t) ~ N(0,sig_eps^2)

rho     = .922
sig_eps = sqrt(0.05)
T       = 1000000
trial_T = round(0.2*T);
zeta    = zeros(T, 1);
eps     = randn(T-1, 1) * sig_eps;

disp(sprintf('theoretical mean = %f and theoretical var = %f', 0, sig_eps^2/(1-rho^2)))


% simulation
for t = 2:T
    zeta(t) = rho* zeta(t-1) + eps(t-1);
end

disp(sprintf('mean = %f and var = %f', mean(zeta(trial_T:T)), var(zeta(trial_T:T))))


% Markov chain

N       = 15;
[s, Pi] = mytauchen(0, rho, sig_eps, N);


% simulation
th      = shock((N+1)/2, Pi, T);

zeta_m  = s(th);

disp(sprintf('mean_sim = %f and var_sim = %f', mean(zeta_m(trial_T:T)), var(zeta_m(trial_T:T))))


% stationary distribution of markov chain
v = Pi^10000;
v = v(1,:)';

mean_stat   = s'*v;
var_stat    = s'.^2 * v - mean_stat^2;

disp(sprintf('mean_stat = %f and var_stat = %f', mean_stat, var_stat));



% simulation adda
disp("Adda's code")

[prob,eps,z]=tauchen(N,0,.922,sig_eps);

th      = shock((N+1)/2, prob, T);

zeta_m  = z(th);

disp(sprintf('mean_sim = %f and var_sim = %f', mean(zeta_m(trial_T:T)), var(zeta_m(trial_T:T))))


% stationary distribution of markov chain
v = prob^10000;
v = v(1,:)';

mean_stat   = z'*v;
var_stat    = z'.^2 * v - mean_stat^2;

disp(sprintf('mean_stat = %f and var_stat = %f', mean_stat, var_stat));




