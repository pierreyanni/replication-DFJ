% Solving a basic asset decumulation problem
clear all;
clc;

% parameters
nu = 2;
r = 0;
y = 0;
beta = 1;
T = 3;
N = 100;
a_init = 1000;

% create survival probabilities vector
s = linspace(1, 0, T);

% create grid on assets
grid_a = linspace(0, a_init, N);

% create matrices to store value functions and consumption
mV = zeros(N, T+1);
mC = zeros(N, T);
 
for t = T:-1:1
    for i = 1:N
        a = grid_a(i);
        f = @(a_prime)utility(a_prime, a, beta, r, y, mV, grid_a, nu, t);
        [aStar, V] = fminbnd(f, 0, a);
        mV(i, t) = -V;
        mC(i, t) = a + (r*a+y) - aStar;
        mAStar(i, t) = aStar;
    end
end

mV
mC
mAStar