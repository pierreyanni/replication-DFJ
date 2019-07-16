%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION: th = shock(th0,Pi,T)
%
% This function simulates a time series of length T for a discrete markov
% process with transition matrix Pi and initial state th0.
%
% INPUTS
%   th0:    Initial state of the process at date 1
%   Pi:     Transition matrix for the discrete markov process. Rows are
%           current state and columns next period state
%   T:      Number of time periods to be simulated
%
% OUTPUTS
%   th:     Column vector with simulated time series. Contains the index of
%           the current state of the process
%   
% Author: Jan Hannes Lang
% Date: 28.04.2010
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function th = shock(th0,Pi,T)

th    = ones(T,1);
th(1) = th0;
cum   = cumsum(Pi, 2);

for i = 2:T
    x     = find(rand <= cum(th(i-1),:));
    th(i) = x(1);
end
