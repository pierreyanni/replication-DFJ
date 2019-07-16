function y0 = interpy(x, y, x0, d, lower_x)

% interpolation
N = length(x);
ind = min(floor(sqrt(x0)/d)+1, N-1);
y0 = y(ind) + (x0-x(ind)) .* (y(ind+1)-y(ind)) ./ (x(ind+1)-x(ind));
end





    