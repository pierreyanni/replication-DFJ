
N_grid =100;

lower_x     = 0;
upper_x     = 300000;
v_x         = linspace(sqrt(lower_x), sqrt(upper_x), N_grid).^2; % tighter grid for smaller values

d = (upper_x - lower_x)/(N_grid + 1);

N = N_grid * 2 * T * 8^2;

N_test = 10000

tic
for x0 = c_ubar + rand(1,N_test)*(upper_x - c_ubar)
    w = @(c) -utility(c) - beta * s(t, n_h) * Pi(index,:) ...
                    * interp_py(v_x, m_V_f, x0 - c);               
    [cons, w_new]   = fminbnd(w, 0, x0, options);
end
toc

c=2663;

tic
for x0 = c_ubar + rand(1,N_test)*(upper_x - c_ubar)
    interp_py(v_x, m_V_f, x0 - c);               
    
end
toc


% fast interpolation function
function v_V = interp_py(v_x, m_V, x0)
% interpolation
N = length(v_x);
if x0 >= v_x(N)
    v_V = m_V(:, N) + (x0-v_x(N)) * (m_V(:,N)-m_V(:,N-1)) / (v_x(N)-v_x(N-1));
else
    ind = find(v_x <= x0, 1, 'last');
    v_V = m_V(:,ind) + (x0-v_x(ind)) * (m_V(:,ind+1)-m_V(:,ind)) / (v_x(ind+1)- v_x(ind));
end
end

