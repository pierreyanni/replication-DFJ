% utility function

function neg_V = utility(a_prime, a, beta, r, y, mat_V, grid_a, nu, t)
    c =  a + (r*a+y) - a_prime;
    u = c.^(1-nu) / (1-nu);
    [m, ind] = min(abs(grid_a - a_prime));
    V_prime = mat_V(ind, t+1);
    neg_V = - (u + beta*V_prime);
end