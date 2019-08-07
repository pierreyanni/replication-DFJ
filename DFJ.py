#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import quantecon as qe
from scipy import linalg
from quantecon.optimize.scalar_maximization import brent_max
from numba import njit

# fast code to solve DFJ


class common_params:
    """This class loads and computes parameters common to all agents"""

    def __init__(self, N_z, N_eps, N_x, upper_x,
                 c_ubar=2663, T=31, N_h=2, r=0.02,
                 nu=3.81, beta=0.97,
                 rho=0.922, sig_z=np.sqrt(0.05), sig_eps=np.sqrt(0.665)):

        self.N_x, self.N_z, self.N_eps, self.N_h = N_x, N_z, N_eps, N_h
        self.upper_x, self.c_ubar = upper_x, c_ubar
        self.T, self.r, self.beta = T, r, beta

        # utility function
        @njit
        def u(x):
            return x**(1-nu)/(1-nu)
        self.u = u

        # markov chain for medical expenses shocks
        z = qe.rouwenhorst(N_z, 0, sig_z, rho)
        eps = qe.rouwenhorst(N_eps, 0, sig_eps, 0)
        self.z, self.eps = z, eps
        self.grid_med = (np.kron(z.state_values, np.ones(N_eps))
                         + np.kron(np.ones(N_z), eps.state_values))
        self.Pi_med = np.kron(z.P, eps.P[1, :])

        # grid on cash on hand (more points for lower values)
        self.grid_x = np.linspace(np.sqrt(c_ubar), np.sqrt(upper_x), N_x)**2

        # tax function
        brackets = np.array([0, 6250, 40200, 68400, 93950,
                             148250, 284700, 2e7])
        tau = np.array([0.0765, 0.2616, 0.4119, 0.3499,
                        0.3834, 0.4360, 0.4761])
        tax = np.zeros(8)
        for i in range(7):
            tax[i+1] = tax[i] + (brackets[i+1]-brackets[i]) * tau[i]
        self.tax = tax
        self.brackets = brackets


class indiv_params:
    """This class loads and computes parameters for
    agents of gender g and income percentile q"""

    def __init__(self, g, inc_perc):
        # agent's characteristics for good (0) and bad health (1)
        m = np.array([[1, 0, g, inc_perc, inc_perc**2],
                      [1, 1, g, inc_perc, inc_perc**2]])
        # survival prob (sqrt() b/c probs for 2 years) by age and health status
        cols = ['constant', 'health', 'male', 'inc_perc', 'inc_perc_sq']
        df = pd.read_csv('raw_data/deathprof.out', delimiter=r"\s+",
                         header=None, index_col=0, names=cols)
        self.pr_s = np.sqrt(np.exp(df.loc[72:, :]@m.T)
                            / (1+np.exp(df.loc[72:, :]@m.T))).values

        # (2 years) prob of bad health by age and health status
        df = pd.read_csv('raw_data/healthprof.out', delimiter=r"\s+",
                         header=None, index_col=0, names=cols)
        _ = np.exp(df.loc[72:, :]@m.T).values
        self.pr_h = _ / (1 + _)

        # income by age and health status
        df = pd.read_csv('raw_data/incprof.out', delimiter=r"\s+", header=None,
                         index_col=0, names=cols)
        # same inc for good and bad health
        self.inc = np.exp(df.loc[:100, :]@m[0, :].T).values

        # mean and variance of medical expenses by age and health status

        cols_var = [x + '_var' for x in cols]
        df = pd.read_csv('raw_data/medexprof_adj.out', delimiter=r"\s+",
                         header=None, index_col=0, names=cols+cols_var)
        mean_med = (df.loc[:100, cols]@m.T).values
        var_med = (df.loc[:100, cols_var]@m.T).values
        self.med = [mean_med, var_med]


@njit
def interp(v_x, v_y, x0):
    """interpolation and extrapolation (using last 2 values)"""
    i = np.fmin(np.searchsorted(v_x, x0, 'left'), len(v_x)-1)
    return v_y[i-1] + (x0 - v_x[i-1])/(v_x[i] - v_x[i-1]) * (v_y[i] - v_y[i-1])


@njit
def interp_mat(v_x, mat_y, v_x0):
    """interpolation and extrapolation; nth value of v_x0 corresponds to
    nth col of mat_y"""
    v_y0 = np.empty_like(v_x0)
    for col in range(len(v_y0)):
        i = np.fmin(np.searchsorted(v_x, v_x0[col], 'left'), len(v_x)-1)
        v_y0[col] = (mat_y[i-1, col]
                     + (v_x0[col] - v_x[i-1])/(v_x[i] - v_x[i-1])
                     * (mat_y[i, col] - mat_y[i-1, col]))
    return v_y0


def solve_model(cp, i_par):
    """iterates backward on value function"""
    # load parameters
    T, r, u, beta = cp.T, cp.r, cp.u, cp.beta
    N_x, N_h, N_z, N_eps = cp.N_x, cp.N_h, cp.N_z, cp.N_eps
    brackets, tax, c_ubar = cp.brackets, cp.tax, cp.c_ubar
    grid_x, Pi_med, grid_med = cp.grid_x, cp.Pi_med, cp.grid_med
    inc, pr_h, med, pr_s = i_par.inc, i_par.pr_h, i_par.med, i_par.pr_s
    N_med = Pi_med.shape[1]

    # prepare matrices for results
    m_c = np.zeros((T, N_x, N_h*N_z))
    m_V = np.empty((T, N_x, N_h*N_z))
    # last period
    m_c[T-1, :, :] = np.column_stack([grid_x] * N_h*N_z)
    m_V[T-1, :, :] = np.column_stack([u(grid_x)] * N_h*N_z)

    @njit
    def objective(c, x, t, ind, med_ex, pr_tr, s_h, v_next):
        """objective function in period t"""
        tot_inc_f = r * (x-c) + inc[t+1]
        net_y = tot_inc_f - interp(brackets, tax, tot_inc_f)
        coh = np.fmax(x - c + net_y - med_ex, c_ubar*np.ones_like(med_ex))
        EV = np.dot(pr_tr[ind, :], interp_mat(grid_x, v_next, coh))
        return u(c) + beta * s_h * EV

    # other periods
    for t in reversed(range(T-1)):
        if t % 5 == 0:
            print(f'period {t}')
        # expand val fn next period by # of transitory shocks
        v_next = np.repeat(m_V[t+1, :, :], N_eps, axis=1)
        # pr_h given for 2 periods:
        Pi_h = linalg.sqrtm(np.array([[1-pr_h[t+1, 0], pr_h[t+1, 0]],
                                      [1-pr_h[t+1, 1], pr_h[t+1, 1]]]))
        pr_tr = np.kron(Pi_h, Pi_med)
        med_ex = np.exp(np.kron(med[0][t+1, :], np.ones(N_med))
                        + np.kron(np.sqrt(med[1][t+1, :]), grid_med))
        v_cons = np.zeros((N_x, N_h*N_z))
        v_new = np.empty((N_x, N_h*N_z))

        for n_x in range(N_x):
            xi = grid_x[n_x]
            for n_h in range(N_h):
                for n_z in range(N_z):
                    ind = n_h * N_z + n_z
                    if n_x == 0:
                        val = objective(c_ubar, xi, t, ind,
                                        med_ex, pr_tr, pr_s[t+1, n_h], v_next)
                        cons = c_ubar
                    else:
                        cons, val, _ = brent_max(
                            objective, c_ubar, xi,
                            args=(xi, t, ind, med_ex, pr_tr,
                                  pr_s[t+1, n_h], v_next),
                            xtol=1)
                    v_cons[n_x, ind] = cons
                    v_new[n_x, ind] = val

        m_c[t, :, :], m_V[t, :, :] = v_cons, v_new

    return m_c, m_V


def simul(g, quintile, cohort, data, N, cp):
    """simulates the model using initial asset position, med_expenses and
    income for N randomly drawn people from given g, quintile and cohort;
    returns survival and asset position for N people"""

    inc_perc = 0.2 * quintile - 0.1  # income percentile
    ip = indiv_params(g, inc_perc)
    m_c, _ = solve_model(cp, ip)
    # load parameters
    z, grid_z, eps = cp.z, cp.z.state_values, cp.eps  # pers and trans shocks
    T, r, brackets, tax = cp.T, cp.r, cp.brackets, cp.tax
    c_ubar, N_z, grid_x = cp.c_ubar, cp.N_z, cp.grid_x
    m_med, v_med, income = ip.med[0], ip.med[1], ip.inc
    pr_h, pr_s = ip.pr_h, ip.pr_s
    # create initial matrices
    m_a = np.empty((N, T)) * np.nan  # asset position
    m_s = np.zeros((N, T))  # present in data (1/0)

    def rand_indiv(data):
        """initial conditions and med shock for given individual"""
        mask = ((data.g == g) & (data.quintile == quintile)
                & (data.cohort == cohort))
        id = np.random.choice(data[mask].index)
        age0, a0, h0, inc0, med0 = data.loc[id, ['age', 'a', 'h',
                                                 'inc', 'med']]
        t0, h0 = int(age0 - 70), int(h0)  # initial period; int for indices
        # estimate initial state of persistent shock
        m_med0, v_med0 = m_med[t0, h0], v_med[t0, h0]  # mean, var med ex
        zeta_est = (np.log(med0) - m_med0) / np.sqrt(v_med0)
        n_z0 = np.argmin(np.abs(zeta_est - grid_z))
        zeta0 = grid_z[n_z0]  # initial persistent shock in grid
        # simulate index for zeta and medical shock
        t_len = T - t0  # simulation length
        zeta_index = z.simulate_indices(t_len, init=n_z0, random_state=12)
        med_shock = (z.simulate(t_len, init=zeta0, random_state=12)
                     + eps.simulate(t_len, init=0))
        return t0, h0, a0, inc0, med0, zeta_index, med_shock

    @njit
    def coh(a, inc, med):
        """computes cash-on-hand"""
        tot_inc_f = r * a + inc
        net_y = tot_inc_f - interp(brackets, tax, tot_inc_f)
        coh = np.fmax(a + net_y - med, c_ubar)
        return coh

    for n in range(N):
        # initial values for loop
        t0, h, a, inc, med, zeta_index, med_shock = rand_indiv(data)
        s, n_z = 1, zeta_index[0]  # initial survival state and persist shock
        m_a[n, t0], m_s[n, t0] = a, 1
        for t in range(t0, T-1):
            x = coh(a, inc, med)
            ind = h * N_z + n_z
            c = interp(grid_x, m_c[t, :, ind], x)  # consumption
            Pi_h = linalg.sqrtm(np.array([[1-pr_h[t+1, 0], pr_h[t+1, 0]],
                                          [1-pr_h[t+1, 1], pr_h[t+1, 1]]]))
            # next period's variables
            h = int(np.random.rand() < Pi_h[h, 1])  # new draw for h
            s = np.where(s == 1, int(np.random.rand() < pr_s[t, h]), 0)
            a = x - c  # next period's asset position
            inc = income[t+1]
            med = np.exp(m_med[t+1, h]
                         + np.sqrt(v_med[t+1, h]) * med_shock[t+1-t0])
            n_z = zeta_index[t+1-t0]
            # save results
            m_a[n, t+1] = a
            m_s[n, t+1] = s

    return m_a, m_s
