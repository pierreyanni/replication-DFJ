clc

% parameters: t (1 to 31), health (1/2 for healthy/sick), 
%             persistent shocks (1 to N_z)

t   = [25 25];
n_h = [1 1];
n_z = [1 9];
       
ind1 = (n_h(1)-1) * N_z + n_z(1);
ind2 = (n_h(2)-1) * N_z + n_z(2);
       

text    = [sprintf('t: %d, health: %d, pers shock: %d', t(1), n_h(1), n_z(1));
           sprintf('t: %d, health: %d, pers shock: %d', t(2), n_h(2), n_z(2))];      
           
figure(3)
ax1 = subplot(3, 1, 1);
plot(v_x, m_c(:, ind1, t(1)), '-', v_x, m_c(:, ind2, t(2)), '--', ...
      v_x, v_x, '.', v_x, c_ubar*ones(N_x,1), '--')
title('consumption')
xlabel('cash-in-hand')
xlim([0,upper_x])
legend(text(1,:), text(2,:), 'location', 'best')

ax2 = subplot(3, 1, 2);
plot(v_x, m_V(:, ind1, t(1)), '-', v_x, m_V(:, ind2, t(2)), '--')
title('value function')
xlabel('cash-in-hand')
legend(text(1,:), text(2,:), 'location', 'best')

subplot(3, 1, 3);
plot(1:T-1, next_med(:, ind1), '-', 1:T-1, next_med(:, ind2), '--');
title('average med expenses next period')
xlim([0 31])
xlabel('t')
legend(text(1,:), text(2,:), 'location', 'best')



figure(4)

ax3 = subplot(3, 1, 1);
plot(1:T-1, s(:,1), '-', 1:T-1, s(:,2), '--');
title('survival probability')
xlabel('t')
xlim([0 31])
legend(ax3, {'good health', 'bad health'}, 'location', 'best')

ax4 = subplot(3, 1, 2);
plot(1:T, p_bh(:,1), '-', 1:T, p_bh(:,2), '--');
title('prob of bad health next t')
xlabel('t')
xlim([0 31])
legend(ax4, {'good health', 'bad health'}, 'location', 'best')

subplot(3, 1, 3);
plot(1:T, inc(:));
title('gross income (not from assets)')
xlim([0 31])
xlabel('t')

