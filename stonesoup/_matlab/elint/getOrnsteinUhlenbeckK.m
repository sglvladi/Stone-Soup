function K = getOrnsteinUhlenbeckK(q_metres, stationary_speed)

% K = getOrnsteinUhlenbeckK(q_metres, stationarySpeed)

% Stationary velocity s.d. of Ornstein-Uhlenbeck is sqrt(q/(2*K)) so set K
% to make prior velocity the stationary distribution
K = q_metres/(2*stationary_speed^2);
