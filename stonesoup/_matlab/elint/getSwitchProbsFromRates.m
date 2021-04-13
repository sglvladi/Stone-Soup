function probmatrix = getSwitchProbsFromRates(rate12, rate21, dt)

% M(i,j) = probability of transitioning from model i to model j
ratematrix = [-rate12 rate12; rate21 -rate21];
probmatrix = expm(ratematrix*dt);
