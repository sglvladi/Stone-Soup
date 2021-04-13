function z = evaluateRange(state, sensorpos, posidx)

% z = evaluateRange(state, sensorpos, posidx)

dpos = bsxfun(@minus, state(posidx,:), sensorpos);
z = sqrt(sum(dpos.^2,1));
