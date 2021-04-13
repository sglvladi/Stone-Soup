function z = evaluateAz(state, sensorpos, posidx)

% z = evaluateAz(state, sensorpos, posidx)

dpos = bsxfun(@minus, state(posidx,:), sensorpos);
z = atan2(dpos(1,:), dpos(2,:));
