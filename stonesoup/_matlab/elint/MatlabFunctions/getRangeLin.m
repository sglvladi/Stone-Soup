function H = getRangeLin(state, sensorpos, posidx)

% H = getRangeLin(state, sensorpos, posidx)

statedim = size(state,1);
H = zeros(1, statedim);
dpos = bsxfun(@minus, state(posidx,:), sensorpos);
range = sqrt(sum(dpos.^2,1));
H(1,posidx) = dpos(:)' ./ range;
