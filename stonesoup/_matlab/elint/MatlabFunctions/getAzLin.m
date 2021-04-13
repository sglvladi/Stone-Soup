function H = getAzLin(state, sensorpos, posidx)

% H = getAzLin(state, sensorpos, posidx)
%
% (Azimuth measured clockwise from 12o'clock)

statedim = size(state,1);
H = zeros(1, statedim);
dpos = bsxfun(@minus, state(posidx,:), sensorpos);
flatrangesq = sum(dpos(1:2).^2);
H(1,posidx) = [dpos(2) -dpos(1) zeros(1,numel(posidx)-2)]/flatrangesq;
