function visDetTrans = getVisibilityDetectionTransition(...
    revealRates, hideRates, measRates, dt)

% visDetTrans = getVisibilityDetectionTransition(...
%    revealRates, hideRates, measRates, dt)
%
% Get visDetTrans(v0, v1, d1, j) = probability that we transition from old
% visibility state v0-1 to new visibility state v1-1 and detection state
% d1-1 in time interval dt for sensor j
%
% (so, for example, visDetTrans(1, 2, 1, 1) is the probability that an
% originally hidden target becomes visible but is not detected for sensor 1

nsensors = numel(revealRates);
visDetTrans = zeros(2, 4, nsensors);
for i=1:nsensors
    % (v,d) = {(0,0),(1,0),(0,1),(1,1)}
    ratematrix = [...
        -revealRates(i) revealRates(i) 0 0;...
        hideRates(i) -(hideRates(i)+measRates(i)) 0 measRates(i);...
        0 0 -revealRates(i) revealRates(i);...
        0 0 hideRates(i) -hideRates(i)];
    probmatrix = expm(ratematrix*dt);
    visDetTrans(:,:,i) = probmatrix(1:2,:);
end

visDetTrans = reshape(visDetTrans, [2 2 2 nsensors]);
