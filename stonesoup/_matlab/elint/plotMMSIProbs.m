function plotMMSIProbs(trackdata, trackid)

% Plot probabilities of MMSIS assigned to track <trackid>

lineidx = find(trackdata.id==trackid);
[uqmmsis, logProbs] = getMMSIProbs(trackdata, lineidx);

colours = 'rbk';
hold on
for i = 1:numel(uqmmsis)
    c = colours(mod(i-1, numel(colours))+1);
    plot(trackdata.time(lineidx), exp(logProbs(:,i)), [c 'x-']);
end
ylim([0 1])

leg = uqmmsis;
for i = 1:numel(leg)
    if numel(leg{i})==0
        leg{i} = 'None';
    end
end
legend(leg, 'Location', 'SouthWest');
