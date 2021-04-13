function drawTrackSwitchProbs(trackdata, colourtoplot, colours)

if ~exist('colours','var') || isempty(colours)
    colours = {'m','c','b','r','g'};
end

plotid = true;

hold on
[uqids,~,idx] = uniquewithcounts(trackdata.id);
for i=1:numel(uqids)
    thisid = uqids(i);
    thistimes = trackdata.time(idx{i})';
    thiscol = colours{mod(thisid-1,numel(colours))+1};
    thisprobs = trackdata.switchProbs(idx{i}, colourtoplot);
    plot(thistimes, thisprobs, [thiscol 'x-']);
    if plotid
        text(thistimes(end), thisprobs(end), ['  ' num2str(uqids(i))],...
        	'Color', thiscol);
    end
end
