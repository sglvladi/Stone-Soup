function plotMeasCoords(sensorData, dimstoplot, plotcolours)

isplotmmsi = false;

if ~exist('dimstoplot', 'var') || isempty(dimstoplot)
    dimstoplot = [1 2];
end
if ~exist('plotcolours', 'var') || isempty(plotcolours)
    plotcolours = {'r.','bx', 'kx'};
end

for s = 1:numel(sensorData)
    thesemeas = sensorData{s}.meas;
    if numel(dimstoplot)>1
        cx = thesemeas.coords(:,dimstoplot(1));
        cy = thesemeas.coords(:,dimstoplot(2));
    else
        cx = thesemeas.times;
        cy = thesemeas.coords(:,dimstoplot(1));
    end
    c = plotcolours{mod(s-1, numel(plotcolours))+1};
    plot(cx, cy, c)
    if isplotmmsi && isfield(thesemeas, 'mmsi')
        for j = 1:size(thesemeas.coords,1)
            text(cx(j), cy(j), [' ' thesemeas.mmsi{j}]);
        end
    end
end
