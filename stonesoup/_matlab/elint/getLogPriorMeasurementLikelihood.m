function [logpdf, logpdfcomps] = getLogPriorMeasurementLikelihood(prior, meas, sensorData,...
    colours)

% Get log pdf of prior measurement - assume uniformly distributed over
% position and the colours which are defined for this measurement

logpdfPos = -sum(log(prior.posMax - prior.posMin));

colourRanges = arrayfun(@(x)diff(x.range), colours(meas.coloursDefined));
logpdfColour = -log(colourRanges);

logpdf = logpdfPos + sum(logpdfColour);

logpdfcomps = [logpdfPos logpdfColour];

if ~isempty(meas.mmsi)
    logpdf = logpdf + sensorData.sensor.logMMSINullLik;
end
