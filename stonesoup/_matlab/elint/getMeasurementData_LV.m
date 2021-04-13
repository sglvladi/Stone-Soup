function meas = getMeasurementData_LV(sensorData, lineindex, colours)

colours = cell2structArray(colours);

% Get kinematic measurement information
measPos = sensorData.meas.coords(lineindex,:)';
H = sensorData.sensor.H;
if isfield(sensorData.sensor, 'R')
    R = sensorData.sensor.R;
else
    % If R not defined in sensor, calculate it from smaj, smin, orient
    meas_semimajor = sensorData.meas.semimajorSD(lineindex);
    meas_semiminor = sensorData.meas.semiminorSD(lineindex);
    meas_orientDeg = sensorData.meas.orientation(lineindex);
    R = getLonLatR(measPos(1:2), meas_semimajor,...
        meas_semiminor, meas_orientDeg);
end

% Get colour information
[measColour, coloursDefined] = getColourLikelihood(...
    sensorData, lineindex, colours);

% Get MMSI if defined
if isfield(sensorData.meas, 'mmsi')
    mmsi = sensorData.meas.mmsi{lineindex};
else
    mmsi = '';
end

meas.time = sensorData.meas.times(lineindex);
meas.pos = measPos;
meas.colour = measColour;
meas.coloursDefined = coloursDefined;
meas.mmsi = mmsi;
meas.H = H;
meas.R = R;

%--------------------------------------------------------------------------

function [meas, defined] = getColourLikelihood(sensorData, lineindex, colours)

if isempty(colours)
    coloursDefined = [];
else
    coloursDefined = find(ismember({colours.name}, fieldnames(sensorData.meas)));
end
ncoloursdef = numel(coloursDefined);
meas = zeros(ncoloursdef, 1);
for i = 1:ncoloursdef
    thisc = coloursDefined(i);
    meas(i) = sensorData.meas.(colours(thisc).name)(lineindex);
end

% Check for missing data
idx = ~isnan(meas);
meas = meas(idx);
defined = coloursDefined(idx);
