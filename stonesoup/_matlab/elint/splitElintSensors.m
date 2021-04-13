function [elint3, elint9] = splitElintSensors(elint)

% Split data randomly into 3GHz and 9GHz radars

prob3 = 0.5; % prob of 3GHz radar

nmeas = numel(elint.meas.times);
is3 = rand(nmeas, 1) < prob3;

elint3 = elint;
elint3.name = 'ELINT3GHz';
elint3.sensor.rates.meas = prob3*elint3.sensor.rates.meas;
elint3.meas = filterMeasurements(elint3.meas, is3, '9');

elint9 = elint;
elint9.name = 'ELINT9GHz';
elint9.sensor.rates.meas = (1-prob3)*elint9.sensor.rates.meas;
elint9.meas = filterMeasurements(elint9.meas, ~is3, '3');

%--------------------------------------------------------------------------

function meas = filterMeasurements(meas, idx, excludestr)

fn = fieldnames(meas);
fnexcl = fn(cellfun(@(x)contains(x, excludestr), fn));
fn = setdiff(fn, fnexcl);
meas = rmfield(meas, fnexcl);
for i = 1:numel(fn)
    meas.(fn{i}) = meas.(fn{i})(idx,:);
end
