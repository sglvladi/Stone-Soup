function [sensorData, colours, truth] = simulateDenseTargets(...
    ntargets, minLonLat, maxLonLat)

if ~exist('ntargets', 'var')
    ntargets = 10;
end

if ~exist('minLonLat', 'var')
    minLonLat = [0; 55];
    maxLonLat = [1; 56];
end

mins2sec = 60.0;
hours2sec = 60*mins2sec;
days2sec = 24*hours2sec;

maxTime = 24*hours2sec;
measRate = 1/hours2sec;

colours = getColourDefinitions();
%------
for c = 1:numel(colours)
    %colours(c).measCov = colours(c).measCov/100;
    %colours(c).isSwitch = false;
    %colours(c).isHarmonic = false;
end
%------
ncolours = numel(colours);

elint{1}.name = 'ELINT3GHz';
elint{1}.sensor.H = double([1 0 0 0; 0 0 1 0]);
elint{1}.sensor.rates.meas = measRate;
elint{1}.sensor.rates.reveal = 1/days2sec;
elint{1}.sensor.rates.hide = 1/(2*days2sec);
elint{1}.sensor.priorVisProb = 0.5;

elint{2}.name = 'ELINT9GHz';
elint{2}.sensor.H = double([1 0 0 0; 0 0 1 0]);
elint{2}.sensor.rates.meas = measRate;
elint{2}.sensor.rates.reveal = 1/days2sec;
elint{2}.sensor.rates.hide = 1/(2*days2sec);
elint{2}.sensor.priorVisProb = 0.5;

%elint_colours = {find(cellfun(@(x)any(x=='3'), {colours.name}))};
elint_colours = {find(cellfun(@(x)any(x=='3'), {colours.name})),...
    find(cellfun(@(x)any(x=='9'), {colours.name}))};

for j = 1:numel(elint)
    elint{j}.meas.times = [];
    elint{j}.meas.coords = [];
    elint{j}.meas.semimajorSD = [];
    elint{j}.meas.semiminorSD = [];
    elint{j}.meas.orientation = [];
    elint{j}.meas.targetNum = [];
    for c = elint_colours{j}
        elint{j}.meas.(colours(c).name) = [];
    end
end

ais.name = 'AIS';
ais.sensor.H = double([1 0 0 0; 0 0 1 0]);
ais.sensor.R = 1e-4*eye(2);
ais.sensor.rates.meas = measRate;
ais.sensor.rates.reveal = 1/days2sec;
ais.sensor.rates.hide = 1/(2*days2sec);
ais.sensor.logMMSINullLik = log(1e-5); % prob that a target with unknown MMSI has a particular MMSI
ais.sensor.logpMMSIMatch = log(1.0);   % p(z_mmsi = x_mmsi | x_mmsi)
ais.sensor.priorVisProb = 0.5;
ais.meas.times = [];
ais.meas.coords = [];
ais.meas.targetNum = [];
ais.meas.mmsi = {};

% Simulate truth (assume static for now)
truth.lonLat = (minLonLat + (maxLonLat - minLonLat).*rand(2, ntargets))';
truth.colour = nan(ntargets, ncolours);
for c = 1:ncolours
    truth.colour(:,c) = colours(c).range(1) + diff(colours(c).range).*rand(1,ntargets);
end

% Draw MMSIs for targets with no duplicate
done = false;
while ~done
    truth.mmsi = char(unidrnd(10,ntargets,9)-1 + '0');
    done = (size(unique(truth.mmsi, 'rows'),1)==size(truth.mmsi,1));
end


% truth.mmsi = truth_mmsi;

for j = 1:numel(elint)
    for t = 1:ntargets
        
        % Simulate time, coords
        nthesemeas = poissrnd(maxTime*elint{j}.sensor.rates.meas);
        thesetimes = datetime(2020,10,22) +...
            seconds(sort(maxTime*rand(nthesemeas, 1)));
        [thissmajSD, thissminSD, thisorientDeg] =...
            sample_semimaj_semimin_orient(nthesemeas);
        coords = nan(nthesemeas, 2);
        for i = 1:nthesemeas
            thispos = truth.lonLat(t,:)';
            R_lonlat = getLonLatR(thispos,...
                thissmajSD(i), thissminSD(i), thisorientDeg(i));
            coords(i,:) = (thispos + sqrtm(R_lonlat)*randn(2,1))';
        end
        elint{j}.meas.times = [elint{j}.meas.times; thesetimes];
        elint{j}.meas.coords = [elint{j}.meas.coords; coords];
        elint{j}.meas.semimajorSD = [elint{j}.meas.semimajorSD; thissmajSD];
        elint{j}.meas.semiminorSD = [elint{j}.meas.semiminorSD; thissminSD];
        elint{j}.meas.orientation = [elint{j}.meas.orientation; thisorientDeg];
        elint{j}.meas.targetNum = double([elint{j}.meas.targetNum; repmat(t, nthesemeas, 1)]);
    
        % Simulate colours
        for c = elint_colours{j}
            thisname = colours(c).name;
            thesecolours = truth.colour(t,c) +...
                sqrt(colours(c).measCov)*randn(nthesemeas, 1);
            elint{j}.meas.(thisname) = [elint{j}.meas.(thisname); thesecolours];
        end
        
    end
    
    % Sort in time
    fn = fieldnames(elint{j}.meas);
    [~, idx] = sort(elint{j}.meas.times);
    for i = 1:numel(fn)
        elint{j}.meas.(fn{i}) = elint{j}.meas.(fn{i})(idx,:);
    end
end

% Add AIS
for t = 1:ntargets
    % Simulate time, coords
    nthesemeas = poissrnd(maxTime*ais.sensor.rates.meas);
    thesetimes = datetime(2020,10,22) + seconds(sort(maxTime*rand(nthesemeas, 1)));
    thesecoords = truth.lonLat(t,:)' + sqrt(ais.sensor.R)*randn(2, nthesemeas);
    ais.meas.times = [ais.meas.times; thesetimes];
    ais.meas.coords = [ais.meas.coords; thesecoords'];
    ais.meas.mmsi = [ais.meas.mmsi; repmat({truth.mmsi(t,:)}, nthesemeas, 1)];
    ais.meas.targetNum = double([ais.meas.targetNum; repmat(t, nthesemeas, 1)]);
end

% Sort in time
fn = fieldnames(ais.meas);
[~, idx] = sort(ais.meas.times);
for i = 1:numel(fn)
    ais.meas.(fn{i}) = ais.meas.(fn{i})(idx,:);
end
    
sensorData = [{ais} elint];
%--------------------------------------------------------------------------

function [semimaj, semimin, orientDeg] = sample_semimaj_semimin_orient(nmeas)

smaj_mean = 1.3e4/10;
smaj_std = 1e4/10;
smin_mean = 210;
smin_std = 85;

smaj = repmat(1000,nmeas,1);%sample_lognormal(smaj_mean, smaj_std, nmeas)';
smin = repmat(1000,nmeas,1);%sample_lognormal(smin_mean, smin_std, nmeas)';
semimaj = max(smaj, smin);
semimin = min(smaj, smin);
orientDeg = rad2deg(pi*rand(1,nmeas) - pi/2)';

function pts = sample_lognormal(mean, stdev, npts)

[mu, sigma] = lognormalMeanStdev2MuSigma(mean, stdev);
pts = exp(mu + sigma*randn(1,npts));

function [mu, sigma] = lognormalMeanStdev2MuSigma(mean, stdev)

% Convert mean and standard deviation of a distribution to the mu and sigma
% parameters of a log-normal (i.e. mean(exp(Normal(mu,sigma)) = mean and
% std(exp(Normal(mu,sigma))) = stdev)

mu = log(mean) - 0.5*log((stdev/mean)^2 + 1);
sigma = sqrt(log((stdev/mean)^2+1));
