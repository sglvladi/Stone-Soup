function measLonLat = simulateLonLatMeasurements(...
    truthLonLat, semimajSD, semiminSD, orientDeg)

nmeas = size(truthLonLat, 2);

% Make singular error parameters the same size as the number of
% measurements
semimajSD = semimajSD(:)'.*ones(1,nmeas);
semiminSD = semiminSD(:)'.*ones(1,nmeas);
orientDeg = orientDeg(:)'.*ones(1,nmeas);

measLonLat = zeros(2, nmeas);
for i = 1:size(truthLonLat, 2)
    R = getLonLatR(truthLonLat(:,i), semimajSD(i), semiminSD(i), orientDeg(i));
    measLonLat(:,i) = mvnrnd(truthLonLat(:,i)', R)';
end
