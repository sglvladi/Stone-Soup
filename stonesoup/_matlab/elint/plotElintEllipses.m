function plotElintEllipses(meas)

lonlat = meas.coords(:,1:2)';
R_lonlat = getLonLatR(lonlat, meas.semimajorSD, meas.semiminorSD,...
    meas.orientation);
gaussellipse(lonlat, R_lonlat, 'b')
