function drawtrackdata(trackdata, dimstoplot, plotcov, colours, sensorNums)

errorconf = 0.95;
ndim = numel(dimstoplot);
nstddev = sqrt(chi2inv(errorconf, ndim));
plotid = true;
[nlines, statedim] = size(trackdata.mean);

if ~exist('colours','var') || isempty(colours)
    colours = {'m','c','b','r','g'};
end

%if plotcov
%    trackcov = reshape(trackdata.cov', statedim, statedim, nlines);
%end

hold on
[uqids,~,idx] = uniquewithcounts(trackdata.id);
for i=1:numel(uqids)
    thisid = uqids(i);
    if exist('sensorNums', 'var')
        jdx = ismember(trackdata.sensorNum(idx{i}), sensorNums);
        thisidx = idx{i}(jdx);
    else
        thisidx = idx{i};
    end
    thistimes = trackdata.time(thisidx)';
    if plotcov
        [thismn, thiscv] = getTrackdataMeansCovs(trackdata, dimstoplot, thisidx);
    else
        thismn = getTrackdataMeansCovs(trackdata, dimstoplot, thisidx);
    end
    %thismn = trackdata.mean(idx{i}, dimstoplot)';
    %if plotcov
    %    thiscv = trackcov(dimstoplot, dimstoplot, idx{i});
    %end
    thiscol = colours{mod(thisid-1,numel(colours))+1};
    if ndim==1
        plot(thistimes, thismn, [thiscol 'x-']);
        if plotcov
            thissd = sqrt(thiscv(:))';
            plot(thistimes, thismn + nstddev*thissd, [thiscol '-']);
            plot(thistimes, thismn - nstddev*thissd, [thiscol '-']);
        end
        if plotid
            text(thistimes(end), thismn(end), ['  ' num2str(uqids(i))],...
                'Color', thiscol);
        end
    else
        plot(thismn(1,:), thismn(2,:), [thiscol 'x-']);
        if plotcov
            for k=1:numel(thisidx)
                gaussellipse(thismn(:,k), nstddev^2*thiscv(:,:,k), thiscol);
            end
        end
        if plotid
            text(thismn(1,end), thismn(2,end), ['  ' num2str(uqids(i))],...
                'Color', thiscol);
        end
    end
end
