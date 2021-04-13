function drawresults(trackdata, dimstoplot, statedim, colours)

errorconf = 0.95;
ndim = numel(dimstoplot);
nstddev = sqrt(chi2inv(errorconf, ndim));
plotid = true;

if ~exist('colours','var') || isempty(colours)
    colours = {'m','c','b','r','g'};
end

nlines = size(trackdata,1);

times = trackdata(:,1);%datetime(datevec(trackdata(:,1)'));
trackids = trackdata(:,2);
trackxy = trackdata(:,dimstoplot+2)';
plotvar = exist('statedim','var') && ~isempty(statedim);

if plotvar
    trackcov = reshape(trackdata(:,2+statedim+(1:statedim^2))',...
        statedim,statedim,nlines);
end

hold on
[uqids,~,idx] = uniquewithcounts(trackids);
for i=1:numel(uqids)
    thisid = uqids(i);
    thiscol = colours{mod(thisid-1,numel(colours))+1};
    if ndim==1
        plot(times(idx{i}), trackxy(idx{i}), [thiscol 'x-']);
        if plotvar
            thissd = sqrt(tocolumn(trackcov(dimstoplot,dimstoplot,idx{i}))');
            plot(times(idx{i}), trackxy(idx{i})+nstddev*thissd, [thiscol '-']);
            plot(times(idx{i}), trackxy(idx{i})-nstddev*thissd, [thiscol '-']);
        end
        if plotid
            text(times(idx{i}(end)), trackxy(idx{i}(end)), ['  ' num2str(uqids(i))],...
                'Color', thiscol);
        end
    else
        plot(trackxy(1,idx{i}), trackxy(2,idx{i}), [thiscol 'x-']);
        if plotvar
            for k=1:numel(idx{i})
                thismn = trackxy(:,idx{i}(k));
                thiscov = trackcov(dimstoplot, dimstoplot, idx{i}(k));
                gaussellipse(thismn, nstddev^2*thiscov, thiscol);
            end
        end
        if plotid
            text(trackxy(1,idx{i}(end)), trackxy(2,idx{i}(end)),...
                ['  ' num2str(uqids(i))], 'Color', thiscol);
        end
    end
end
