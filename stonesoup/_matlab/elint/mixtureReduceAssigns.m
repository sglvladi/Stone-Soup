function tracks = mixtureReduceAssigns(tracks, params)

for i = 1:numel(tracks)
    tracks{i} = mixtureReduceAssignsTrack(tracks{i}, params);
end

%--------------------------------------------------------------------------

function track = mixtureReduceAssignsTrack(track, params)

oldassignlogprobs = track.logweights;
idx = clusterMeasHistory(track, params.measHistLen, params.compLogProbThresh);
[track, old2newmap] = mergeMeasurementHistory(track, idx, params);

track.state = mixtureReduceAssignsPart(track.state, old2newmap, oldassignlogprobs);
for c = 1:numel(track.colours)
    track.colours(c) = mixtureReduceAssignsPart(track.colours(c),...
        old2newmap, oldassignlogprobs);
end

%--------------------------------------------------------------------------

function [track, old2newmap] = mergeMeasurementHistory(track, idx, params)

nnewcomps = numel(idx);

% old2newmap(i) = new measurement component corresponding to old comp i
% (NaN if old component is deleted)
old2newmap = nan(1, numel(track.logweights));

% Merge measurement histories in track
newlogweights = nan(1, nnewcomps);
newmmsis = cell(1, nnewcomps);
newmeasHist = nan(size(track.measHist,1), nnewcomps);
newexistProbs = nan(1, nnewcomps);
newvisProbsGivenExist = nan(size(track.visProbsGivenExist,1), nnewcomps);
for i = 1:nnewcomps
    [normlogw, newlogweights(i)] = normaliseinlogs(track.logweights(idx{i}));
    newexistProbs(i) = sum(exp(normlogw).*track.existProbs(idx{i}));
    newvisProbsGivenExist(:,i) = sum(...
        exp(normlogw).*track.visProbsGivenExist(:,idx{i}), 2);
    newhist = mergeHistory(track.measHist(:,idx{i}));
    newmeasHist(:,i) = newhist;
    old2newmap(idx{i}) = i;
    % Set component MMSI to be the most likely MMSI for these components
    if numel(unique(track.mmsis(idx{i})))==1
        newmmsis{i} = track.mmsis{idx{i}(1)};
    else
        [uqmmsis, uqmmsiLogProbs] = getMMSILogProbs(track, idx{i});
        [~, mind] = max(uqmmsiLogProbs);
        newmmsis{i} = uqmmsis{mind};
    end
end
% Remove NaNs from start of meas history
nnan = find(any(isnan(newmeasHist),2), 1, 'last');
if nnan > 0
    newmeasHist = newmeasHist(nnan+1:end,:);
end
% Trim down to specified length
if size(newmeasHist,1) > params.measHistLen
    newmeasHist = newmeasHist(end-params.measHistLen+1:end,:);
end

track.logweights = normaliseinlogs(newlogweights);
track.mmsis = newmmsis;
track.measHist = newmeasHist;
track.existProbs = min(max(newexistProbs, 0), 1);
track.visProbsGivenExist = min(max(newvisProbsGivenExist, 0), 1);

%--------------------------------------------------------------------------

function part = mixtureReduceAssignsPart(part, old2newmap, oldassignlogprobs)

% Weight components according to probability of old measurement assignments
% (since they might get merged)
oldlogweights = part.logweightsGivenMeas + oldassignlogprobs(part.measHistIndices);

% Renumber measurement history components
part.measHistIndices = old2newmap(part.measHistIndices);

% Delete any removed ones
tokeep = ~isnan(part.measHistIndices);
if ~all(tokeep)
    oldlogweights = oldlogweights(tokeep);
    tokeep = ~isnan(part.measHistIndices);
    part.logweightsGivenMeas = part.logweightsGivenMeas(tokeep);
    part.means = part.means(:,tokeep);
    part.covs = part.covs(:,:,tokeep);    
    part.measHistIndices = part.measHistIndices(tokeep);
    part.modelHists = part.modelHists(:,tokeep);
end

if all(part.modelHists==1)
    [~,~,idx] = uniquewithcounts(part.measHistIndices);
else
    [~,~,idx] = uniquecolumnswithcounts([part.measHistIndices; part.modelHists]);
end
nnewcomp = numel(idx);
statedim = size(part.means, 1);

newlogweights = nan(1, nnewcomp);
newmeans = nan(statedim, nnewcomp);
newcovs = nan(statedim, statedim, nnewcomp);
newmeashist = nan(1, nnewcomp);
newmodelhist = nan(size(part.modelHists,1), nnewcomp);

for c = 1:nnewcomp
    if numel(idx{c}) > 1
        [newmeans(:,c), newcovs(:,:,c), newlogweights(c)] =...
            mergegaussians(part.means(:,idx{c}), part.covs(:,:,idx{c}),...
            oldlogweights(idx{c}));
    else
        newmeans(:,c) = part.means(:,idx{c});
        newcovs(:,:,c) = part.covs(:,:,idx{c});
        newlogweights(c) = oldlogweights(idx{c});
    end
    newmeashist(c) = part.measHistIndices(idx{c}(1));
    newmodelhist(:,c) = part.modelHists(:,idx{c}(1));
end

% Normalise log weights
[~,~,idx] = uniquewithcounts(newmeashist);
for i = 1:numel(idx)
    newlogweights(idx{i}) = normaliseinlogs(newlogweights(idx{i}));
end

part.logweightsGivenMeas = newlogweights;
part.means = newmeans;
part.covs = newcovs;
part.measHistIndices = newmeashist;
part.modelHists = newmodelhist;
