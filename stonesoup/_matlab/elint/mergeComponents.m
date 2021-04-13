function mergeComponents(means, covs, logweights, idx)

% ::TODO::

% Go through each measurement history
for ai = 1:numel(oldassignidx)
    % Components with assignment history belonging to this
    thesecompidx = ismember(part.measHistIndices, oldassignidx{ai});
    % Joint weight of assignment and model
    theselogweights = oldassignlogweights(part.measHistIndices(thesecompidx)) +...
        part.logweightsGivenMeas(thesecompidx);
    thesemeans = part.means(:,thesecompidx);
    thesecovs = part.covs(:,:,thesecompidx);
    % Merge model components with this measurement history
    thismodelHist = oldModelHists(:,thesecompidx);
    [newmodelHists{ai},~,idx] = uniquecolumnswithcounts(thismodelHist);
    thisncomps = numel(idx);
    newmeasHistIndices{ai} = repmat(ai, 1, thisncomps);
    newMeans{ai} = nan(statedim, thisncomps);
    newCovs{ai} = nan(statedim, statedim, thisncomps);
    for ci = 1:thisncomps
        [newMeans{ai}(:,ci), newCovs{ai}(:,:,ci), newlogweights{ai}(ci)] =...
            mergegaussians(thesemeans(:,idx{ci}), thesecovs(:,:,idx{ci}),...
            theselogweights(idx{ci}));
    end
    newlogweights{ai} = normaliseinlogs(newlogweights{ai});
    % Prune low-weight model components?
end