function gatedTrackIndices = doGating(tracks, txmodel, meas, gateSD, colours)

numTracks = numel(tracks);
isgated = false(1,numTracks);

% covPlot(meas, tracks, colours, gateSD)

for i = 1:numTracks
    
    if isGatedMMSI(meas, tracks{i})
        
        dt = meas.time - tracks{i}.prevTime;
        
        % Get transition kernel (dependent on the mean track position)
        [F, Q] = getStateTransition(tracks{i}, dt, txmodel);
        ncomps = numel(tracks{i}.state.logweightsGivenMeas);
        mahalsq_col = getColourMahalSq(meas, dt, tracks{i}, colours);
        for c = 1:ncomps
            associdx = tracks{i}.state.measHistIndices(c);
            mu = F*tracks{i}.state.means(:,c);
            P = F*tracks{i}.state.covs(:,:,c)*F' + Q;
            zbar = meas.H*mu;
            S = meas.H*P*meas.H' + meas.R;
            [~, mahalsq_state] = lognormpdf(meas.pos, zbar, S);
            mahalsq = mahalsq_state + mahalsq_col(associdx);
            if mahalsq <= gateSD^2
                isgated(i) = true;
                break
            end
        end
        
    end
end

gatedTrackIndices = find(isgated);

%--------------------------------------------------------------------------

function mahalsqAssoc = getColourMahalSq(meas, dt, track, colours)

nassoccomps = numel(track.logweights);
ncoloursdef = numel(meas.coloursDefined);
mahalsqAssoc = zeros(1, nassoccomps);

for c = 1:ncoloursdef
    cidx = meas.coloursDefined(c);
    meascol = meas.colour(c);
    if ~colours(cidx).isSwitch
        
        % Get track mean and covariance for model components
        pred_comp_mn = track.colours(cidx).means;
        pred_comp_cv = track.colours(cidx).covs(:)' + colours(cidx).q*dt;
        dim = size(pred_comp_mn, 1);
        assert(dim==1);

        % Deal with harmonics
        if colours(cidx).isHarmonic
            nharmonics = numel(colours(cidx).harmonicLogProbs);
            pred_comp_mn = (1:nharmonics)'.*pred_comp_mn;
            pred_comp_cv = ((1:nharmonics).^2)'.*pred_comp_cv;
        end
        
        % Get mahal squared for model components (minimum across harmonics)
        meas_comp_cv = pred_comp_cv + colours(cidx).measCov;
        mahalsqModel = min((pred_comp_mn - meascol).^2./meas_comp_cv,[],1);
        
        % Get assoc mahal squared distance equal to minimum across the
        % models
        [ii,~,idx] = uniquewithcounts(track.colours(cidx).measHistIndices);
        for i = 1:numel(ii)
            mahalsqAssoc(ii(i)) = mahalsqAssoc(ii(i)) + min(...
                mahalsqModel(idx{i}));
        end
        
    end
end

%--------------------------------------------------------------------------

function isgated = isGatedMMSI(meas, track)

if ~isfield(meas, 'mmsi') || isempty(meas.mmsi)
    isgated = true;
else
    % MMSI gated is there are any components with no MMSI or one which
    % matches this
    isg_comp = cellfun(@(x)(isempty(x) || isequal(x, meas.mmsi)), track.mmsis);
    isgated = any(isg_comp);
end

%--------------------------------------------------------------------------

function covPlot(meas, tracks, colours, gateSD)

c = meas.coloursDefined(1);
mn = cellfun(@(x)x.colours(c).means, tracks, 'UniformOutput', false);
mn = cat(2, mn{:});
cv = cellfun(@(x)x.colours(c).covs(:)' + colours(c).measCov, tracks,...
    'UniformOutput', false);
cv = cat(2, cv{:});
id = cellfun(@(x)[1 zeros(size(x.colours(c).means,2,1)-1)], tracks, 'UniformOutput', false);
id = cumsum(cat(2, id{:}));
figure
hold on
for i = 1:numel(mn)
    plot([i i], mn(i) + gateSD*sqrt(cv(i))*[-1 1], 'b.-')
end
plot([0 i+1], meas.colour(1)*[1 1], 'r-')
