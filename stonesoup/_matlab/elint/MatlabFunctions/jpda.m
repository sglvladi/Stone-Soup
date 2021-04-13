function postprobs = jpda(hyps, priorprobs)

% postprobs = jpda(hyps, priorprobs)
%
% Do Mutual Exclusion constraint correction using JPDA
% hyps = cell array where hyps{i} = vector of hypotheses for ith target
%   (null hypothesis is NaN)
% probs = cell array of corresponding probabilities

% postprobs = jpda(hyps, priorprobs)
%
% Do Mutual Exclusion constraint correction using JPDA
%
% In:
%   hyps:       cell array where hyps{i} is a vector of hypotheses for
%                 target i (NaN represents the null hypothesis)
%   priorprobs: cell array of corresponding probabilities before JPDA
% 
% Output:
%   postprobs:  Hypothesis probabilities after JPDA

[jointhyps, jointprobs] = enumerateJointHypotheses(hyps, priorprobs);
ntargets = numel(hyps);
postprobs = cell(1,ntargets);
for t=1:ntargets
    thesehyps = hyps{t};
    nthesehyps = numel(thesehyps);
    postprobs{t} = zeros(1,nthesehyps);
    for h=1:nthesehyps
        if isnan(thesehyps(h))
            idx = isnan(jointhyps(:,t));
        else
            idx = thesehyps(h)==jointhyps(:,t);
        end
        postprobs{t}(h) = sum(jointprobs(idx));
    end        
end
