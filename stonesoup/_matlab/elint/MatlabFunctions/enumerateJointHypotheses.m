function [jointhyps, jointprobs] = enumerateJointHypotheses(hyps, probs)

% [jointhyps, jointprobs] = enumerateJointHypotheses(hypotheses)
%
% Enumerate joint target hypotheses and calculate joint probabilities where
% at most one target can be assigned a non-null hypothesis
%
% In:
%   hyps:  cell array where hyps{i} is a vector of hypotheses for target i
%           (NaN represents the null hypothesis)
%   probs: cell array of corresponding probabilities
% 
% Output:
%   jointhyps:  Array where (i,t) is the hypothesis for target t in the ith
%                 joint hypothesis
%   jointprobs: Column vector where jointprobs(i) is the joint probability
%                 of the ith joint hypothesis (and sum(jointprobs)==1)

ntargets = numel(hyps);
if ntargets==0
    jointhyps = [];
    jointprobs = [];
    return
end

jointhyps = hyps{1}(:);
jointprobs = probs{1}(:);
for t=2:ntargets
    thesehyps = hyps{t};
    theseprobs = probs{t};
    nthesehyps = numel(thesehyps);
    newjointhyps = cell(1,nthesehyps);
    newjointprobs = cell(1,nthesehyps);
    for i=1:nthesehyps
        thishyp = thesehyps(i);
        thisprob = theseprobs(i);
        isvalid = all(jointhyps~=thishyp, 2); % note NaN~=NaN
        nvalid = sum(isvalid);
        newjointhyps{i} = [jointhyps(isvalid,:) repmat(thishyp, nvalid, 1)];
        newjointprobs{i} = jointprobs(isvalid)*thisprob;
    end
    jointhyps = cat(1,newjointhyps{:});
    jointprobs = cat(1,newjointprobs{:});
end
jointprobs = jointprobs/sum(jointprobs);
