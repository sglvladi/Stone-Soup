function [postMean, postCov, loglik] = kalmanStepHarmonic(...
    priorMean, priorCov, H, R, meas, logHarmonicProbs)

% % Use normal Kalman update for comparison
% [postMean, postCov, loglik] = kalmanStep(priorMean, priorCov, H, R, meas);
% return

nHarmonics = numel(logHarmonicProbs);
statedim = size(priorMean, 1);

postMeans = nan(statedim, nHarmonics);
postCovs = nan(statedim, statedim, nHarmonics);
logprobs = nan(1, nHarmonics);

for n = 1:nHarmonics
    thisH = n*H;
    jointmean = [priorMean; thisH*priorMean];
    jointcov = [priorCov priorCov*thisH';...
        thisH*priorCov thisH*priorCov*thisH' + R];
    [postMeans(:,n), postCovs(:,:,n), loglikelihood] = kalmanupdate(...
        jointmean, jointcov, meas);
    logprobs(n) = loglikelihood + logHarmonicProbs(n);    
end

[logprobs, loglik] = normaliseinlogs(logprobs);
[postMean, postCov] = mergegaussians(postMeans, postCovs, logprobs);
