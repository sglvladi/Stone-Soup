function [postMeans, postCovs, logliks] = kalmanStepHarmonic1D(...
    priorMeans, priorCovs, H, R, meas, logHarmonicProbs)

nHarmonics = numel(logHarmonicProbs);
[postMeans, postCovs, logliks] = kalmanStep1D(priorMeans, priorCovs,...
    (1:nHarmonics)'.*H, R, meas);

logprobs = logHarmonicProbs(:) + logliks;
[postMeans, postCovs, logliks] = mergegaussians1D(postMeans, postCovs,...
    logprobs);
