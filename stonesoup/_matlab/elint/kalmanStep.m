function [postMean, postCov, loglik] = kalmanStep(...
    priorMean, priorCov, H, R, meas)

yhat = H*priorMean;
jointmean = [priorMean; yhat];
jointcov = [priorCov priorCov*H'; H*priorCov H*priorCov*H' + R];
[postMean, postCov, loglik] = kalmanupdate(jointmean, jointcov, meas);
