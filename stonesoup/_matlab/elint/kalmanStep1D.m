function [postMeans, postCovs, logliks] = kalmanStep1D(...
    priorMeans, priorCovs, H, R, meas)

% [postMeans, postCovs, logliks] = kalmanStep1D(priorMeans, priorCovs, H, R, meas)
%
% Calculate updated Kalman filter means and variances for 1-d distributions
% H can be multiple valued for different measurement matrices

H = H(:);
R = R(:);
M = priorCovs(:)';
predmeas = H*priorMeans;

S = (H.^2).*M + R(:);
K = (M.*H)./S;
postCovs = (1 - K.*H).*M;
postMeans = priorMeans + K.*(meas - predmeas);

if nargout > 2
    logliks = lognormpdf1d(predmeas, meas, S);
end
