function test_kalmanStep1d

rng(1, 'twister');

H = [1; 2; 3];
R = 0.1;
meas = 0.5;
ncomp = 5;
priorMeans = rand(1, ncomp);
priorCovs = 0.1 + 0.5*rand(1, 1, ncomp);

[postMeans, postCovs, logliks] = kalmanStep1D(priorMeans, priorCovs, H, R, meas);

[newmeans, newcovs, newlogweights] = mergegaussians1D(...
    postMeans, postCovs, logliks);
for i = 1:size(postMeans, 2)
    [newmeans2(i), newcovs2(i), newlogweights2(i)] = mergegaussians(postMeans(:,i)',...
        permute(postCovs(:,i), [3 2 1]), logliks(:,i)');
end

keyboard