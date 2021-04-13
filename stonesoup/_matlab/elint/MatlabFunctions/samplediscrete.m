function idx = samplediscrete(w, nsamples)

% idx = samplediscrete(w, nsamples)

w = w(:)'/sum(w(:));
idx = randsample(numel(w), nsamples, true, w)';