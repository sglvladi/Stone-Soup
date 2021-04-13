function idx = doresample(weights, nsamples)

% idx = doresample(weights, nsamples)

cumw = cumsum(weights)./sum(weights);
nweights = numel(weights);
idx = zeros(1,nsamples);
us = rand(1,nsamples);

for i=1:nsamples
    thisu = us(i);
    upper = nweights;
    lower = 1;
    while lower<upper
        this = floor((lower + upper)/2);
        if cumw(this) > thisu
            upper = this;
        elseif cumw(this) <= thisu
            lower = this + 1;
        end
    end
    idx(i) = lower;
end
