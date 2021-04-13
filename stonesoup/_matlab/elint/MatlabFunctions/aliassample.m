function samples = aliassample(F, L, nsamples)

% samples = aliassample(F, L, nsamples)
%
% Sample from discrete pdf preprocessed with aliascalc.m (Alias algorithm)
% Kronmal and Peterson, On the Alias Method for Generating Random Variables
% from a Discrete Distribution (1979)
% http://www.jstor.org/stable/2683739?seq=1#page_scan_tab_contents

u = numel(F)*rand(1,nsamples);
i = ceil(u);
u = i - u;
samples = zeros(nsamples,1);

idx = (u <= F(i));
samples(idx) = i(idx);
samples(~idx) = L(i(~idx));
