function [F,L] = aliascalc(pdf)

% [F,L] = aliascalc(pdf)
%
% Precalculate F, L from a discrete PDF for use in aliassample.m (sampling
% with Alias algorithm)
% Kronmal and Peterson, On the Alias Method for Generating Random Variables
% from a Discrete Distribution (1979)
% http://www.jstor.org/stable/2683739?seq=1#page_scan_tab_contents

n = numel(pdf);
F = n*pdf/sum(pdf);
L = 1:n;

G = find(F>=1);
S = find(F<1);
numS = numel(S);
numG = numel(G);

while numS>0 && numG>0
    k = G(numG);
    j = S(numS);
    L(j) = k;
    F(k) = F(k) - (1-F(j));
    if F(k)<1
        numG = numG - 1;
        S(numS) = k;
    else
        numS = numS - 1;
    end
end
