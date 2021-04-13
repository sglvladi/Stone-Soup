function Neff = getNeff(logweights)

% Neff = getNeff(logweights)
%
% Get effective particle size, 1/(sum(w.^2))

Neff = 1/sum(exp(2*normaliseinlogs(logweights)));
