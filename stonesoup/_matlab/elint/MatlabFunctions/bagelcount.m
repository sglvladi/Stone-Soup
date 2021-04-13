function result = bagelcount(nbagels, ntypes)

% result = bagelcount(nbagels, ntypes)
%
% Enumerate possible ways of choosing nbagels bagels from ntypes types
% result(i,j) = number of bagels of type j in combination i

combs = nchoosek(1:(nbagels+ntypes-1), ntypes-1);
result = [combs(:,1) diff(combs,1,2) nbagels+ntypes-combs(:,end)] - 1;
