function d = getDistanceTable(xs, ys)

% d = getDistanceTable(xs, ys)
%
% Get d(i,j) = Euclidean distance between xs(:,i) and ys(:,j)

d = sqrt(sum((permute(xs, [2 3 1]) - permute(ys, [3 2 1])).^2, 3));
