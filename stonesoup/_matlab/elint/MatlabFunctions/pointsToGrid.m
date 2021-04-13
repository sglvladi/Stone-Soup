function [xs, ys, vgrid] = pointsToGrid(xy, v)

% [xs, ys, vgrid] = pointsToGrid(xy, v)
%
% Convert points xy (as a 2*n set of points) to grid so vgrid(i,j) is the
% value at xs(j), ys(i)
% Can then show values with imagesc(xs, ys, vgrid)
% Missing grid values are NaN

[xs, ~, jx] = unique(xy(1,:));
[ys, ~, jy] = unique(xy(2,:));
nx = numel(xs);
ny = numel(ys);

vgrid = nan(ny, nx);
idx = sub2ind(size(vgrid), jy, jx);
vgrid(idx) = v;
