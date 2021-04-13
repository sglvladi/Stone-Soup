function [uq, n, idx] = uniquecolumnswithcounts(v)

if nargout>2
    [uq, n, idx] = uniquerowswithcounts(v');
else
    [uq, n] = uniquerowswithcounts(v');
end
uq = uq';
