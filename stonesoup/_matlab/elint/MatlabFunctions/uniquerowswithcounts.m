function [uq, cnt, idx] = uniquerowswithcounts(v)

% [uq, cnt, idx] = uniquerowswithcounts(v)

[vsort, p] = sortrows(v);
[~,~,j] = unique(vsort, 'rows');
lastidx = [0; find(diff([j;inf]))];

uq = vsort(lastidx(2:end),:);
cnt = diff(lastidx)';
nuq = size(uq,1);

if nargout>2
    idx = cell(1, nuq);
    for i=1:nuq
        idx{i} = p(lastidx(i)+1:lastidx(i+1))';
    end
end