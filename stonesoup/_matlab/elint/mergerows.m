function outdata = mergerows(data)

% Merge struct array of fields and sort into time order

fn = fieldnames(data{1});
outdata = data{1};
for i = 2:numel(data)
    for fi = 1:numel(fn)
        f = fn{fi};
        outdata.(f) = [outdata.(f); data{i}.(f)];
    end
end

[~, idx] = sort(outdata.times);
for fi = 1:numel(fn)
    f = fn{fi};
    outdata.(f) = outdata.(f)(idx,:);
end
