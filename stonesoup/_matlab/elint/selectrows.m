function data = selectrows(data, idx)

fn = fieldnames(data);
for i=1:numel(fn)
    data.(fn{i}) = data.(fn{i})(idx,:);
end
