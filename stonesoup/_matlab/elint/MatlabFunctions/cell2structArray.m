function s = cell2structArray(a)
%STRUCT2 Summary of this function goes here
%   Detailed explanation goes here
    fn = fieldnames(a{1});
    
    s = struct();
    for j=1:length(a)     
        for i = 1:numel(fn)
            s(j).(fn{i}) = a{j}.(fn{i});
        end
    end
end