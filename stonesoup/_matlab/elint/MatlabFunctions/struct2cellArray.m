function b = struct2cellArray(a)
%STRUCT2 Summary of this function goes here
%   Detailed explanation goes here
    b = {};
    for i=a
        b{end+1} = i;
    end
end

