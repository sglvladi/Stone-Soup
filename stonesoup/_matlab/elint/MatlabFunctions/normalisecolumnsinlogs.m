function [A, s] = normalisecolumnsinlogs(A)

% [A, s] = normalisecolumnsinlogs(A)

s = -inf(1, size(A, 2));
if ~isempty(A)
    for i=1:size(A, 2)
        [A(:,i), s(i)] = normaliseinlogs(A(:,i));
    end
end