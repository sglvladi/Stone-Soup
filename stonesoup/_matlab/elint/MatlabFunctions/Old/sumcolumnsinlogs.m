function s = sumcolumnsinlogs(A)

% s = sumcolumnsinlogs(A)

s = zeros(1, size(A, 2));
for i=1:size(A, 2)
    [~, s(i)] = normaliseinlogs(A(:,i));
end
