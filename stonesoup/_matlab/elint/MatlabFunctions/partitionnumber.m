function p = partitionnumber(ns)

% p = partitionnumber(ns)
%
% Calculate the partition function at nonnegative integers ns
% https://en.wikipedia.org/wiki/Partition_function_(number_theory)

n = max(ns(:));
table = [1 zeros(1,n)];
for k = 1:n
    for j = k+1:n+1
        table(j) = table(j) + table(j-k);
    end
end
p = reshape(table(ns+1),size(ns));
