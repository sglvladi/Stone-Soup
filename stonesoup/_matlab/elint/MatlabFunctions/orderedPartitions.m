function result = orderedPartitions(n,k)

% result = orderedPartitions(n,k)
%
% Return columns of result being the ordered partitions of n into k
% nonnegative integers so that sum(result,1)==n

if k==1
    result = n;
elseif n==0
    result = zeros(k,1);
else
    result = cell(1,n+1);
    for i=0:n
        thisr = orderedPartitions(n-i,k-1);        
        result{i+1} = [repmat(i,1,size(thisr,2)); thisr];
    end
    result = cat(2,result{:});
end