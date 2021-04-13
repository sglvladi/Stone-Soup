function b = bell(Ns)

% b = bell(Ns)
%
% Calculate Bell's number (number of partitions on a set of n elements) for
% the values in Ns

N = max(Ns(:));
table = zeros(N+1,N+1);
table(1,1) = 1;
for n=1:N
    table(n+1,:) = (0:N).*table(n,:) + [0 table(n,1:end-1)];
end
bs = sum(table,2);
b = reshape(bs(Ns+1), size(Ns));