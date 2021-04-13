function b = matricesMulVectors(A, x)

% b = matricesMulVectors(A, x)
%
% Input
%   A: m*n*npts tensor of d*d matrices
%   x: n*npts matrix of n*1 vectors
%
% Output
%   b: m*npts matrix of vectors where b(:,j) = A(:,:,j)*x(:,j)

[m, n, npts] = size(A);
b = zeros(m, npts);
for i=1:m
    b(i,:) = sum(reshape(A(i,:,:), n, npts).*x,1);
end
