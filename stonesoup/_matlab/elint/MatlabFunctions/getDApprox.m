function [df, f] = getDApprox(func, x, h)

% df = getDApprox(func, x)
%
% Get df(i,j) = numeric approximation to dfunc_i/dx_j(x) where func is a
% function handle

if nargin<3
    h = 1e-6;
end

statedim = size(x,1);
I = eye(statedim);
f = func(x);
fdim = numel(f);
df = zeros(fdim, statedim);
for i=1:statedim
    xh = bsxfun(@plus, x, h*I(:,i));
    df(:,i) = tocolumn((func(xh) - f)/h);
end
