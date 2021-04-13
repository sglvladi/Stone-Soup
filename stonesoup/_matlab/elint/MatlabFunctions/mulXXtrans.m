function xx = mulXXtrans(x)

% xx = mulXXtrans(x)
%
% Get xx(:,:,k) = x(:,k)*x(:,k)'

xx = permute(x, [1 3 2]).*permute(x, [3 1 2]);
