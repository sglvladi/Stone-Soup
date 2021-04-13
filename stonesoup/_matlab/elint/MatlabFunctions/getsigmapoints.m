function [pts, w] = getsigmapoints(mu, P)

% [pts, w] = getsigmapoints(mu, P)

kappa = 1;
xdim = size(mu,1);
dx = sqrtm((xdim+kappa)*P);%sqrt(xdim+kappa) * sqrtm(P); % Do this with eigenvectors later?
pts = [mu bsxfun(@plus, mu, dx) bsxfun(@minus, mu, dx)];
if any(imag(pts(:)))
    warning('Complex component on sigma points');
end
w = [kappa/(xdim + kappa) repmat(0.5/(xdim + kappa), 1, 2*xdim)];
