function table = getJoinDistanceTable(means, covs, logweights)

weights = exp(logweights);
[xdim, npts] = size(means);
invcovs = nan(xdim, xdim, npts);
for i = 1:npts
    invcovs(:,:,i) = inv(covs(:,:,i));
end

% Symmetric KL divergence with w(i)w(j)/(w(i) + w(j)) term
table = zeros(npts, npts);
for i = 1:npts
    for j = i+1:npts
        %w = prod(weights([i j]) / sum(weights([i j])));
        dx = means(:,i) - means(:,j);
        tr = trace(invcovs(:,:,j)*covs(:,:,i)) +...
            trace(invcovs(:,:,i)*covs(:,:,j));
        symKL = 0.5*(tr + dx'*sum(invcovs(:,:,[i j]),3)*dx) - xdim;
        table(i,j) = symKL; %w*symKL;
        table(j,i) = table(i,j);
    end
end
