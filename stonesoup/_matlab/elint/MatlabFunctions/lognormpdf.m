function [logpdf, mahalsq] = lognormpdf(x, mu, C)

% logpdf = lognormpdf(x, mu, C)

if size(x,1)==1
    mahalsq = (x-mu).^2./C;
    logpdf = -0.5*(mahalsq + log(2*pi*C));
else
    dx = bsxfun(@minus, x, mu);
    mahalsq = sum(dx.*(C\dx),1);
    logpdf = -0.5*(mahalsq + log(det(2*pi*C)));
end