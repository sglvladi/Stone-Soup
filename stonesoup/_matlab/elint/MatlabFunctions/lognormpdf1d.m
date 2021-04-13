function [logpdf, mahalsq] = lognormpdf1d(x, mu, var)

% logpdf = lognormpdf1d(x, mu, var)

mahalsq = (x-mu).^2./var;
logpdf = -0.5*(mahalsq + log(2*pi*var));
