function gaussellipse(mean, cov, c, m, ln)

% gaussellipse(mean, cov, c)

if ~exist('c','var') || isempty(c)
    c = 'b';
end
if ~exist('ln','var') || isempty(ln)
    ln = '-';
end

hold on

npoints = 20;
theta = linspace(0, 2*pi, npoints+1);
circlepoints = [sin(theta); cos(theta)];
nellipses = size(mean,2);
for i=1:nellipses
    [v,d] = eig(cov(:,:,i));
    vd = v*sqrt(d);
    ellipsepoints = vd*circlepoints;
    plot(ellipsepoints(1,:)+mean(1,i), ellipsepoints(2,:)+mean(2,i), ln, 'Color', c)
end
if exist('m', 'var') && ~isempty(m)
    plot(mean(1,:), mean(2,:), [m '-'], 'Color', c)
end
