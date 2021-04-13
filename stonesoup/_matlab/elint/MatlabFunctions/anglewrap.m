function x = anglewrap(x, angleidx)

% x = anglewrap(x, angleidx)

x(angleidx,:) = mod(x(angleidx,:) + pi, 2*pi) - pi;
