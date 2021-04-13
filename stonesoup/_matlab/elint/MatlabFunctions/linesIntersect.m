function lambdamu = linesIntersect(x1, v1, x2, v2)

% lambdamu = linesIntersect(x1, v1, x2, v2)
%
% Get lambda, mu such that x1 + lambda*v1 = x2 + mu*v2

A = [-v1(:) v2(:)];
if abs(det(A)) < 1e-16
    lambdamu = [NaN; NaN];
else
    lambdamu = A\(x1 - x2);
end
