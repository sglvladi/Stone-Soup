function d = logdet(A)

% Calculate log determinant of positive definite matrix A
d = 2*sum(log(diag(chol(A))));
