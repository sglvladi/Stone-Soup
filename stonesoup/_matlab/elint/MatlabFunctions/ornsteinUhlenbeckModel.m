function [A, Q] = ornsteinUhlenbeckModel(dt, qs, Ks)

% [A, Q] = ornsteinUhlenbeckModel(dt, qs, Ks)
%
% See page 81 of Simon's thesis

ndim = 2*numel(qs);
A = zeros(ndim);
Q = zeros(ndim);
for i=1:numel(qs)
    idx = (i*2) + [-1 0];
    thisK = Ks(i);
    if thisK==0
        [thisA, thisQ] = constantVelocityModel(dt, qs(i));
    else
        expmKdt = exp(-thisK*dt);
        expm2Kdt = exp(-2*thisK*dt);
        thisA = [1 (1 - expmKdt)/thisK; 0 expmKdt];
        q11 = qs(i)/thisK^2*(dt - 2/thisK*(1 - expmKdt) + 1/(2*thisK)*(1 - expm2Kdt));
        q12 = qs(i)/thisK*((1 - expmKdt)/thisK - 1/(2*thisK)*(1 - expm2Kdt));
        q22 = qs(i)/(2*thisK)*(1 - expm2Kdt);
        thisQ = [q11 q12; q12 q22];
    end
    A(idx,idx) = thisA;
    Q(idx,idx) = thisQ;
end
