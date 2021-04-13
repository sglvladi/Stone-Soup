function [F, Q] = getTransitionModel(lonLat, dt, txmodel)%, colours)

thisq = txmodel.q_metres./(degree2metres(lonLat).^2);
if txmodel.isOrnsteinUhlenbeck
    [F, Q] = ornsteinUhlenbeckModel(dt, thisq, txmodel.K*[1 1]);
else
    [F, Q] = constantVelocityModel(dt, thisq);
end
