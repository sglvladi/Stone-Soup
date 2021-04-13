function idx = isinbox(xs, ax)

% idx = isinbox(xs, ax)

idx = (xs(1,:)>=ax(1) & xs(1,:)<=ax(2) & xs(2,:)>=ax(3) & xs(2,:)<=ax(4));
