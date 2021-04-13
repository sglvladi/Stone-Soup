function [postprobs, net] = ehm2(hyps, probs)

% [postprobs, net] = ehm2(hyps, probs)

trackorder = 1:numel(hyps); % Add better track ordering later?
[postprobs, net] = makenet(hyps, probs, trackorder);
