function [postprobs, net] = makenet(hyps, probs, trackorder)

% [postprobs, net] = makenet(hyps, probs, trackorder)
%
% Make net and compute updated probabilities for EHM2 algorithm

ntracks = numel(hyps);
if ~exist('trackorder', 'var')
    trackorder = 1:ntracks;
end

[parents, roots, acc] = maketree(hyps, trackorder);
net = makeemptynet(parents, roots, acc, hyps);
net = makenodes(net, hyps);
% backward weights need to be calculated first
net = setbackwardweights(net, hyps, probs);
net = setforwardweights(net, hyps, probs);
postprobs = getpostprobs(net, hyps, probs);

%--------------------------------------------------------------------------

function net = makeemptynet(parents, roots, acc, hyps)

% Make empty net
ntracks = numel(parents);
net.roots = roots;
net.levels = cell(1, ntracks);
for t = 1:ntracks
    net.levels{t}.target = t;
    net.levels{t}.parent = parents(t);
    net.levels{t}.children = find(parents==t);
    % Measurements which could be assigned to a track or descendants
    net.levels{t}.accbackward = acc{t};
end
% Get measurements which could be assigned to a track or ancestors
order = getforwardtraversal(net);
for t = order
    if isnan(parents(t))
        net.levels{t}.accforward = hyps{t}(~isnan(hyps{t}));
    else
        net.levels{t}.accforward = union(hyps{t}(~isnan(hyps{t})),...
            net.levels{parents(t)}.accforward);
    end
    net.levels{t}.nodes = {};
end

%--------------------------------------------------------------------------

function net = makenodes(net, hyps)

% Create root nodes
for t = net.roots
    net.levels{t}.nodes{1} = makenewnode(net.levels{t}, hyps{t}, []);
    net.levels{t}.nodes{1}.forward = 1;
end

% Create child nodes for each level
order = getforwardtraversal(net);
for t = order
    childlevels = net.levels{t}.children;
    % Go through each child target
    for cidx = 1:numel(childlevels)
        thischild = childlevels(cidx);
        % Possible measurements to use in measset
        idmeas = intersect(net.levels{t}.accforward,...
            net.levels{thischild}.accbackward);
        nodelookup = nan(1, 2^numel(idmeas));
        for n = 1:numel(net.levels{t}.nodes)
            for aidx = 1:numel(hyps{t})
                thishyp = hyps{t}(aidx);
                % Check if assignment valid
                oldmeasset = net.levels{t}.nodes{n}.measset;
                if ismember(thishyp, oldmeasset)
                    continue;
                end
                newmeasset = intersect([oldmeasset thishyp], idmeas);
                % Represent the measurement set as a binary string and
                % convert to an integer index
                hashval = sum(ismember(idmeas, newmeasset).*...
                    (2.^(0:numel(idmeas)-1))) + 1;
                % Find index of child node if it exists
                L = nodelookup(hashval);
                % Create it if not
                if isnan(L)
                    L = numel(net.levels{thischild}.nodes) + 1;
                    net.levels{thischild}.nodes{L} = makenewnode(...
                        net.levels{thischild}, hyps{thischild}, newmeasset);
                    nodelookup(hashval) = L;
                end
                net.levels{t}.nodes{n}.children(cidx, aidx) = L;
            end
        end
    end
end

function node = makenewnode(level, hyps, measset)

node.children = nan(numel(level.children), numel(hyps));
node.forward = 0;
node.backward = [];
node.measset = measset;

%--------------------------------------------------------------------------

function net = setbackwardweights(net, hyps, probs)

order = getbackwardtraversal(net);
for t = order
    for n = 1:numel(net.levels{t}.nodes)
        weight = 0;
        for aidx = 1:numel(hyps{t})
            thishyp = hyps{t}(aidx);
            % Check if assignment valid
            if ismember(thishyp, net.levels{t}.nodes{n}.measset)
                continue;
            end
            % Pass up weights of children
            thisweight = probs{t}(aidx);
            children = net.levels{t}.children;
            for cidx = 1:numel(children)
                idx = net.levels{t}.nodes{n}.children(cidx, aidx);
                thisweight = thisweight *...
                    net.levels{children(cidx)}.nodes{idx}.backward;
            end
            weight = weight + thisweight;
        end
        net.levels{t}.nodes{n}.backward = weight;
    end
end

%--------------------------------------------------------------------------

function net = setforwardweights(net, hyps, probs)

order = getforwardtraversal(net);
for t = order
    % Push forward weight contributions down to children
    for n = 1:numel(net.levels{t}.nodes)
        for aidx = 1:numel(hyps{t})
            thishyp = hyps{t}(aidx);
            % Check if assignment valid
            if ismember(thishyp, net.levels{t}.nodes{n}.measset)
                continue;
            end
            % Get backward weights
            children = net.levels{t}.children;
            childbackwards = nan(1, numel(children));
            for cidx = 1:numel(children)
                idx = net.levels{t}.nodes{n}.children(cidx, aidx);
                childbackwards(cidx) = net.levels{children(cidx)}.nodes{idx}.backward;
            end
            % Set forward weights of children
            for cidx = 1:numel(children)
                sibweight = prod(childbackwards([1:cidx-1 cidx+1:end]));
                thisweight = probs{t}(aidx) * net.levels{t}.nodes{n}.forward *...
                    sibweight;
                idx = net.levels{t}.nodes{n}.children(cidx, aidx);
                c = children(cidx);
                net.levels{c}.nodes{idx}.forward =...
                    net.levels{c}.nodes{idx}.forward + thisweight;
            end
        end
    end
end

%--------------------------------------------------------------------------

function postprobs = getpostprobs(net, hyps, probs)

ntracks = numel(hyps);
postprobs = cell(1, ntracks);
for t = 1:ntracks
    postprobs{t} = zeros(1, numel(hyps{t}));
    for aidx = 1:numel(hyps{t})
        thishyp = hyps{t}(aidx);
        for n = 1:numel(net.levels{t}.nodes)
            % Check if assignment valid
            if ismember(thishyp, net.levels{t}.nodes{n}.measset)
                continue;
            end
            thisweight = probs{t}(aidx) * net.levels{t}.nodes{n}.forward;
            % Go through children
            children = net.levels{t}.children;
            for cidx = 1:numel(children)
                idx = net.levels{t}.nodes{n}.children(cidx, aidx);
                thisweight = thisweight * net.levels{children(cidx)}.nodes{idx}.backward;
            end
            postprobs{t}(aidx) = postprobs{t}(aidx) + thisweight;            
        end
    end
    postprobs{t} = postprobs{t} / sum(postprobs{t});
end

%--------------------------------------------------------------------------

function order = getforwardtraversal(net)

% Get traversal order through net where parents are visited before children
order = getforwardrec(net, net.roots);

function order = getforwardrec(net, roots)

order = cell(1, numel(roots));
for i = 1:numel(roots)
	order{i} = [roots(i) getforwardrec(net, net.levels{roots(i)}.children)];
end
order = cat(2, order{:});

function order = getbackwardtraversal(net)

% Get traversal order through net where children are visited before parents
order = getbackwardrec(net, net.roots);

function order = getbackwardrec(net, roots)

order = cell(1, numel(roots));
for i = 1:numel(roots)
	order{i} = [getbackwardrec(net, net.levels{roots(i)}.children) roots(i)];
end
order = cat(2, order{:});
