function idx = findstringincell(str, acell)

% idx = findstringincell(str, acell)
%
% Return idx(i) = true for the cells which match str

idx = cellfun(@(x)isequal(x, str), acell);