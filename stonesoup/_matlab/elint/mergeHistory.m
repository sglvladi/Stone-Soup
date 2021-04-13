function newhist = mergeHistory(oldhists)

% Given histories in oldhists, merge common assignments and set others to
% NaNs

if size(oldhists, 2) < 2
    newhist = oldhists;
else
    % Get number of steps in the association history which the oldhists have in
    % common
    nincommon = sum(cumprod(flipud(all(diff(oldhists,1,2)==0,2))));
    % Keep those and set the rest to be NaN
    newhist = [nan(size(oldhists,1)-nincommon, 1);...
        oldhists(end-nincommon+1:end,1)];
end