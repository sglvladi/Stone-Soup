function appendTitle(str)

% appendTitle(str)
%
% Append string to titles of figures in current dir

d = dir('*.fig');
for i=1:numel(d)
    fname = d(i).name;
    [~,fnameroot,~] = fileparts(fname);
    h = openfig(fname);
    hg = gca;
    thetitle = hg.Title.String;%(1:find(hg.Title.String==':',1,'last')-1);
    title(thetitle + string(str));
    saveas(h, [fnameroot '.fig'], 'fig');
    close(h)
end
