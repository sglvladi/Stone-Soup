function fig2epsc

% fig2epsc
%
% Convert .fig files in current directory to .eps

d = dir('*.fig');
for i=1:numel(d)
    fname = d(i).name;
    [~,fnameroot,~] = fileparts(fname);
    h = openfig(fname);
    saveas(h, [fnameroot '.eps'], 'epsc');
    close(h)
end
