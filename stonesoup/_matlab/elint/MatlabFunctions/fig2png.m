function fig2png

% fig2epsc
%
% Convert .fig files in current directory to .png

d = dir('*.fig');
for i=1:numel(d)
    fname = d(i).name;
    [~,fnameroot,~] = fileparts(fname);
    h = openfig(fname);
    saveas(h, [fnameroot '.png'], 'png');
    close(h)
end
