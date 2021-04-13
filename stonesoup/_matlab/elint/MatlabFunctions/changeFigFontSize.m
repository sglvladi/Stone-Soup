function changeFigFontSize(fontsize)

close all
d = dir('*.fig');
for i=1:numel(d)
    fname = d(i).name;
    [~,fnameroot,~] = fileparts(fname);
    h = openfig(fname);
    set(findall(gcf,'-property','FontSize'),'FontSize',fontsize);
    saveas(h, [fnameroot '.fig'], 'fig');
    close(h)
end