function data = mycsvread(filename)

% data = mycsvread(filename)

getfields = @(x)strsplit(x, ',', 'CollapseDelimiters', false);

fid = fopen(filename);
ln = fgetl(fid);
nlines = 0;
maxnfields = 0;
while ischar(ln)
    maxnfields = max(maxnfields, numel(getfields(ln)));
    nlines = nlines + 1;
    ln = fgetl(fid);
end
frewind(fid);

data = cell(nlines, maxnfields);
for i=1:nlines
    ln = fgetl(fid);
    fields = getfields(ln);
    data(i,1:numel(fields)) = fields;
end
fclose(fid);
