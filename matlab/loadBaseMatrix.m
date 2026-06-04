function B = loadBaseMatrix(bg, ils, Z)
%LOADBASEMATRIX  Load a 5G NR base matrix NR_{bg}_{ils}_{Z}.txt.
%   Returns an integer matrix; -1 means "no edge", otherwise the circulant
%   shift value in [0, Z).  Data files live in ../data relative to this file
%   (3GPP TS 38.212 base graphs, shifts already reduced mod Z).
%
%   Parsed explicitly line-by-line (like the Python loader) so trailing spaces
%   / multiple delimiters cannot introduce spurious NaN columns.
    here = fileparts(mfilename('fullpath'));
    dataDir = fullfile(fileparts(here), 'data');
    path = fullfile(dataDir, sprintf('NR_%d_%d_%d.txt', bg, ils, Z));
    if ~isfile(path)
        error('loadBaseMatrix:missing', 'base matrix not found: %s', path);
    end
    lines = splitlines(string(fileread(path)));
    rows = {};
    for k = 1:numel(lines)
        s = strtrim(lines(k));
        if strlength(s) == 0
            continue;
        end
        vals = sscanf(s, '%d').';          % row vector of integers
        rows{end+1} = vals;                %#ok<AGROW>
    end
    widths = cellfun(@numel, rows);
    assert(all(widths == widths(1)), 'inconsistent row widths in %s', path);
    B = vertcat(rows{:});
end
