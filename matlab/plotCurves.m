function plotCurves(curves, varargin)
%PLOTCURVES  Semilog-y BER/FER plot (MATLAB equivalent of plotting.semilogy).
%   curves : struct array with fields  x (Eb/N0 dB), y (BER), label, [fer], [rate].
%   Options (name/value): 'XLabel','YLabel','Title','Path' (png/svg/fig),'CSV'.
    p = inputParser;
    p.addParameter('XLabel', 'Eb/N0 [dB]');
    p.addParameter('YLabel', 'BER');
    p.addParameter('Title', '');
    p.addParameter('Path', 'plot.png');
    p.addParameter('CSV', '');
    p.parse(varargin{:});
    o = p.Results;

    colors = [0.12 0.47 0.71; 0.84 0.15 0.16; 0.17 0.63 0.17; 0.58 0.40 0.74;
              1.00 0.50 0.05; 0.09 0.75 0.81; 0.55 0.34 0.29; 0.89 0.47 0.76;
              0.74 0.74 0.13; 0.22 0.23 0.47; 0.50 0.50 0.50; 0.68 0.78 0.91];

    fig = figure('Visible', 'off', 'Position', [100 100 820 560], 'Color', 'w');
    ax = axes(fig); hold(ax, 'on'); box(ax, 'on');
    leg = {};
    anyData = false;
    for k = 1:numel(curves)
        c = curves(k);
        x = c.x(:); y = c.y(:);
        keep = y > 0;                  % only positive BER on a log axis
        if ~any(keep), continue; end
        col = colors(mod(k - 1, size(colors, 1)) + 1, :);
        semilogy(ax, x(keep), y(keep), '-o', 'Color', col, ...
            'MarkerFaceColor', col, 'LineWidth', 1.8, 'MarkerSize', 5);
        leg{end + 1} = c.label; %#ok<AGROW>
        anyData = true;
    end
    if ~anyData
        close(fig); return;
    end
    set(ax, 'YScale', 'log');
    grid(ax, 'on'); ax.GridAlpha = 0.25;
    xlabel(ax, o.XLabel); ylabel(ax, o.YLabel);
    if ~isempty(o.Title), title(ax, o.Title, 'FontWeight', 'bold'); end
    legend(ax, leg, 'Location', 'eastoutside', 'Interpreter', 'none', 'FontSize', 9);

    saveFigure(fig, o.Path);
    close(fig);
    if ~isempty(o.CSV)
        writeCsv(curves, o.CSV);
    end
    fprintf('wrote %s%s\n', o.Path, ternaryStr(~isempty(o.CSV), [' and ' o.CSV], ''));
end

function saveFigure(fig, path)
    [~, ~, ext] = fileparts(path);
    switch lower(ext)
        case '.png',  exportgraphics(fig, path, 'Resolution', 150);
        case '.svg',  saveas(fig, path, 'svg');
        case '.pdf',  exportgraphics(fig, path, 'ContentType', 'vector');
        case '.fig',  savefig(fig, path);
        otherwise,    exportgraphics(fig, path, 'Resolution', 150);
    end
end

function writeCsv(curves, path)
%WRITECSV  Same layout as plotting.write_csv.
    fid = fopen(path, 'w');
    for k = 1:numel(curves)
        c = curves(k);
        fprintf(fid, '# %s\n', c.label);
        fprintf(fid, 'ebn0_db,ber,fer\n');
        x = c.x(:); y = c.y(:);
        if isfield(c, 'fer') && ~isempty(c.fer), fer = c.fer(:); else, fer = nan(size(x)); end
        for i = 1:numel(x)
            if isnan(fer(i))
                fprintf(fid, '%g,%g,\n', x(i), y(i));
            else
                fprintf(fid, '%g,%g,%g\n', x(i), y(i), fer(i));
            end
        end
    end
    fclose(fid);
end

function s = ternaryStr(cond, a, b)
    if cond, s = a; else, s = b; end
end
