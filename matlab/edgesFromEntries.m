function [echk, evar] = edgesFromEntries(rows, cols, shifts, Z)
%EDGESFROMENTRIES  Expand QC base entries into a flat lifted edge list.
%   rows, cols, shifts are 0-indexed block-row / block-col / shift of each
%   base entry.  Entry (i,j,v) -> Z edges:
%       check (i*Z + r)  <->  var (j*Z + mod(r+v,Z))     r = 0..Z-1
%   Returns 1-indexed node lists echk, evar (column vectors), so they index
%   directly into MATLAB arrays.  Port of nr_ldpc.edges_from_entries.
    rows = double(rows(:)); cols = double(cols(:)); shifts = double(shifts(:));
    if isempty(rows)
        echk = zeros(0, 1); evar = zeros(0, 1);
        return;
    end
    r = 0:(Z - 1);                                   % 1 x Z
    chk = (rows * Z + r) + 1;                         % nE x Z (implicit expansion)
    var = (cols * Z + mod(r + shifts, Z)) + 1;        % nE x Z
    echk = chk(:);
    evar = var(:);
end
