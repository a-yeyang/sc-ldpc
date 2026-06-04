function analyzeBg()
%ANALYZEBG  Validate structural assumptions of the 5G NR base graphs (port of
%   analyze_bg.py).  Verifies, on the real base matrices, the structure the
%   efficient 5G NR QC-LDPC encoder relies on:
%     * dimensions (BG1: 46x68, Kb=22 ; BG2: 42x52, Kb=10)
%     * first 4 rows only touch the first 4 parity columns (the "core")
%     * rows >= 4 have an identity diagonal in parity cols, shift 0
%     * the 4Z x 4Z lifted core block is invertible over GF(2)
    addpath(fileparts(mfilename('fullpath')));
    analyzeOne(2, 1, 24, 'BG2 iLS=1 Z=24', 10);
    analyzeOne(1, 1, 24, 'BG1 iLS=1 Z=24', 22);
end

function B = analyzeOne(bg, ils, Z, name, Kb)
    B = loadBaseMatrix(bg, ils, Z);
    [mb, nb] = size(B);
    fprintf('\n=== %s  (NR_%d_%d_%d.txt) ===\n', name, bg, ils, Z);
    fprintf('shape = %d x %d   Kb(info cols) = %d   parity cols = %d\n', mb, nb, Kb, nb - Kb);
    nnz_ = sum(B(:) >= 0);
    fprintf('non-zero base entries = %d\n', nnz_);
    assert(nb - Kb == mb, 'parity cols must equal #rows');

    vals = B(B >= 0);
    fprintf('shift range = [%d, %d]  (Z=%d)\n', min(vals), max(vals), Z);
    assert(max(vals) < Z, 'shift >= Z!');

    % (1) first 4 rows touch only parity cols Kb..Kb+3
    top = B(1:4, Kb + 4 + 1:nb);
    fprintf('first 4 rows, parity cols >= Kb+4 : nnz = %d (expect 0)\n', sum(top(:) >= 0));
    assert(all(top(:) < 0));

    % (2) rows >=4 : identity diagonal at col (Kb+i), shift 0; nothing else
    ok_diag = true;
    for i = 4:(mb - 1)
        diag_col = Kb + i;                 % 0-indexed parity col -> 1-indexed below
        if B(i + 1, diag_col + 1) ~= 0
            ok_diag = false;
            fprintf('  row %d: diagonal col %d shift = %d (expect 0)\n', i, diag_col, B(i + 1, diag_col + 1));
        end
        others = [];
        for c = (Kb + 4):(nb - 1)
            if c ~= diag_col && B(i + 1, c + 1) >= 0
                others(end + 1) = c; %#ok<AGROW>
            end
        end
        if ~isempty(others)
            ok_diag = false;
            fprintf('  row %d: unexpected parity entries at %s\n', i, mat2str(others));
        end
    end
    fprintf('accumulator identity-diagonal structure (rows>=4): %s\n', ternaryStr(ok_diag));
    assert(ok_diag);

    fprintf('core 4x4 (rows 0-3, parity cols Kb..Kb+3) shift submatrix:\n');
    disp(B(1:4, Kb + 1:Kb + 4));

    % (3) lifted 4Z x 4Z core invertible over GF(2)
    core = zeros(4 * Z, 4 * Z);
    for i = 0:3
        for j = 0:3
            core(i*Z + (1:Z), j*Z + (1:Z)) = liftBlock(B(i + 1, Kb + j + 1), Z);
        end
    end
    inv = gf2inv(core);
    fprintf('lifted core block 4Z x 4Z (%dx%d) invertible over GF(2): %d\n', ...
        4*Z, 4*Z, ~isempty(inv));
    assert(~isempty(inv));

    e_cols = zeros(1, 4);
    for j = 0:3
        e_cols(j + 1) = sum(B(5:mb, Kb + j + 1) >= 0);
    end
    fprintf('E-block: #rows>=4 connecting to core parity p0..p3 = %s\n', mat2str(e_cols));
    fprintf('STRUCTURE OK\n');
end

function blk = liftBlock(shift, Z)
    if shift < 0
        blk = zeros(Z, Z);
    else
        blk = circshift(eye(Z), mod(shift, Z), 2);
    end
end

function s = ternaryStr(ok)
    if ok, s = 'OK'; else, s = 'FAIL'; end
end
