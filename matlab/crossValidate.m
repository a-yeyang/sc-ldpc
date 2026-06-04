function crossValidate()
%CROSSVALIDATE  Bit-exact check of the MATLAB component encoder + BP decoder
%   against deterministic reference vectors produced by the Python implementation
%   (/tmp/ref_component.json).  The component code has no RNG, so MATLAB must
%   reproduce Python bit-for-bit.
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    refPath = fullfile(here, 'ref_component.json');     % shipped alongside this file
    if ~isfile(refPath)
        refPath = '/tmp/ref_component.json';            % fallback: freshly generated
    end
    assert(isfile(refPath), 'reference vectors not found (ref_component.json)');
    ref = jsondecode(fileread(refPath));

    % ---- core_inv fingerprint (GF(2) inverse) ----
    code = NRLDPCCode(2, 1, 24);
    ci = code.core_inv;
    ri = ref.core_inv_bg2_z24;
    assert(isequal(size(ci), ri.shape(:)'), 'core_inv shape mismatch');
    assert(sum(ci(:)) == ri.total, 'core_inv total mismatch (%d vs %d)', sum(ci(:)), ri.total);
    assert(trace(ci) == ri.diag, 'core_inv diag mismatch');
    assert(isequal(sum(ci, 2), ri.rowsum(:)), 'core_inv rowsum mismatch');
    fprintf('PASS  core_inv GF(2) inverse matches Python (sum=%d, diag=%d)\n', ri.total, ri.diag);

    % ---- per-case component encode + decode ----
    cases = ref.component_cases;
    for k = 1:numel(cases)
        if iscell(cases), c = cases{k}; else, c = cases(k); end
        code = NRLDPCCode(c.bg, c.ils, c.Z, c.mp);   % c.mp is the actual mp used
        % dimensions
        assert(code.Kb == c.Kb && code.mb == c.mb && code.nb == c.nb, 'dims mismatch');
        assert(code.K == c.K && code.N == c.N && code.M == c.M, 'KNM mismatch');
        assert(abs(code.rate() - c.rate) < 1e-12, 'rate mismatch');
        assert(code.n_punct == c.n_punct, 'n_punct mismatch');
        % encode the SAME deterministic message
        msg = double(c.msg(:));
        cw = code.encode(msg);
        cwRef = double(c.cw(:));
        assert(numel(cw) == numel(cwRef), 'cw length mismatch');
        assert(isequal(cw, cwRef), 'CODEWORD mismatch (bg%d ils%d Z%d mp%d)', ...
            c.bg, c.ils, c.Z, c.mp);
        % systematic + valid
        assert(isequal(cw(1:code.K), msg), 'not systematic');
        [echk, evar] = code.edges();
        tan = Tanner(echk, evar, code.M, code.N);
        assert(tan.syndrome_weight(cw) == 0, 'H*c ~= 0');
        assert(tan.syndrome_weight(cw) == c.syndrome_cw, 'syndrome(cw) mismatch');
        % decode the SAME deterministic LLR (min-sum on the realistic punctured LLR)
        llr = detLlrFromCw(cw, code.n_punct, code.N);
        hardMs = tan.decode(llr, 50, 'minsum', 0.8);
        assert(isequal(hardMs, double(c.hard_minsum(:))), ...
            'min-sum hard decision mismatch (bg%d ils%d Z%d mp%d)', c.bg, c.ils, c.Z, c.mp);
        % sum-product on a non-zero LLR (no punctured zeros -> no NaN corner -> bit-exact)
        llrNz = detLlrBase(cw, code.N);
        hardSp = tan.decode(llrNz, 50, 'sumproduct', 0.8);
        assert(isequal(hardSp, double(c.hard_sumprod_nz(:))), ...
            'sum-product hard decision mismatch (bg%d ils%d Z%d mp%d)', c.bg, c.ils, c.Z, c.mp);
        fprintf('PASS  bg%d ils%d Z%d mp%d : encode+decode bit-exact (K=%d N=%d R=%.4f)\n', ...
            c.bg, c.ils, c.Z, c.mp, code.K, code.N, code.rate());
    end
    fprintf('\nALL CROSS-VALIDATION CHECKS PASSED (MATLAB == Python, bit-for-bit)\n');
end

function llr = detLlrBase(cw, N)
%DETLLRBASE  Deterministic channel-like LLR (no zeroing), mirrors gen_reference.det_llr_base.
    idx = (0:N-1)';
    llr = 1.5 * (1.0 - 2.0 * cw) + 0.8 * cos(idx * 0.7);
    flip = mod(idx, 17) == 0;
    llr(flip) = -llr(flip);
end

function llr = detLlrFromCw(cw, n_punct, N)
%DETLLRFROMCW  Realistic LLR with punctured columns set to 0 (used for min-sum).
    llr = detLlrBase(cw, N);
    llr(1:n_punct) = 0.0;          % punctured columns 0,1 -> LLR 0
end
