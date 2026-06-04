function simulate(cmd)
%SIMULATE  Monte-Carlo BER/FER over AWGN: 5G NR component vs SC-LDPC (port of
%   simulate.py).  Sub-commands (results written into the current folder):
%       simulate component   -> results_component.mat / .json
%       simulate sc          -> results_sc.mat / .json
%       simulate windowed    -> results_windowed.mat / .json
%       simulate plot        -> ber_curves.png + results.csv  (combines the above)
%   Each sweep uses an adaptive stop: it stops a point once target_ferr frame
%   errors are collected (with a minimum frame count) and skips lower SNRs once
%   a point is error-free.
    addpath(fileparts(mfilename('fullpath')));
    if nargin < 1, cmd = 'plot'; end
    Z = 24;   % lifting size used throughout
    switch cmd
        case 'component', runComponent(Z);
        case 'sc',        runSC(Z);
        case 'windowed',  runWindowed(Z);
        case 'plot',      runPlot();
        otherwise, error('unknown command %s', cmd);
    end
end

function res = runSweep(ed, ebn0_list, rate, n_frames, target_ferr, min_frames, label)
    rng = RandStream('mt19937ar', 'Seed', 12345);
    xs = []; bers = []; fers = [];
    fprintf('\n# %s  (rate=%.4f)\n', label, rate);
    for ei = 1:numel(ebn0_list)
        ebn0 = ebn0_list(ei);
        sigma = ch.ebn0_to_sigma(ebn0, rate);
        be = 0; bits = 0; fe = 0; nf = 0;
        t0 = tic;
        while nf < n_frames
            [e, b, f] = ed(sigma, rng);
            be = be + e; bits = bits + b; fe = fe + f; nf = nf + 1;
            if nf >= min_frames && fe >= target_ferr, break; end
        end
        ber = be / bits; fer = fe / nf;
        xs(end + 1) = ebn0; bers(end + 1) = ber; fers(end + 1) = fer; %#ok<AGROW>
        fprintf('  Eb/N0=%+.2fdB  BER=%.3e  FER=%.3e  (%d frames, %.1fs)\n', ...
            ebn0, ber, fer, nf, toc(t0));
        if fe == 0, break; end
    end
    res = struct('x', xs, 'y', bers, 'fer', fers, 'label', label, 'rate', rate);
end

function runComponent(Z)
    code = NRLDPCCode(2, 1, Z);
    [echk, evar] = code.edges();
    tan = Tanner(echk, evar, code.M, code.N);
    punct = false(code.N, 1); punct(1:code.n_punct) = true;
    tx = ~punct;
    ed = @(sigma, rng) edComponent(code, tan, tx, sigma, rng);
    cur = runSweep(ed, [0.6 0.9 1.2 1.5 1.8 2.1], code.rate(), 8000, 150, 300, ...
        sprintf('5G NR BG2 component (Z=%d, R=%.3f)', Z, code.rate()));
    saveResult('results_component', cur);
end

function [err, bits, fe] = edComponent(code, tan, tx, sigma, rng)
    msg = double(randi(rng, [0 1], code.K, 1));
    cw = code.encode(msg);
    y = ch.awgn(ch.bpsk(cw(tx)), sigma, rng);
    llr = zeros(code.N, 1); llr(tx) = ch.llr_awgn(y, sigma);
    hard = tan.decode(llr, 50, 'minsum', 0.8);
    err = sum(hard(1:code.K) ~= msg);
    bits = code.K; fe = double(err > 0);
end

function runSC(Z)
    sc = SCLDPCCode(NRLDPCCode(2, 1, Z), 2, 30, 0);
    tan = sc.full_tanner();
    ed = @(sigma, rng) edSC(sc, tan, sigma, rng);
    cur = runSweep(ed, [0.0 0.2 0.35 0.5 0.65 0.8], sc.rate(), 700, 50, 80, ...
        sprintf('SC-LDPC full-BP (L=%d, w=%d, R=%.3f)', sc.L, sc.w, sc.rate()));
    saveResult('results_sc', cur);
end

function [err, bits, fe] = edSC(sc, tan, sigma, rng)
    [cw, info] = sc.encode(rng);
    llr = sc.make_llr(cw, sigma, rng);
    hard = tan.decode(llr, 40, 'minsum', 0.8);
    err = sum(sc.extract_info(hard) ~= info);
    bits = numel(info); fe = double(err > 0);
end

function runWindowed(Z)
    sc = SCLDPCCode(NRLDPCCode(2, 1, Z), 2, 30, 0);
    W = 8;
    sc.window_layout(W);
    ed = @(sigma, rng) edWindowed(sc, W, sigma, rng);
    cur = runSweep(ed, [0.35 0.5 0.65 0.8], sc.rate(), 120, 30, 40, ...
        sprintf('SC-LDPC windowed (W=%d)', W));
    saveResult('results_windowed', cur);
end

function [err, bits, fe] = edWindowed(sc, W, sigma, rng)
    [cw, info] = sc.encode(rng);
    llr = sc.make_llr(cw, sigma, rng);
    hard = sc.decode_windowed(llr, W, 15, 'minsum', 0.8);
    err = sum(sc.extract_info(hard) ~= info);
    bits = numel(info); fe = double(err > 0);
end

function runPlot()
    names = {'results_component', 'results_sc', 'results_windowed'};
    curves = [];
    for k = 1:numel(names)
        fn = [names{k} '.mat'];
        if isfile(fn)
            S = load(fn, 'cur');
            curves = appendCurve(curves, S.cur);
        else
            fprintf('(missing %s, skipping)\n', fn);
        end
    end
    if isempty(curves), fprintf('nothing to plot\n'); return; end
    plotCurves(curves, 'YLabel', 'BER', ...
        'Title', '5G NR LDPC vs Spatially-Coupled LDPC (BPSK / AWGN)', ...
        'Path', 'ber_curves.png', 'CSV', 'results.csv');
end

function saveResult(name, cur)
    save([name '.mat'], 'cur');
    fid = fopen([name '.json'], 'w'); fprintf(fid, '%s', jsonencode(cur)); fclose(fid);
    fprintf('saved %s.mat / .json\n', name);
end

function curves = appendCurve(curves, cur)
    if isempty(curves), curves = cur; else, curves(end + 1) = cur; end
end
