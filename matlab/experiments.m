function experiments(cmd, varargin)
%EXPERIMENTS  Parametric SC-LDPC experiments over AWGN/BPSK (port of experiments.py).
%   Sweeps the four design knobs (run them separately):
%       experiments L         % coupling length  L
%       experiments w         % coupling depth   w
%       experiments rate      % 5G NR code rate  R  (BG2 truncation)
%       experiments rate_high % high code rates  R  (BG1 truncation)
%       experiments len       % code length      N  (lifting size Z)
%       experiments chain 300 % very long chain, sliding-window decoded
%       experiments chain_plot
%       experiments chain_high 0.8 [Ls...]
%       experiments plot      % re-plot L/w/rate/len from saved results
%   Each writes results_exp_<name>.mat/.json and exp_<name>.png.
    addpath(fileparts(mfilename('fullpath')));
    if nargin < 1, cmd = 'plot'; end
    switch cmd
        case 'L', exp_L();
        case 'w', exp_w();
        case 'rate', exp_rate();
        case 'rate_high', exp_rate_high();
        case 'len', exp_len();
        case 'chain', exp_chain(str2double(varargin{1}));
        case 'chain_plot', plot_chain();
        case 'chain_high'
            target = str2double(varargin{1});
            if numel(varargin) > 1
                Ls = cellfun(@str2double, varargin(2:end));
            else
                Ls = [30 100 200 300];
            end
            exp_chain_high(target, Ls);
        case 'chain_high_plot', plot_chain_high();
        case 'plot', run_plot();
        otherwise, error('unknown command %s', cmd);
    end
end

% --------------------------------------------------------------------------- %
function mp = best_mp_for_rate(bg, target)
    Kb = NRLDPCCode.KB(bg);
    mbFull = mbFullOf(bg);
    mps = 4:mbFull;
    [~, k] = min(abs(Kb ./ (Kb + mps - 2) - target));
    mp = mps(k);
end

function m = mbFullOf(bg)
    if bg == 1, m = 46; else, m = 42; end
end

function res = runSweep(ed, ebn0_list, rate, label, n_frames, target_ferr, min_frames, target_berr)
    if nargin < 5 || isempty(n_frames), n_frames = 130; end
    if nargin < 6 || isempty(target_ferr), target_ferr = 40; end
    if nargin < 7 || isempty(min_frames), min_frames = 50; end
    if nargin < 8, target_berr = []; end
    rng = RandStream('mt19937ar', 'Seed', 2025);
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
            if nf >= min_frames && (fe >= target_ferr || ...
                    (~isempty(target_berr) && be >= target_berr))
                break;
            end
        end
        ber = be / bits; fer = fe / nf;
        xs(end + 1) = ebn0; bers(end + 1) = ber; fers(end + 1) = fer; %#ok<AGROW>
        fprintf('  Eb/N0=%+.2fdB  BER=%.3e  FER=%.3e  (%d fr, %.1fs)\n', ...
            ebn0, ber, fer, nf, toc(t0));
        if fe == 0, break; end
    end
    res = struct('x', xs, 'y', bers, 'fer', fers, 'label', label, 'rate', rate);
end

function res = sc_curve(bg, ils, Z, mp, w, L, ebn0_list, label, max_iter, varargin)
    if nargin < 9 || isempty(max_iter), max_iter = 40; end
    comp = NRLDPCCode(bg, ils, Z, mp);
    sc = SCLDPCCode(comp, w, L, 0);
    tan = sc.full_tanner();
    ed = @(sigma, rng) edSC(sc, tan, max_iter, sigma, rng);
    res = runSweep(ed, ebn0_list, sc.rate(), label, varargin{:});
end

function [err, bits, fe] = edSC(sc, tan, max_iter, sigma, rng)
    [cw, info] = sc.encode(rng);
    llr = sc.make_llr(cw, sigma, rng);
    hard = tan.decode(llr, max_iter, 'minsum', 0.8);
    err = sum(sc.extract_info(hard) ~= info);
    bits = numel(info); fe = double(err > 0);
end

function res = comp_curve(bg, ils, Z, mp, ebn0_list, label, max_iter, varargin)
    if nargin < 7 || isempty(max_iter), max_iter = 50; end
    code = NRLDPCCode(bg, ils, Z, mp);
    [echk, evar] = code.edges();
    tan = Tanner(echk, evar, code.M, code.N);
    punct = false(code.N, 1); punct(1:code.n_punct) = true;
    tx = ~punct;
    ed = @(sigma, rng) edComp(code, tan, tx, max_iter, sigma, rng);
    res = runSweep(ed, ebn0_list, code.rate(), label, varargin{:});
end

function [err, bits, fe] = edComp(code, tan, tx, max_iter, sigma, rng)
    msg = double(randi(rng, [0 1], code.K, 1));
    cw = code.encode(msg);
    y = ch.awgn(ch.bpsk(cw(tx)), sigma, rng);
    llr = zeros(code.N, 1); llr(tx) = ch.llr_awgn(y, sigma);
    hard = tan.decode(llr, max_iter, 'minsum', 0.8);
    err = sum(hard(1:code.K) ~= msg);
    bits = code.K; fe = double(err > 0);
end

function res = sc_window_curve(bg, ils, Z, mp, w, L, W, ebn0_list, label, max_iter, varargin)
    if nargin < 10 || isempty(max_iter), max_iter = 12; end
    comp = NRLDPCCode(bg, ils, Z, mp);
    sc = SCLDPCCode(comp, w, L, 0);
    sc.window_layout(W);
    ed = @(sigma, rng) edWin(sc, W, max_iter, sigma, rng);
    res = runSweep(ed, ebn0_list, sc.rate(), label, varargin{:});
end

function [err, bits, fe] = edWin(sc, W, max_iter, sigma, rng)
    [cw, info] = sc.encode(rng);
    llr = sc.make_llr(cw, sigma, rng);
    hard = sc.decode_windowed(llr, W, max_iter, 'minsum', 0.8);
    err = sum(sc.extract_info(hard) ~= info);
    bits = numel(info); fe = double(err > 0);
end

% --------------------------------------------------------------------------- %
function exp_L()
    Z = 16;
    curves = comp_curve(2, 0, Z, [], [0.6 0.9 1.2 1.5 1.8 2.1], ...
        sprintf('component BG2 Z=%d (R=0.20)', Z));
    for L = [8 16 40]
        c = sc_curve(2, 0, Z, [], 2, L, [0.0 0.3 0.5 0.7 0.9 1.1], sprintf('SC w=2 L=%d', L));
        curves(end + 1) = c; %#ok<AGROW>
    end
    saveExp('L', curves, 'Effect of coupling length L (BG2, Z=16, w=2)');
end

function exp_w()
    Z = 16; L = 30;
    curves = comp_curve(2, 0, Z, [], [0.6 0.9 1.2 1.5 1.8 2.1], ...
        sprintf('component BG2 Z=%d (R=0.20)', Z));
    for w = [1 2 4]
        c = sc_curve(2, 0, Z, [], w, L, [0.0 0.3 0.5 0.7 0.9 1.1], sprintf('SC L=%d w=%d', L, w));
        curves(end + 1) = c; %#ok<AGROW>
    end
    saveExp('w', curves, 'Effect of coupling depth w (BG2, Z=16, L=30)');
end

function exp_rate()
    Z = 24; w = 2; L = 30;
    targets = [0.20 0.33 0.50];
    grids = {[0.4 0.7 1.0 1.3 1.6 1.9], [1.0 1.3 1.6 1.9 2.2 2.5], [1.6 1.9 2.2 2.5 2.8 3.1]};
    curves = [];
    for r = 1:numel(targets)
        R = targets(r); ebn0 = grids{r};
        mp = NRLDPCCode.mp_for_rate(2, R);
        c1 = comp_curve(2, 1, Z, mp, ebn0, sprintf('component R=%.2f', R));
        c2 = sc_curve(2, 1, Z, mp, w, L, ebn0 - 0.6, sprintf('SC R=%.2f', R));
        curves = catCurves(curves, c1, c2);
    end
    saveExp('rate', curves, 'Effect of code rate (5G NR rate matching, Z=24, w=2, L=30)');
end

function exp_rate_high()
    bg = 1; Z = 24; w = 2; L = 30;
    targets = [0.6 0.7 0.8 0.9];
    sc_grids = {[0.8 1.2 1.6 2.0 2.4], [1.4 1.8 2.2 2.6 3.0], ...
                [2.0 2.4 2.8 3.2 3.6], [3.0 3.4 3.8 4.2 4.6 5.0]};
    comp_grids = {[1.4 1.8 2.2 2.6 3.0], [2.0 2.4 2.8 3.2 3.6], ...
                  [2.6 3.0 3.4 3.8 4.2], [3.8 4.2 4.6 5.0 5.4 5.8]};
    curves = [];
    for r = 1:numel(targets)
        mp = best_mp_for_rate(bg, targets(r));
        Rc = NRLDPCCode(bg, 1, Z, mp).rate();
        c1 = comp_curve(bg, 1, Z, mp, comp_grids{r}, sprintf('component R=%.2f', Rc), [], ...
            4000, 120, 200);
        c2 = sc_curve(bg, 1, Z, mp, w, L, sc_grids{r}, sprintf('SC R=%.2f', Rc), [], ...
            150, 40, 40);
        curves = catCurves(curves, c1, c2);
    end
    saveExp('rate_high', curves, 'High code rates (5G NR BG1 rate matching, Z=24, w=2, L=30)');
end

function exp_len()
    w = 2; L = 24;
    curves = comp_curve(2, 0, 32, [], [0.6 0.9 1.2 1.5 1.8 2.1], 'component BG2 Z=32 (R=0.20)');
    for Z = [16 32 64]
        c = sc_curve(2, 0, Z, [], w, L, [0.2 0.4 0.6 0.8 1.0], ...
            sprintf('SC Z=%d (N/pos=%d)', Z, 52 * Z));
        curves(end + 1) = c; %#ok<AGROW>
    end
    saveExp('len', curves, 'Effect of code length / lifting size Z (BG2, w=2, L=24)');
end

function exp_chain(L)
    Z = 16; w = 2; W = 6;
    ebn0 = [0.4 0.7 0.9 1.0 1.1 1.3 1.5];
    cur = sc_window_curve(2, 0, Z, [], w, L, W, ebn0, ...
        sprintf('SC L=%d (window W=%d)', L, W), 12, 40, 1e9, 12, 400);
    saveJson(sprintf('results_exp_chain_%d', L), cur);
    fprintf('\nsaved results_exp_chain_%d\n', L);
end

function plot_chain()
    curves = [];
    for L = [30 100 200 300]
        fn = sprintf('results_exp_chain_%d.mat', L);
        if isfile(fn), S = load(fn, 'cur'); curves = catCurves(curves, S.cur); end
    end
    if ~isempty(curves)
        plotCurves(curves, 'YLabel', 'BER', 'Path', 'exp_chain.png', ...
            'Title', 'Threshold saturation: coupling length L (BG2 Z=16, w=2, windowed)');
    end
end

function exp_chain_high(target, Ls)
    Z = 24; w = 2; W = 6;
    gridMap = containers.Map({0.7, 0.8, 0.9}, ...
        {[1.8 2.2 2.5 2.7 2.9 3.1 3.4], [2.2 2.6 2.9 3.1 3.3 3.5 3.8], [3.2 3.6 3.9 4.1 4.3 4.6 4.9]});
    mp = best_mp_for_rate(1, target);
    grid = gridMap(target);
    Rc = NRLDPCCode(1, 1, Z, mp).rate();
    for L = Ls
        cur = sc_window_curve(1, 1, Z, mp, w, L, W, grid, ...
            sprintf('L=%d (R=%.2f)', L, Rc), 12, 40, 1e9, 12, 400);
        saveJson(sprintf('results_exp_chain_high_%d_%d', round(target * 100), L), cur);
        fprintf('\nsaved chain_high R~%.1f L=%d\n', target, L);
    end
end

function plot_chain_high()
    for target = [0.7 0.8 0.9]
        R2 = round(target * 100);
        curves = [];
        for L = [30 100 200 300]
            fn = sprintf('results_exp_chain_high_%d_%d.mat', R2, L);
            if isfile(fn), S = load(fn, 'cur'); curves = catCurves(curves, S.cur); end
        end
        if ~isempty(curves)
            Rc = NRLDPCCode(1, 1, 24, best_mp_for_rate(1, target)).rate();
            plotCurves(curves, 'YLabel', 'BER', 'Path', sprintf('exp_chain_high_%d.png', R2), ...
                'Title', sprintf('Long chains at high rate R=%.2f (BG1 Z=24, w=2, windowed)', Rc));
        end
    end
end

function run_plot()
    for name = {'L', 'w', 'rate', 'len'}
        fn = sprintf('results_exp_%s.mat', name{1});
        if isfile(fn)
            S = load(fn, 'curves');
            plotCurves(S.curves, 'YLabel', 'BER', 'Path', sprintf('exp_%s.png', name{1}), ...
                'Title', sprintf('experiment: %s', name{1}));
        end
    end
end

% --------------------------------------------------------------------------- %
function saveExp(name, curves, ttl)
    save(sprintf('results_exp_%s.mat', name), 'curves');
    fid = fopen(sprintf('results_exp_%s.json', name), 'w');
    fprintf(fid, '%s', jsonencode(curves)); fclose(fid);
    plotCurves(curves, 'YLabel', 'BER', 'Title', ttl, 'Path', sprintf('exp_%s.png', name));
    fprintf('\nwrote results_exp_%s.mat/.json and exp_%s.png\n', name, name);
end

function saveJson(name, cur)
    save([name '.mat'], 'cur');
    fid = fopen([name '.json'], 'w'); fprintf(fid, '%s', jsonencode(cur)); fclose(fid);
end

function curves = catCurves(curves, varargin)
    for k = 1:numel(varargin)
        if isempty(curves), curves = varargin{k}; else, curves(end + 1) = varargin{k}; end
    end
end
