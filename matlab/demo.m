function demo()
%DEMO  End-to-end demonstration of the 5G-NR-based SC-LDPC system (port of demo.py).
%   Builds the code, prints the coupled band structure, encodes a random message,
%   transmits over BPSK/AWGN, decodes with the full-graph and sliding-window
%   decoders, and visualises the SC "decoding wave".
    addpath(fileparts(mfilename('fullpath')));
    line = repmat('=', 1, 70);
    fprintf('%s\n', line);
    fprintf('5G NR  ->  Spatially-Coupled LDPC   (BPSK / AWGN)\n');
    fprintf('%s\n', line);

    comp = NRLDPCCode(2, 1, 24);
    fprintf('\nComponent code: 5G NR BG2, Z=%d  (K=%d, N=%d, rate=%.3f, %d punctured bits)\n', ...
        comp.Z, comp.K, comp.N, comp.rate(), comp.n_punct);

    sc = SCLDPCCode(comp, 2, 12, 0);
    fprintf('SC-LDPC: coupling memory w=%d, coupling length L=%d, seed=%d\n', sc.w, sc.L, sc.seed);
    fprintf('  info bits K=%d, transmitted N=%d, rate=%.3f (component rate %.3f - small termination loss)\n\n', ...
        sc.K, sc.N_tx, sc.rate(), comp.rate());
    showBand(sc);

    % ---- encode ----
    rng = RandStream('mt19937ar', 'Seed', 2024);
    [cw, info] = sc.encode(rng);
    tan = sc.full_tanner();
    syn = tan.syndrome_weight(cw);
    fprintf('\nEncoded one codeword. Parity-check syndrome weight = %d (0 => valid codeword)\n', syn);

    % ---- transmit + decode at a working SNR ----
    ebn0 = 1.0;
    sigma = ch.ebn0_to_sigma(ebn0, sc.rate());
    llr = sc.make_llr(cw, sigma, rng);

    hard_full = tan.decode(llr, 60, 'minsum', 0.8);
    ber_full = mean(sc.extract_info(hard_full) ~= info);
    fprintf('\n[Full-graph BP]   Eb/N0=%g dB   info-BER=%.2e   %s\n', ...
        ebn0, ber_full, recoveredStr(ber_full));

    hard_win = sc.decode_windowed(llr, 6, 30, 'minsum', 0.8);
    ber_win = mean(sc.extract_info(hard_win) ~= info);
    fprintf('[Windowed   BP]   W=6                 info-BER=%.2e   %s\n', ...
        ber_win, recoveredStr(ber_win));

    % ---- visualise the decoding wave near threshold ----
    fprintf('\nDecoding wave (per-position bit errors) just below threshold:\n');
    ebn0_low = 0.3;
    sigma = ch.ebn0_to_sigma(ebn0_low, sc.rate());
    llr = sc.make_llr(cw, sigma, rng);
    fprintf('  channel only (hard on LLR), Eb/N0=%g dB:\n', ebn0_low);
    hard0 = double(llr < 0);
    fprintf('   %s\n', fmtErrs(perPositionErrors(sc, hard0, cw)));
    for it = [2 5 10 30]
        hard = tan.decode(llr, it, 'minsum', 0.8);
        e = perPositionErrors(sc, hard, cw);
        tag = '';
        if it == 2, tag = '   (wave eats inward from both terminated ends)'; end
        fprintf('  after %2d BP iters: %s%s\n', it, fmtErrs(e), tag);
    end

    fprintf('\nDone.  Run `simulate(''component''|''sc''|''windowed'')` then `simulate(''plot'')` for BER curves.\n');
end

function showBand(sc)
    fprintf('Coupled base matrix  H_SC[i,t] = B_(i-t)   (rows = check positions i, cols = variable positions t)\n');
    hdr = '    ';
    for t = 0:sc.L - 1, hdr = [hdr, sprintf('%d', mod(t, 10))]; end %#ok<AGROW>
    fprintf('%s   <- variable position t\n', hdr);
    for i = 0:sc.L + sc.w - 1
        row = '';
        for t = 0:sc.L - 1
            d = i - t;
            if d >= 0 && d <= sc.w, row = [row 'B']; else, row = [row '.']; end %#ok<AGROW>
        end
        tag = '';
        if i >= sc.L, tag = '  <- termination checks'; end
        fprintf('i=%2d %s%s\n', i, row, tag);
    end
end

function errs = perPositionErrors(sc, cw_hat, cw)
    blk = sc.nb * sc.Z;
    errs = zeros(1, sc.L);
    for t = 0:sc.L - 1
        seg = t * blk + (1:blk);
        errs(t + 1) = sum(cw_hat(seg) ~= cw(seg));
    end
end

function s = fmtErrs(errs)
    s = strtrim(sprintf('%3d ', errs));
end

function s = recoveredStr(ber)
    if ber == 0, s = 'recovered'; else, s = 'errors'; end
end
