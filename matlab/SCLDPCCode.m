classdef SCLDPCCode < handle
%SCLDPCCODE  Spatially-coupled LDPC built on a 5G NR component code.
%
%   "Systematic edge spreading" (coupling memory w, coupling length L):
%   the 5G NR base matrix B = [B_sys | B_par] is split into w+1 components
%       B = B_0 + B_1 + ... + B_w
%   where only the systematic edges are spread (each systematic base entry is
%   randomly assigned to one component), while the whole parity part B_par is
%   kept in B_0.  The coupled parity-check matrix is the band-diagonal
%       H_SC[i,t] = B_(i-t),   0 <= i-t <= w,  i=0..L+w-1, t=0..L-1.
%   Keeping B_par entirely in B_0 lets every spatial position be encoded with
%   the exact 5G NR parity structure.  The chain is terminated by forcing the
%   systematic bits of the last w positions to zero.
%   Port of sc_ldpc.py::SCLDPCCode.
%
%   NOTE on reproducibility: the random edge-spreading uses a MATLAB RandStream
%   seeded with `seed`.  This produces a *statistically equivalent* but not
%   bit-identical assignment to numpy's Generator, so Monte-Carlo BER numbers
%   match Python in trend, not to the last digit.  All structural invariants
%   (H_SC*c = 0, systematic, rate, termination) are exactly reproduced.

    properties
        comp; w; L; Z; mb; nb; Kb; seed
        comps                 % 1x(w+1) cell of base matrices (-1 = no edge)
        sys_entries           % 1x(w+1) cell of structs: rows/cols (1-idx), shifts
        ent_i; ent_ri; ent_t; ent_cj; ent_v   % coupled entry table (0-indexed blocks)
        num_chk; num_var
        punct_mask; known_mask; info_mask; tx_mask
        K; N_tx
        winCache              % containers.Map  W -> windows cell
    end

    methods
        function obj = SCLDPCCode(component, w, L, seed)
            if nargin < 2, w = 2; end
            if nargin < 3, L = 40; end
            if nargin < 4, seed = 0; end
            obj.comp = component;
            obj.w = w; obj.L = L; obj.seed = seed;
            obj.Z = component.Z; obj.mb = component.mb;
            obj.nb = component.nb; obj.Kb = component.Kb;
            obj.winCache = containers.Map('KeyType', 'double', 'ValueType', 'any');
            obj.edge_spread(seed);
            obj.build_coupled_entries();
            obj.build_masks();
        end

        % ----------------------------- edge spreading ----------------------- %
        function edge_spread(obj, seed)
            B = obj.comp.B; Kb = obj.Kb; w = obj.w;
            [mb, nb] = size(B);
            rs = RandStream('mt19937ar', 'Seed', seed);
            comps = cell(1, w + 1);
            for a = 1:w + 1, comps{a} = -ones(mb, nb); end
            comps{1}(:, Kb + 1:end) = B(:, Kb + 1:end);          % parity -> component 0
            % systematic part -> spread over components 0..w
            [ri, cj] = find(B(:, 1:Kb) >= 0);                    % 1-indexed
            compIdx = randi(rs, w + 1, numel(ri), 1);            % 1..w+1
            for e = 1:numel(ri)
                comps{compIdx(e)}(ri(e), cj(e)) = B(ri(e), cj(e));
            end
            % sanity: spreading reconstructs B exactly
            merged = -ones(mb, nb);
            for a = 1:w + 1
                msk = comps{a} >= 0;
                merged(msk) = comps{a}(msk);
            end
            assert(isequal(merged, B), 'edge spreading does not reconstruct B');
            obj.comps = comps;
            % per-component systematic entries (1-indexed rows/cols, shifts)
            obj.sys_entries = cell(1, w + 1);
            for a = 1:w + 1
                [rr, cc] = find(comps{a}(:, 1:Kb) >= 0);
                vv = comps{a}(sub2ind(size(comps{a}), rr, cc));
                obj.sys_entries{a} = struct('rows', rr, 'cols', cc, 'shifts', vv);
            end
        end

        % --------------------- coupled entry table -------------------------- %
        function build_coupled_entries(obj)
            L = obj.L; w = obj.w; mb = obj.mb; nb = obj.nb;
            I = {}; RI = {}; T = {}; CJ = {}; V = {};
            for a = 0:w
                [r, c] = find(obj.comps{a + 1} >= 0);            % 1-indexed
                v = obj.comps{a + 1}(sub2ind([mb, nb], r, c));
                r0 = r - 1; c0 = c - 1;                          % 0-indexed block row/col
                for t = 0:L - 1
                    i = t + a;                                   % check block index
                    if i > L + w - 1, continue; end
                    n = numel(r0);
                    I{end+1} = i * ones(n, 1);   %#ok<AGROW>
                    RI{end+1} = r0;              %#ok<AGROW>
                    T{end+1} = t * ones(n, 1);   %#ok<AGROW>
                    CJ{end+1} = c0;              %#ok<AGROW>
                    V{end+1} = v;                %#ok<AGROW>
                end
            end
            obj.ent_i = cat(1, I{:});
            obj.ent_ri = cat(1, RI{:});
            obj.ent_t = cat(1, T{:});
            obj.ent_cj = cat(1, CJ{:});
            obj.ent_v = cat(1, V{:});
            obj.num_chk = (L + w) * mb * obj.Z;
            obj.num_var = L * nb * obj.Z;
        end

        function tan = full_tanner(obj)
            mb = obj.mb; nb = obj.nb;
            bigrow = obj.ent_i * mb + obj.ent_ri;
            bigcol = obj.ent_t * nb + obj.ent_cj;
            [echk, evar] = edgesFromEntries(bigrow, bigcol, obj.ent_v, obj.Z);
            tan = Tanner(echk, evar, obj.num_chk, obj.num_var);
        end

        function tan = window_tanner(obj, c_lo, c_hi, v_lo, v_hi)
            %WINDOW_TANNER  Sub-Tanner over check blocks [c_lo,c_hi], var blocks [v_lo,v_hi].
            mb = obj.mb; nb = obj.nb; Z = obj.Z;
            m = (obj.ent_i >= c_lo) & (obj.ent_i <= c_hi) & ...
                (obj.ent_t >= v_lo) & (obj.ent_t <= v_hi);
            bigrow = (obj.ent_i(m) - c_lo) * mb + obj.ent_ri(m);
            bigcol = (obj.ent_t(m) - v_lo) * nb + obj.ent_cj(m);
            nC = (c_hi - c_lo + 1) * mb;
            nV = (v_hi - v_lo + 1) * nb;
            [echk, evar] = edgesFromEntries(bigrow, bigcol, obj.ent_v(m), Z);
            tan = Tanner(echk, evar, nC * Z, nV * Z);
        end

        % --------------------- punctured / known masks ---------------------- %
        function build_masks(obj)
            L = obj.L; w = obj.w; Kb = obj.Kb; nb = obj.nb; Z = obj.Z;
            N = L * nb * Z;
            punct = false(N, 1); known = false(N, 1); info = false(N, 1);
            for t = 0:L - 1
                base = t * nb * Z;
                punct(base + (1:2 * Z)) = true;                  % cols 0,1
                if t >= L - w                                    % terminated positions
                    known(base + (1:Kb * Z)) = true;
                else                                             % information positions
                    info(base + (1:Kb * Z)) = true;
                end
            end
            known = known & ~punct;
            obj.punct_mask = punct;
            obj.known_mask = known;
            obj.info_mask = info;
            obj.tx_mask = ~(punct | known);
            obj.K = sum(info);
            obj.N_tx = sum(obj.tx_mask);
        end

        function r = rate(obj)
            r = obj.K / obj.N_tx;
        end

        function n = n_info_positions(obj)
            n = obj.L - obj.w;
        end

        % ------------------------------ encoder ----------------------------- %
        function [cw, info] = encode(obj, rng_or_msg)
            %ENCODE  Accepts a RandStream (random info) or a length-K info bit
            %   array.  Returns codeword bits [N] and info bits [K].
            L = obj.L; w = obj.w; Kb = obj.Kb; mb = obj.mb; nb = obj.nb; Z = obj.Z;
            if isa(rng_or_msg, 'RandStream')
                info = double(randi(rng_or_msg, [0 1], obj.K, 1));
            else
                info = double(rng_or_msg(:));
                assert(numel(info) == obj.K, 'info length must be K=%d', obj.K);
            end
            cw = zeros(L * nb * Z, 1);
            sys = cell(1, L);
            % place info into the first L-w positions, terminate the rest to zero
            for p = 1:(L - w)
                block = info((p - 1) * Kb * Z + 1 : p * Kb * Z);
                sys{p} = reshape(block, Z, Kb).';                % Kb x Z
            end
            for p = (L - w + 1):L
                sys{p} = zeros(Kb, Z);
            end
            % encode position by position
            for i = 0:L - 1
                S = zeros(mb, Z);
                for a = 0:w
                    t = i - a;
                    if t < 0, continue; end
                    se = obj.sys_entries{a + 1};
                    for e = 1:numel(se.rows)
                        S(se.rows(e), :) = xor(S(se.rows(e), :), ...
                            shiftVec(sys{t + 1}(se.cols(e), :), se.shifts(e)));
                    end
                end
                p = obj.comp.solve_parity(S);                    % 5G recursive parity solve
                base = i * nb * Z;
                cw(base + (1:Kb * Z)) = reshape(sys{i + 1}.', [], 1);
                cw(base + Kb * Z + (1:(nb - Kb) * Z)) = reshape(p.', [], 1);
            end
        end

        % ------------------------- channel helpers -------------------------- %
        function llr = make_llr(obj, cw, sigma, rng, large)
            %MAKE_LLR  Modulate the transmitted bits over BPSK+AWGN, return full LLR.
            if nargin < 5, large = 30.0; end
            tx = ch.bpsk(cw(obj.tx_mask));
            y = ch.awgn(tx, sigma, rng);
            llr = zeros(obj.num_var, 1);
            llr(obj.tx_mask) = ch.llr_awgn(y, sigma);
            llr(obj.known_mask) = large;        % terminated bits are known to be 0
            % punctured bits keep llr = 0
        end

        function bits = extract_info(obj, hard)
            bits = hard(obj.info_mask);
        end

        % --------------------- sliding-window decoder ----------------------- %
        function windows = window_layout(obj, W)
            %WINDOW_LAYOUT  Build & cache per-target window sub-Tanners.
            if obj.winCache.isKey(W)
                windows = obj.winCache(W);
                return;
            end
            L = obj.L; w = obj.w;
            windows = cell(1, L);
            for t = 0:L - 1
                v_lo = max(0, t - w);
                v_hi = min(t + W - 1, L - 1);
                c_lo = t;
                if v_hi < L - 1
                    c_hi = v_hi;
                else
                    c_hi = L + w - 1;        % include termination checks at the tail
                end
                tan = obj.window_tanner(c_lo, c_hi, v_lo, v_hi);
                windows{t + 1} = struct('tan', tan, 'v_lo', v_lo, 'v_hi', v_hi);
            end
            obj.winCache(W) = windows;
        end

        function dec_bits = decode_windowed(obj, llr_full, W, max_iter, method, alpha, large)
            %DECODE_WINDOWED  Sliding-window BP.  A window of W spatial positions
            %   slides from t=0 to L-1; after BP only the leftmost ("target")
            %   position is finalised and output, then the window advances by one.
            %   Memory/latency scale with W instead of L.
            if nargin < 3 || isempty(W), W = 6; end
            if nargin < 4 || isempty(max_iter), max_iter = 50; end
            if nargin < 5 || isempty(method), method = 'minsum'; end
            if nargin < 6 || isempty(alpha), alpha = 0.8; end
            if nargin < 7 || isempty(large), large = 30.0; end
            L = obj.L; nb = obj.nb; Z = obj.Z;
            blk = nb * Z;
            dec_bits = zeros(obj.num_var, 1);
            windows = obj.window_layout(W);
            for t = 0:L - 1
                win = windows{t + 1};
                tan = win.tan; v_lo = win.v_lo; v_hi = win.v_hi;
                nloc = (v_hi - v_lo + 1) * blk;
                llr_loc = zeros(nloc, 1);
                for tp = v_lo:v_hi
                    g0 = tp * blk;
                    l0 = (tp - v_lo) * blk;
                    if tp < t        % already decided -> known
                        decblk = dec_bits(g0 + (1:blk));
                        llr_loc(l0 + (1:blk)) = large * (1 - 2 * decblk);
                    else
                        llr_loc(l0 + (1:blk)) = llr_full(g0 + (1:blk));
                    end
                end
                hard_loc = tan.decode(llr_loc, max_iter, method, alpha);
                l0 = (t - v_lo) * blk;
                dec_bits(t * blk + (1:blk)) = hard_loc(l0 + (1:blk));
            end
        end
    end
end
