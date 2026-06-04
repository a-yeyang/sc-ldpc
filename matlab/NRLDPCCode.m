classdef NRLDPCCode < handle
%NRLDPCCODE  A lifted 5G NR QC-LDPC component code (3GPP TS 38.212 base graphs).
%
%   Efficient QC-LDPC encoder exploiting the standardised parity structure
%
%       H_parity = [ C  0 ]      C : 4x4 block "core"      (first 4 rows / cols)
%                  [ E  I ]      I : identity accumulator  (rows >= 4)
%
%   Encoding a row-syndrome s (contribution of every non-parity variable to each
%   check row):
%       p_core = C^{-1} s_core            (one small GF(2) solve, precomputed)
%       p_i    = s_i XOR E_i p_core       for i >= 4 (pure accumulation)
%
%   The same solve_parity is reused by the spatially-coupled encoder.
%   Port of nr_ldpc.py::NRLDPCCode.

    properties
        bg; ils; Z; Kb; mb_full; B; mb; nb; mp
        K; N; M; n_punct
        core_inv          % 4Z x 4Z GF(2) inverse of the core block (0/1 double)
        E                 % cell over rows 4..mb-1: each is [coreRow(1-idx), shift] pairs
    end

    methods
        function obj = NRLDPCCode(bg, ils, Z, mp)
            if nargin < 4, mp = []; end
            obj.bg = bg; obj.ils = ils; obj.Z = Z;
            B = loadBaseMatrix(bg, ils, Z);
            obj.Kb = NRLDPCCode.KB(bg);
            obj.mb_full = size(B, 1);
            if isempty(mp), mp = obj.mb_full; end
            assert(mp >= 4 && mp <= obj.mb_full, ...
                'mp must be in [4, %d]', obj.mb_full);
            % keep first mp rows and first Kb+mp columns (info + first mp parity)
            obj.B = B(1:mp, 1:obj.Kb + mp);
            [obj.mb, obj.nb] = size(obj.B);
            obj.mp = mp;
            assert(obj.nb - obj.Kb == obj.mb);
            obj.K = obj.Kb * Z;          % info bits
            obj.N = obj.nb * Z;          % (rate-matched) codeword bits
            obj.M = obj.mb * Z;          % parity bits / checks
            obj.n_punct = 2 * Z;         % first two systematic columns punctured
            obj.build_parity_structure();
        end

        function build_parity_structure(obj)
            B = obj.B; Z = obj.Z; Kb = obj.Kb; mb = obj.mb;
            % 4Z x 4Z lifted core (rows 0..3, parity cols Kb..Kb+3)
            core = zeros(4 * Z, 4 * Z);
            for i = 0:3
                for j = 0:3
                    v = B(i + 1, Kb + j + 1);
                    if v >= 0
                        blk = circshift(eye(Z), v, 2);   % P^v : row r has a 1 at col (r+v) mod Z
                        core(i*Z + (1:Z), j*Z + (1:Z)) = blk;
                    end
                end
            end
            obj.core_inv = gf2inv(core);
            assert(~isempty(obj.core_inv), '5G NR core block must be invertible');
            % E block: for rows >= 4, which core parity cols (0..3) they touch
            obj.E = cell(1, max(0, mb - 4));
            for i = 4:(mb - 1)
                lst = zeros(0, 2);
                for c = 0:3
                    v = B(i + 1, Kb + c + 1);
                    if v >= 0
                        lst(end + 1, :) = [c + 1, v];   %#ok<AGROW>  c+1 = 1-indexed row of p
                    end
                end
                obj.E{i - 3} = lst;
            end
        end

        function p = solve_parity(obj, s)
            %SOLVE_PARITY  Given row syndromes s (mb x Z), return parity blocks p
            %   (mb x Z) with H_parity * p = s over GF(2).
            Z = obj.Z; mb = obj.mb;
            p = zeros(mb, Z);
            % core: p_core = C^{-1} s_core   (block-order vectorisation)
            s_core_vec = reshape(s(1:4, :).', [], 1);
            p_core_vec = mod(obj.core_inv * s_core_vec, 2);
            p(1:4, :) = reshape(p_core_vec, Z, 4).';
            % accumulator rows
            for idx = 1:(mb - 4)
                i = 4 + idx;                 % 1-indexed row of p / s
                acc = s(i, :);
                lst = obj.E{idx};
                for e = 1:size(lst, 1)
                    acc = xor(acc, shiftVec(p(lst(e, 1), :), lst(e, 2)));
                end
                p(i, :) = acc;
            end
        end

        function s = message_syndrome(obj, msg_blocks)
            %MESSAGE_SYNDROME  Row syndromes from the systematic bits only (mb x Z).
            Z = obj.Z; mb = obj.mb; Kb = obj.Kb; B = obj.B;
            s = zeros(mb, Z);
            for i = 1:mb
                for j = 1:Kb
                    v = B(i, j);
                    if v >= 0
                        s(i, :) = xor(s(i, :), shiftVec(msg_blocks(j, :), v));
                    end
                end
            end
        end

        function cw = encode(obj, msg)
            %ENCODE  msg: K bits -> codeword: N bits (systematic, first K are msg).
            Z = obj.Z; Kb = obj.Kb;
            msg = double(msg(:));
            msg_blocks = reshape(msg, Z, Kb).';      % Kb x Z, row j = info block j
            s = obj.message_syndrome(msg_blocks);
            p = obj.solve_parity(s);
            cw = [reshape(msg_blocks.', [], 1); reshape(p.', [], 1)];
        end

        function [echk, evar] = edges(obj)
            [rr, cc] = find(obj.B >= 0);             % 1-indexed
            v = obj.B(sub2ind(size(obj.B), rr, cc));
            [echk, evar] = edgesFromEntries(rr - 1, cc - 1, v, obj.Z);
        end

        function r = rate(obj)
            %RATE  Transmitted code rate accounting for the 2 punctured columns.
            r = obj.K / (obj.N - obj.n_punct);
        end
    end

    methods (Static)
        function k = KB(bg)
            switch bg
                case 1, k = 22;
                case 2, k = 10;
                otherwise, error('unknown base graph %d', bg);
            end
        end

        function mp = mp_for_rate(bg, rate)
            %MP_FOR_RATE  Parity rows mp giving (punctured) rate ~= Kb/(Kb+mp-2).
            Kb = NRLDPCCode.KB(bg);
            mp = max(4, round(Kb / rate - Kb + 2));
        end
    end
end
