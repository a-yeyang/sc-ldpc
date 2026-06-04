classdef Tanner < handle
%TANNER  Belief-propagation decoder for QC-LDPC / SC-LDPC Tanner graphs.
%
%   Built from a flat 1-indexed edge list (e_chk, e_var).  The graph is stored
%   in a padded [node x max_degree] layout so both the check-node and
%   variable-node updates are fully vectorised.  Two flooding-schedule decoders:
%     * normalised min-sum  (default, the 5G workhorse)
%     * sum-product (tanh)
%   A syndrome-based early stop is used.  Port of decoder.py::Tanner.

    properties (Constant)
        MSG_CAP = 1e2;   % clip LLR message magnitudes (degree-1 boundary checks -> +inf otherwise)
    end

    properties
        E; num_chk; num_var; e_chk; e_var
        C_tab; C_mask; C_safe; dc       % padded layout for check nodes
        V_tab; V_mask; V_safe; dv       % padded layout for variable nodes
    end

    methods
        function obj = Tanner(e_chk, e_var, num_chk, num_var)
            obj.e_chk = double(e_chk(:));
            obj.e_var = double(e_var(:));
            obj.E = numel(obj.e_chk);
            obj.num_chk = double(num_chk);
            obj.num_var = double(num_var);
            [obj.C_tab, obj.C_mask, obj.C_safe, obj.dc] = ...
                Tanner.buildSlots(obj.e_chk, obj.num_chk);
            [obj.V_tab, obj.V_mask, obj.V_safe, obj.dv] = ...
                Tanner.buildSlots(obj.e_var, obj.num_var);
        end

        function [hard, it] = decode(obj, llr_ch, max_iter, method, alpha)
            %DECODE  llr_ch: length num_var channel LLRs (>0 favours bit 0).
            %   Returns hard-decision bits (length num_var) and #iterations.
            if nargin < 3 || isempty(max_iter), max_iter = 50; end
            if nargin < 4 || isempty(method), method = 'minsum'; end
            if nargin < 5 || isempty(alpha), alpha = 0.8; end
            CAP = Tanner.MSG_CAP;
            llr_ch = double(llr_ch(:));
            m_vc = llr_ch(obj.e_var);             % var->chk messages (per edge)
            hard = double(llr_ch < 0);
            it = 0;
            for it = 1:max_iter
                % ---- check update ----
                if strcmp(method, 'minsum')
                    m_cv = obj.check_minsum(m_vc, alpha, CAP);
                else
                    m_cv = obj.check_sumproduct(m_vc, CAP);
                end
                % ---- variable update + decision ----
                gathered = m_cv(obj.V_safe);
                gathered(~obj.V_mask) = 0.0;
                sum_cv = sum(gathered, 2);
                total = llr_ch + sum_cv;
                hard = double(total < 0);
                % syndrome early stop
                syn = mod(accumarray(obj.e_chk, hard(obj.e_var), [obj.num_chk, 1]), 2);
                if ~any(syn)
                    break;
                end
                % outgoing var->chk = total - incoming (per edge), clipped
                new_vc = min(max(total - gathered, -CAP), CAP);
                m_vc(obj.V_tab(obj.V_mask)) = new_vc(obj.V_mask);
            end
        end

        function m_cv = check_minsum(obj, m_vc, alpha, CAP)
            v = m_vc(obj.C_safe);                 % numChk x maxdc
            sgn = sign(v);
            sgn(sgn == 0) = 1.0;
            sgn(~obj.C_mask) = 1.0;
            signprod = prod(sgn, 2);              % numChk x 1
            absval = abs(v);
            absval(~obj.C_mask) = Inf;
            [min1, arg] = min(absval, [], 2);     % smallest magnitude per check
            min1 = min(min1, CAP);
            lin = sub2ind(size(absval), (1:obj.num_chk)', arg);
            absval2 = absval;
            absval2(lin) = Inf;
            min2 = min(min(absval2, [], 2), CAP); % 2nd smallest (capped; +inf for degree-1)
            out_mag = repmat(min1, 1, size(v, 2));
            out_mag(lin) = min2;                  % at the argmin slot use the 2nd min
            out_sign = signprod .* sgn;           % exclude self: signprod*sign removes own sign
            out = alpha * out_sign .* out_mag;
            m_cv = zeros(obj.E, 1);
            m_cv(obj.C_tab(obj.C_mask)) = out(obj.C_mask);
        end

        function m_cv = check_sumproduct(obj, m_vc, CAP)
            t = tanh(min(max(m_vc, -30), 30) / 2.0);
            tt = t(obj.C_safe);
            tt(~obj.C_mask) = 1.0;
            prodt = prod(tt, 2);
            others = prodt ./ tt;                 % exclude self by division
            others(~isfinite(others)) = 0.0;      % guard 0/0 from zero-LLR edges (minsum is default)
            others = min(max(others, -1 + 1e-12), 1 - 1e-12);
            out = min(max(2.0 * atanh(others), -CAP), CAP);
            m_cv = zeros(obj.E, 1);
            m_cv(obj.C_tab(obj.C_mask)) = out(obj.C_mask);
        end

        function w = syndrome_weight(obj, hard)
            hard = double(hard(:));
            syn = mod(accumarray(obj.e_chk, hard(obj.e_var), [obj.num_chk, 1]), 2);
            w = sum(syn);
        end
    end

    methods (Static)
        function [tab, mask, safe, deg] = buildSlots(nodeOfEdge, numNodes)
            %BUILDSLOTS  Padded [numNodes x maxdeg] edge-index table (pad=0) + mask.
            E = numel(nodeOfEdge);
            deg = accumarray(nodeOfEdge, 1, [numNodes, 1]);
            maxd = max([deg; 0]);
            [srt, order] = sort(nodeOfEdge);          % group edges by node
            ptr = [0; cumsum(deg)];                   % length numNodes+1
            tab = zeros(numNodes, maxd);
            if E > 0
                slot = (1:E)' - ptr(srt);             % 1-indexed slot within node
                lin = sub2ind([numNodes, maxd], srt, slot);
                tab(lin) = order;
            end
            mask = tab > 0;
            safe = tab;
            safe(~mask) = 1;                          % valid index for padded slots
        end
    end
end
