function inv = gf2inv(M)
%GF2INV  Inverse of a square binary matrix over GF(2) (Gauss-Jordan).
%   Returns a 0/1 double matrix, or [] if M is singular over GF(2).
%   Port of nr_ldpc.gf2_inv.
    n = size(M, 1);
    A = [mod(double(M), 2) ~= 0, logical(eye(n))];   % n x 2n augmented, logical
    r = 1;
    for c = 1:n
        piv = find(A(r:n, c), 1, 'first');
        if isempty(piv)
            continue;
        end
        p = r + piv - 1;
        if p ~= r
            A([r, p], :) = A([p, r], :);
        end
        sel = A(:, c);
        sel(r) = false;                                  % all other rows with a 1 in col c
        A(sel, :) = xor(A(sel, :), A(r, :));             % implicit expansion over rows
        r = r + 1;
        if r > n
            break;
        end
    end
    if r > n
        inv = double(A(:, n+1:end));
    else
        inv = [];
    end
end
