function y = shiftVec(x, v)
%SHIFTVEC  Apply the QC circulant P^v to a length-Z row vector.
%   Mirrors the Python  shift_vec(x, v) = np.roll(x, -v), i.e.
%       (P^v x)[r] = x[(r+v) mod Z].
%   In MATLAB (1-indexed):  y(r) = x(mod(r-1+v, Z)+1)  ==  circshift(x, -v).
%   x is a GF(2) row vector (uint8/double 0/1); the same convention is used by
%   both the encoder (syndrome accumulation) and the lifted Tanner graph, so
%   H*c = 0 holds consistently.
    y = circshift(x, -double(v));
end
