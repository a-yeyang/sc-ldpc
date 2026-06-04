function x = bpsk(bits)
%CH.BPSK  Map bit 0 -> +1, bit 1 -> -1.
    x = 1.0 - 2.0 * double(bits);
end
