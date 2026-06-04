function llr = llr_awgn(y, sigma)
%CH.LLR_AWGN  Channel LLR for BPSK (0->+1):  L = 2y/sigma^2  (>0 favours bit 0).
    llr = 2.0 * y / (sigma * sigma);
end
