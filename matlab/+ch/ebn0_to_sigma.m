function sigma = ebn0_to_sigma(ebn0_db, rate)
%CH.EBN0_TO_SIGMA  Noise std-dev for BPSK (unit symbol energy Es=1) at Eb/N0 [dB].
%   Es/N0 = R * Eb/N0,  N0 = 1/(Es/N0),  sigma = sqrt(N0/2).
    ebn0 = 10.0 .^ (ebn0_db / 10.0);
    esn0 = rate .* ebn0;          % Es/N0 = R * Eb/N0
    n0 = 1.0 ./ esn0;             % Es = 1
    sigma = sqrt(n0 / 2.0);
end
