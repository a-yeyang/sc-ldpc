function y = awgn(x, sigma, rng)
%CH.AWGN  Add white Gaussian noise of std-dev sigma to x.
%   `rng` is a RandStream (threaded explicitly, mirroring numpy's Generator) so
%   that message generation and noise share one reproducible stream.
    y = x + sigma * randn(rng, size(x));
end
