function runTests()
%RUNTESTS  Fast self-checks for the SC-LDPC implementation (port of tests.py).
%   Run:  matlab -batch "runTests"   (from the matlab/ folder, or addpath it).
    addpath(fileparts(mfilename('fullpath')));
    test_component_encode();
    test_sc_encode_valid();
    test_sc_rate();
    test_decode_highsnr();
    fprintf('\nAll tests passed.\n');
end

function test_component_encode()
    rng = RandStream('mt19937ar', 'Seed', 0);
    configs = {[2 1 24], [1 1 24], [2 0 16], [2 6 52]};
    for k = 1:numel(configs)
        cfg = configs{k};
        code = NRLDPCCode(cfg(1), cfg(2), cfg(3));
        [echk, evar] = code.edges();
        tan = Tanner(echk, evar, code.M, code.N);
        for it = 1:10
            msg = double(randi(rng, [0 1], code.K, 1));
            c = code.encode(msg);
            assert(tan.syndrome_weight(c) == 0, 'H*c != 0');
            assert(isequal(c(1:code.K), msg), 'not systematic');
        end
    end
    fprintf('PASS  component encoder: H*c=0 and systematic (BG1/BG2, several Z)\n');
end

function test_sc_encode_valid()
    rng = RandStream('mt19937ar', 'Seed', 1);
    sc = SCLDPCCode(NRLDPCCode(2, 1, 24), 2, 20, 0);
    tan = sc.full_tanner();
    for it = 1:5
        [cw, ~] = sc.encode(rng);
        assert(tan.syndrome_weight(cw) == 0, 'H_SC*c != 0');
    end
    fprintf('PASS  SC encoder: H_SC*c=0  (terminated, systematic edge spreading)\n');
end

function test_sc_rate()
    sc = SCLDPCCode(NRLDPCCode(2, 1, 24), 2, 30, 0);
    assert(sc.K == (sc.L - sc.w) * sc.Kb * sc.Z, 'K formula');
    assert(0.18 < sc.rate() && sc.rate() < 0.20 && 0.20 < sc.comp.rate() + 1e-9, 'rate bounds');
    fprintf('PASS  SC rate=%.4f < component rate=%.4f (termination loss)\n', ...
        sc.rate(), sc.comp.rate());
end

function test_decode_highsnr()
    rng = RandStream('mt19937ar', 'Seed', 2);
    sc = SCLDPCCode(NRLDPCCode(2, 1, 24), 2, 20, 0);
    tan = sc.full_tanner();
    sigma = ch.ebn0_to_sigma(2.0, sc.rate());
    full_ok = true; win_ok = true;
    for it = 1:4
        [cw, info] = sc.encode(rng);
        llr = sc.make_llr(cw, sigma, rng);
        full_ok = full_ok && isequal(sc.extract_info(tan.decode(llr, 60)), info);
        win_ok = win_ok && isequal(sc.extract_info(sc.decode_windowed(llr, 6, 30)), info);
    end
    assert(full_ok && win_ok, 'decoding failed at 2 dB');
    fprintf('PASS  full-graph & windowed decoders recover the message at 2 dB\n');
end
