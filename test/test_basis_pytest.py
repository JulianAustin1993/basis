import numpy as np
from basis import Monomial


class TestMonomial:
    def test__evaluate_basis(self):
        K = 8
        x = np.linspace(0,1,5)
        bs = Monomial((0,1), K)
        bs_eval = bs(x)
        assert np.allclose(bs_eval[:, 0], np.ones(len(x)))
        for i in np.arange(K):
            assert np.allclose(bs_eval[:, i], x**i)
        assert np.allclose(bs_eval.shape, (len(x),K))

    def test_penalty(self):
        K = 8
        bs = Monomial((0, 1), K)
        P = bs.penalty(1)
        assert np.allclose(P, P.T)
        assert np.allclose(P[0], np.zeros((K,K)))
        assert len(P) == K
        assert np.allclose(P[3],np.array([0.0, 1.0, 1.5, 1.8, 2.0, 2.142857, 2.25, 7/3]))

