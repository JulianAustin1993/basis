import numpy as np
import pytest

from basis import Monomial, Exponential


class TestMonomial:
    def test__evaluate_basis(self):
        K = 8
        x = np.linspace(0, 1, 5)
        bs = Monomial((0, 1), K)
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


class TestExponential:
    def test_theta(self):
        K = 6
        domain = (0, 1)
        with pytest.raises(ValueError):
            Exponential(domain, K, (0, 1, 2, 3, 4))
        with pytest.raises(ValueError):
            Exponential(domain, K, (0, 1, 2, 2, 3, 4))
        assert np.allclose(Exponential(domain, K).theta, np.arange(K))

    def test__evaluate_basis(self):
        K = 7
        x = np.linspace(0, 10, 8)
        bs = Exponential((0, 10), K)
        bs_eval = bs(x)
        assert np.allclose(bs_eval[:, 0], np.ones(len(x)))
        for i in np.arange(K):
            assert np.allclose(bs_eval[:, i], np.exp(x * i))
        assert np.allclose(bs_eval.shape, (len(x), K))

    def test_penalty(self):
        K = 8
        bs = Exponential((0, 1), K)
        P = bs.penalty(2)
        assert np.allclose(P, P.T)
        assert np.allclose(P[0], np.zeros((K, K)))
        assert len(P) == K
        assert np.allclose(P[4], np.array([0.0,
                                           471.7221,
                                           4292.5738,
                                           22538.7393,
                                           95358.6556,
                                           360092.6190,
                                           1268666.8298,
                                           4267322.1004]))
