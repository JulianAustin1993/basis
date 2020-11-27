import numpy as np
import pytest

from basis import Monomial, Exponential, Fourier, Bspline


class TestMonomial:
    def test__evaluate_basis(self):
        K = 8
        x = np.linspace(0, 1, 5)
        bs = Monomial((0, 1), K)
        bs_eval = bs(x)
        assert np.allclose(bs_eval[:, 0], np.ones(len(x)))
        for i in np.arange(K):
            assert np.allclose(bs_eval[:, i], x ** i)
        assert np.allclose(bs_eval.shape, (len(x), K))

    def test_penalty(self):
        K = 8
        bs = Monomial((0, 1), K)
        P = bs.penalty(1)
        assert np.allclose(P, P.T)
        assert np.allclose(P[0], np.zeros((K, K)))
        assert len(P) == K
        assert np.allclose(P[3], np.array([0.0, 1.0, 1.5, 1.8, 2.0, 2.142857, 2.25, 7 / 3]))


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


class TestFourier:
    def test_K(self):
        K = 6
        domain = (0, 1)
        period = 1.0
        with pytest.raises(ValueError):
            Fourier(domain, K, period)
        assert Fourier(domain, 5, period).K == 5

    def test_period(self):
        K = 5
        domain = (0, 1)
        period = -1.0
        with pytest.raises(ValueError):
            Fourier(domain, K, period)
        assert Fourier(domain, K, 1.0).period == 1.0

    def test__evaluate_basis(self):
        K = 5
        domain = (0, 2)
        period = 2.0
        bs = Fourier(domain, K, period)
        x = np.linspace(*domain, 5)
        assert np.allclose(bs(x, 1)[:, 2], np.array([0.0,
                                                     -3.141593,
                                                     0.0,
                                                     3.141593,
                                                     0.0]))

    def test_penalty_analytic(self):
        K = 5
        domain = (-1, 1)
        period = 2.0
        bs = Fourier(domain, K, period)
        assert np.allclose(np.diag(bs.penalty(2)), np.array([0.0, 97.40909, 97.40909, 1558.545, 1558.545]))

    def test_penalty_numeric(self):
        K = 5
        domain = (-1, 1)
        period = 1.999
        bs = Fourier(domain, K, period)
        bs2 = Fourier(domain, K, 2)
        assert np.sum((bs.penalty(1, 12) - bs2.penalty(1)) ** 2) < 0.1


class TestBspline:
    def test_knots(self):
        bs = Bspline((-1, 1), 8, order=3)
        assert np.allclose(bs.knots, np.array([-1, -1, -1, -2 / 3,
                                               -1 / 3, 0, 1 / 3,
                                               2 / 3, 1, 1, 1]))
        with pytest.raises(ValueError):
            bs.knots = np.ones(8)

    def test_order(self):
        with pytest.raises(ValueError):
            Bspline((-1, 1), 8, order='foo')

    def test__evaluate_basis(self):
        bs = Bspline((-1, 1), 8, order=3)
        x = np.linspace(-1, 1, 9)
        assert np.allclose(bs(x)[:, 2], np.array([0, 0.28125, 0.75, 0.28125, 0, 0, 0, 0, 0]))

    def test_penalty(self):
        bs = Bspline((-1, 1), 8, order=4)
        assert np.allclose(bs.penalty(2)[:, 2], np.array([54.6875, -105.46875, 70.3125, -20.83333333,
                                                          -1.302083, 2.604167, 0, 0]), atol=1e-2)
