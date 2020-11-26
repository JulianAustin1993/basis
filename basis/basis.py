from abc import ABC, abstractmethod
import numpy as np


def default_knots(m, K, domain):
    """Calculate default knot placement for B-spline basis system with `N` basis functions of order `m` across `domain`.

    Args:
        m (int): The order of B-spline functions.

        K (int): The number of basis functions in the basis system.

        domain (tuple): The domain over which to place the knots specified by the lower and upper bound of the system.

    Returns:
        (np.ndarray): The full knot vector of the B-spline basis system.

    """
    L = K - m
    tau = np.linspace(*domain, L+2)
    return np.pad(tau, (m-2, m-2), 'edge')


class Basis(ABC):
    """
    Representation of univariate basis system.

    Attributes:
        domain (tuple): The domain over which the basis system covers.

        K (int): The number of basis functions to use in the basis system.

    """

    def __init__(self, domain, K):
        """Inits Basis class.

        Args:
            domain (tuple):  The domain of the basis system specified by the lower and upper bound of the system.

            K (int): Number of basis functions to use in the basis system.

        """
        self.domain = domain
        self.K = K

    @property
    def domain(self):
        """Getter for the domain attribute.

        """
        return self.__domain

    @domain.setter
    def domain(self, domain):
        """Setter for domain attribute.

        Args:
            domain (tuple):  The domain of the basis system specified by the lower and upper bound of the system.

        Raises:
             ValueError: If domain not length 2.

        """
        if len(domain) != 2:
            raise ValueError("domain must be of length 2.")
        self.__domain = domain

    @property
    def K(self):
        """Getter for the K attribute."""
        return self.__K

    @K.setter
    def K(self, K):
        """Setter for the K attribute.

        Args:
            K (int): Number of basis functions to use in the basis system.

        Raises:
             ValueError if K is not an integer and cannot be converted to one.

        """
        self.__K = int(K)

    @abstractmethod
    def _evaluate_basis(self, x, q):
        """Evaluate the qth derivative of all basis functions at locations x.

        Args:
            x (np.ndarray): Locations to evaluate basis function at.

            q (int): The order of the derivative to take of the basis functions.

        Returns:
            (np.ndarray): A :math:`n \times K` matrix with :math:`k^\text{th}` columns corresponding to the qth
                derivative of the :math:`k^\text{th}` basis functions evaluated at locations `x` of length :math:`n`.

        """
        raise NotImplementedError()

    def __call__(self, x, q=0):
        """Evaluate the qth derivative of all basis functions at locations x.

        Args:
            x (np.ndarray): Locations to evaluate basis function at.

            q (int, Optional): The order of the derivative to take of the basis functions. Defaults to 0.

        Returns:
            (np.ndarray): A :math:`n \times K` matrix with :math:`k^\text{th}` columns corresponding to the qth
                derivative of the :math:`k^\text{th}` basis functions evaluated at locations `x` of length :math:`n`.

        Raises:
            ValueError: If not all locations `x` lie in the domain of the basis system.

        """
        if not (np.all(np.less_equal(self.domain[0], x)) and np.all(np.less_equal(x, self.domain[1]))):
            raise ValueError("Arguments must all be within the domain of the basis system")
        basis_mat = self._evaluate_basis(x, q)
        return basis_mat

    @abstractmethod
    def penalty(self, q, k=12):
        """Calculate the qth order penalty matrix for the basis system.

         The form of the penalty matrix is given by:

        .. math::
            P = [p_{kl}]

        .. math::
            p_{kl} = \int B_k^{(q)}(t) B_l^{(q)}(t)dt

        where :math:`B_k` is the :math:`k^\text{th}` basis function.

        Args:
            q (int): The order of the derivative of the penalty matrix.

            k (int, Optional): Number of samples for romberg integration calculated by :math:`2^k + 1`. Defaults to 12.

        Returns:
            (np.ndarray): A :math:`K x K` matrix with elements given by :math:`p_{kl}`.

        """
        raise NotImplementedError()


class Monomial(Basis):
    """Representation of the univariate monomial basis system.

     Basis system is specified as the collection :math:`\{B_k\}_{k=1}^K` where:

    .. math::
        B_k(t) = t^{k-1}

    Attributes:
        domain (tuple):  The domain of the basis system specified by the lower and upper bound of the system.

        K (int): Number of basis functions to use in the basis system.

    """

    def __init__(self, domain, K):
        """Inits the Monomial class to represent the monomial basis system across the domain.
        Args:
            domain (tuple): The domain of the basis system specified by the lower and upper bound of the system.

            K (int): Number of basis functions to use in the basis system.

        """
        super().__init__(domain, K)

    def _evaluate_basis(self, x, q):
        """Evaluate the qth derivative of all basis functions at locations x for the Monomial basis system.

        The qth derivative of basis function :math:`B_k(t)` is given by:
        .. math:
            \frac{d^{(q)}B_k(t)}{dt} = \prod_{i=1}^{q}(k-i) t^{k-q}

        Args:
            x (np.ndarray): Locations to evaluate basis function at.

            q (int): The order of the derivative to take of the basis functions.

        Returns:
            (np.ndarray): A :math:`n \times K` matrix with :math:`k^\text{th}` columns corresponding to the qth
                derivative of the :math:`k^\text{th}` basis functions evaluated at locations `x` of length :math:`n`.

        """
        deg = self.K
        monomial_vecs = np.vander(x, N=deg, increasing=True)
        if q != 0:
            fac = [np.prod(range(f + 1 - q, f + 1)) if f > 0 else 0 for f in range(0, deg)]
            monomial_vecs = fac * ((np.c_[np.ones((len(x), q)), monomial_vecs])[:, :-q])
        return monomial_vecs

    def penalty(self, q, k=12):
        """Calculate the qth order penalty matrix for the basis system.

         The form of the penalty matrix is given by:

        .. math::
            P = [p_{kl}]

        .. math::
            p_{kl} = \int B_k^{(q)}(t) B_l^{(q)}(t)dt

        where :math:`B_k` is the :math:`k^\text{th}` basis function.

        Args:
            q (int): The order of the derivative of the penalty matrix.

            k (int, Optional): Number of samples for romberg integration calculated by :math:`2^k + 1`. Defaults to 12.

        Returns:
            (np.ndarray): A :math:`K x K` matrix with elements given by :math:`p_{kl}`.

        """
        inner_product = np.zeros((self.K, self.K))
        for i in np.arange(q, self.K):
            ifac = 1
            for k in np.arange(1, q + 1):
                ifac *= i - k + 1
            for j in np.arange(i, self.K):
                jfac = 1
                for k in np.arange(1, q + 1):
                    jfac *= j - k + 1
                ipow = i + j - 2 * q + 1
                inner_product[i, j] = (self.domain[1] ** ipow - self.domain[0] ** ipow) * ifac * jfac / ipow
                inner_product[j, i] = inner_product[i, j]
        return inner_product
