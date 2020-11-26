Basis
=====
Basis is a package for building univariate basis systems in python. The package is heavily based on the basis functions
in the R_ package fda_. As such the package is highly influenced by the monograph Functional Data Analysis, [1].


What does Basis do?
-------------------
The Basis package offers classes to construct and evaluate well known basis systems such as:

* Monomial
* Fourier
* B-spline
* Exponential

The package alo offers the ability to evaluate the integrated squared q\ :sup:`th` derivative of the basis system. This
is often useful in calculating a penalty matrix for use in a smoothing method.


.. _R : https://www.r-project.org/
.. _fda : https://cran.r-project.org/web/packages/fda/
.. [1] J. O. Ramsay and B. W. Silverman, Functional data analysis. New York (N.Y.): Springer Science+Business Media, 2010.


