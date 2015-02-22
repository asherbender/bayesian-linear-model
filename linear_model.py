"""Bayesian Linear Model.

References:

   [1] Murphy, K. P., Machine learning: A probabilistic perspective,
       The MIT Press, 2012



.. sectionauthor:: Asher Bender <a.bender.dev@gmail.com>
.. codeauthor:: Asher Bender <a.bender.dev@gmail.com>

"""
import numpy as np


class BayesianLinearRegression(object):

    def __init__(self, basis=None,
                 location=None, dispersion=None,
                 shape=0.0, scale=0.0):

        # Ensure the basis function expansion is a callable function.
        self.__basis = basis
        if not callable(basis):
            msg = "The input 'basis' must be a callable function."
            raise Exception(msg)

        # Check that the location parameter is an array.
        self.__mu_N = location
        if location:
            if not isinstance(location, np.ndarray) and location.ndim != 2:
                msg = 'The location parameter must be a 2D-numpy array.'
                raise Exception(msg)

        # Check that the dispersion parameter is an array.
        self.__S_N = dispersion
        if dispersion:
            if not isinstance(dispersion, np.ndarray) and dispersion.ndim != 2:
                msg = 'The dispersion parameter must be a 2D-numpy array.'
                raise Exception(msg)

        # Check that the shape parameter is a positive scalar.
        self.__alpha_N = shape
        if not np.isscalar(shape) or (shape < 0):
            msg = 'The shape parameter must be a positive scalar.'
            raise Exception(msg)

        # Check that the scale parameter is a positive scalar.
        self.__beta_N = scale
        if not np.isscalar(scale) or (scale < 0):
            msg = 'The scale parameter must be a positive scalar.'
            raise Exception(msg)

        # Force object to validate the sufficient statistics after first call
        # to the update() method.
        self.__initialised = False

    @property
    def location(self):
        return self.__mu_N

    @property
    def dispersion(self):
        return self.__S_N

    @property
    def shape(self):
        return self.__alpha_N

    @property
    def scale(self):
        return self.__beta_N

    def __initialise(self, D):
        """Initialise sufficient statistics of the Gaussian.

        This method initialises the sufficient statistics of the multivariate
        normal distribution if they have not been specified. Uninformative
        values are used. If values have been specified, they are checked to
        ensure the dimensionality has been specified correctly.

        """

        # Store dimensionality of the data.
        self.__D = D

        # If the location parameter has not been set, use an uninformative
        # value.
        if self.__mu_N is None:
            self.__mu_N = np.zeros((self.__D, 1))

        # Check the location parameter has the same dimensional as the input
        # data (after basis function expansion).
        elif self.__mu_N.shape[1] != self.__D:
            msg = 'The location parameter is a ({0[0]} x {0[0]}) matrix. The '
            msg += 'design matrix (input data after basis function expansion) '
            msg += 'is {1}-dimensional. The location parameter must be a '
            msg += '({1} x 1) matrix.'
            raise Exception(msg % (self.__mu_N.shape, self.__D))

        # If the dispersion parameter has not been set, use an uninformative
        # value.
        if self.__S_N is None:
            self.__S_N = np.eye(self.__D)

        # Check the dispersion parameter has the same dimensional as the input
        # data (after basis function expansion).
        elif ((self.__S_N.shape[0] != self.__D) and
              (self.__S_N.shape[0] != self.__D)):
            msg = 'The dispersion parameter is a ({0[0]} x {0[0]}) matrix. '
            msg += 'The design matrix (input data after basis function '
            msg += 'expansion) is {1}-dimensional. The dispersion parameter '
            msg += 'must be a ({1} x {1}) matrix.'
            raise Exception(msg.format(self.__mu_N.shape, self.__D))

        # The sufficient statistics have been validated. Prevent object from
        # checking the sufficient statistics again.
        self.__initialised = True

    def update(self, X, Y):
        """Update sufficient statistics of the model distribution.

        """

        # Ensure inputs are valid objects and the same length.
        if (not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray) or
            (X.ndim != 2) or (Y.ndim != 2) or (len(X) != len(Y))):
            msg = 'X must be a (N x M) matrix and Y must be a (N x 1) vector.'
            raise Exception(msg)

        # Perform basis function expansion.
        try:
            phi = self.__basis(X)
        except Exception as e:
            msg = 'Could not perform basis function expansion with the '
            msg += 'function %s\n\n' % str(self.__basis)
            msg += 'Error thrown:\n %s' % str(e)
            raise Exception(msg)

        # Check sufficient statistics are valid (only once).
        if not self.__initialised:
            self.__initialise(phi.shape[1])

        # Check dimensions of input data.
        if phi.shape[1] != self.__D:
            msg = 'The input data, after basis function expansion, is '
            msg += '{0}-dimensional. Expected {1}-dimensional data.'
            Exception(msg.format(phi.shape[1], self.__D))

        # Cache sufficient statistics before update.
        mu_0 = self.__mu_N
        S_0 = self.__S_N

        # Update Precision (Eq 7.71).
        self.__S_N = S_N = S_0 + np.dot(phi.T, phi)

        # Update mean using cholesky decomposition (Eq 7.70).
        #
        # For:
        #     Ax = b
        #     L = chol(A)
        #
        # The solution can be given by:
        #     x = L.T \ (L \ b)
        #
        b = S_0.dot(mu_0) + phi.T.dot(Y)
        L = np.linalg.cholesky(S_N)
        self.__mu_N = mu_N = np.linalg.solve(L.T, np.linalg.solve(L, b))

        # Update shape parameter (Eq 7.72).
        self.__alpha_N += len(X)/2.0

        # Update rate parameter (Eq 7.73).
        self.__beta_N += 0.5 * (reduce(np.dot, [mu_0.T, S_0, mu_0]) +
                                Y.T.dot(Y) -
                                reduce(np.dot, [mu_N.T, S_N, mu_N]))

    def predict(self, X):
        pass

    def evidence(self):
        pass
