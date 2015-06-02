"""
.. sectionauthor:: Asher Bender <a.bender.dev@gmail.com>
.. codeauthor:: Asher Bender <a.bender.dev@gmail.com>

.. |bool| replace:: :class:`.bool`
.. |callable| replace:: :func:`.callable`
.. |False| replace:: :data:`.False`
.. |float| replace:: :class:`.float`
.. |int| replace:: :class:`.int`
.. |ndarray| replace:: :class:`~numpy.ndarray`
.. |None| replace:: :data:`.None`
.. |True| replace:: :data:`.True`
.. |tuple| replace:: :func:`.tuple`

"""
import numpy as np
import scipy.stats
from scipy.special import gammaln
from numpy.core.umath_tests import inner1d


class BayesianLinearModel(object):
    r"""Bayesian linear model.

    Instantiate a Bayesian linear model. If no sufficient statistics are
    supplied at initialisation, the following uninformative semi-conjugate
    prior will be used:

    .. math::

          \mathbf{w}_0 &= \mathbf{0}        \\
          \mathbf{V_0} &= \infty\mathbf{I}  \\
          a_0          &= \frac{-D}{2}      \\
          b_0          &= 0                 \\

    Args:
      basis (|callable|): Function for performing basis function expansion on
        the input data.
      location (|ndarray|, *optional*): Prior mean (:math:`\mathbf{w}_0`) of
        the normal distribution.
      dispersion (|ndarray|, *optional*): Prior dispersion
        (:math:`\mathbf{V}_0`) of the normal distribution.
      shape (|float|, *optional*): Prior shape parameter (:math:`a_0`) of the
        inverse Gamma distribution.
      rate (|float|, *optional*): Prior rate parameter (:math:`b_0`) of the
        inverse Gamma distribution.

    Raises:
      ~exceptions.Exception: If any of the input parameters are invalid.

    """

    def __init__(self, basis, location=None, dispersion=None, shape=None,
                 scale=None):

        # Ensure the basis function expansion is a callable function.
        self.__basis = basis
        if not callable(basis):
            msg = "The input 'basis' must be a callable function."
            raise Exception(msg)

        # Number of observations.
        self.__N = 0

        # Check that the location parameter is an array.
        self.__mu_N = location
        if location is not None:
            if not isinstance(location, np.ndarray) and location.ndim != 2:
                msg = 'The location parameter must be a 2D-numpy array.'
                raise Exception(msg)

        # Check that the dispersion parameter is an array.
        self.__S_0 = dispersion
        self.__S_N = dispersion
        if location is not None:
            if not isinstance(dispersion, np.ndarray) and dispersion.ndim != 2:
                msg = 'The dispersion parameter must be a 2D-numpy array.'
                raise Exception(msg)

        # Check that the shape parameter is a positive scalar.
        self.__alpha_0 = shape
        self.__alpha_N = shape
        if shape is not None:
            if not np.isscalar(shape) or (shape < 0):
                msg = 'The shape parameter must be a positive scalar.'
                raise Exception(msg)

        # Check that the scale parameter is a positive scalar.
        self.__beta_0 = scale
        self.__beta_N = scale
        if scale is not None:
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
        return np.linalg.inv(self.__S_N)

    @property
    def shape(self):
        return self.__alpha_N

    @property
    def scale(self):
        return self.__beta_N

    def __initialise(self, D, N):
        """Initialise sufficient statistics of the distribution.

        This method initialises the sufficient statistics of the multivariate
        normal distribution if they have not been specified. If values have
        been specified, they are checked to ensure the dimensionality has been
        specified correctly.

        If values have not been specified, uninformative values are used
        (Section 7.6.3.2 of [1]):]

            m = zero(D, 1)
            V = inf * eye(D)
            alpha = -D/2
            beta = 0

        the update equations 7.78 - 7.82 can be achieved by setting the prior
        values in equations 7.70 - 7.73 to:

            m_0 = zero(D, 1)
            V_0 = zero(D, D)
            alpha_0 = -D/2
            beta_0 = 0

        """

        # Store dimensionality of the data.
        self.__D = float(D)

        # If the location parameter has not been set, use an uninformative
        # value (Eq 7.78 ref [1]).
        if self.__mu_N is None:
            self.__mu_N = np.zeros((self.__D, 1))

        # Check the location parameter has the same dimensional as the input
        # data (after basis function expansion).
        elif self.__mu_N.shape[0] != self.__D:
            msg = 'The location parameter is a ({0[0]} x {0[1]}) matrix. The '
            msg += 'design matrix (input data after basis function expansion) '
            msg += 'is {1}-dimensional. The location parameter must be a '
            msg += '({1} x 1) matrix.'
            raise Exception(msg.format(self.__mu_N.shape, self.__D))

        # If the dispersion parameter has not been set, use an uninformative
        # value (Eq 7.79 ref [1]).
        if self.__S_N is None:
            self.__S_N = np.zeros((self.__D, self.__D))

        # Check the dispersion parameter has the same dimensional as the input
        # data (after basis function expansion).
        elif ((self.__S_N.shape[0] != self.__D) and
              (self.__S_N.shape[0] != self.__D)):
            msg = 'The dispersion parameter is a ({0[0]} x {0[0]}) matrix. '
            msg += 'The design matrix (input data after basis function '
            msg += 'expansion) is {1}-dimensional. The dispersion parameter '
            msg += 'must be a ({1} x {1}) matrix.'
            raise Exception(msg.format(self.__mu_N.shape, self.__D))

        # Convert covariance into precision.
        else:
            self.__S_0 = np.linalg.inv(self.__S_0)
            self.__S_N = np.linalg.inv(self.__S_N)

        # Use uninformative shape (Eq 7.80 ref [1]).
        if self.__alpha_N is None:
            self.__alpha_N = -D / 2.0

        # Check the shape parameter is greater than zero.
        elif self.__alpha_N < 0:
            msg = 'The shape parameter must be greater than or equal to zero.'
            raise Exception(msg)

        # Use uninformative scale (Eq 7.81 and 7.82 ref [1]).
        if self.__beta_N is None:
            self.__beta_N = 0

        # Check the rate parameter is greater than zero.
        elif self.__beta_N < 0:
            msg = 'The rate parameter must be greater than or equal to zero.'
            raise Exception(msg)

        # Ensure distribution is defined (i.e. D < N - 1). See:
        #
        #     Maruyama, Y. and E. George (2008). A g-prior extension
        #     for p > n. Technical report, U. Tokyo.
        #
        if D >= (N - 1):
            msg = 'Update is only defined for D < N - 1. Initialise with more '
            msg += 'than {0:d} observations.'
            raise Exception(msg.format(D + 1))

        # The sufficient statistics have been validated. Prevent object from
        # checking the sufficient statistics again.
        self.__initialised = True

    def __design_matrix(self, X):
        """Perform basis function expansion to create design matrix."""

        # Perform basis function expansion.
        try:
            return self.__basis(X)
        except Exception as e:
            msg = 'Could not perform basis function expansion with the '
            msg += 'function %s\n\n' % str(self.__basis)
            msg += 'Error thrown:\n %s' % str(e)
            raise Exception(msg)

    def update(self, X, y):
        r"""Update sufficient statistics of the Normal-inverse-gamma distribution.

        .. math::

            \mathbf{w}_N &= \mathbf{V_N}\left(\mathbf{V_0}^{-1}\mathbf{w}_0 +
                                              \Phi^T\mathbf{y}\right)        \\
            \mathbf{V_N} &= \left(\mathbf{V_0}^{-1} + \Phi^T\Phi\right)^{-1} \\
            a_N          &= a_0 + \frac{n}{2}                                \\
            b_N          &= b_0 + \frac{k}{2}
                            \left(\mathbf{w}_0^T\mathbf{V}_0^{-1}\mathbf{w}_0 +
                                  \mathbf{y}^T\mathbf{y} -
                                  \mathbf{w}_N^T\mathbf{V}_N^{-1}\mathbf{w}_N
                            \right)

        Args:
          X (|ndarray|): (N x M) model inputs.
          y (|ndarray|): (N x 1) target outputs.

        Raises:
          ~exceptions.Exception: If there are not enough inputs or the
            dimensionality of the data is wrong.

        """

        # Ensure inputs are valid objects and the same length.
        if (not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray) or
            (X.ndim != 2) or (y.ndim != 2) or (len(X) != len(y))):
            msg = 'X must be a (N x M) matrix and y must be a (N x 1) vector.'
            raise Exception(msg)

        # Perform basis function expansion.
        phi = self.__design_matrix(X)
        N, D = phi.shape
        self.__N += N

        # Check sufficient statistics are valid (only once).
        if not self.__initialised:
            self.__initialise(D, N)

        # Check dimensions of input data.
        if self.__D != D:
            msg = 'The input data, after basis function expansion, is '
            msg += '{0}-dimensional. Expected {1}-dimensional data.'
            Exception(msg.format(D, self.__D))

        # Store prior parameters.
        mu_0 = self.__mu_N
        S_0 = self.__S_N

        # Update precision (Eq 7.71 ref [1], modified for precision).
        self.__S_N = S_N = S_0 + np.dot(phi.T, phi)

        # Update mean using cholesky decomposition.
        #
        # For:
        #     Ax = b
        #     L = chol(A)
        #
        # The solution can be given by:
        #     x = L.T \ (L \ b)
        #
        # Update mean (Eq 7.70 ref[1], modified for precision).
        b = S_0.dot(mu_0) + phi.T.dot(y)
        L = np.linalg.cholesky(self.__S_N)
        self.__mu_N = mu_N = np.linalg.solve(L.T, np.linalg.solve(L, b))

        # Update shape parameter (Eq 7.72 ref [1]).
        self.__alpha_N += N / 2.0

        # Update scale parameter (Eq 7.73 ref [1]).
        self.__beta_N += float(0.5 * (mu_0.T.dot(S_0.dot(mu_0)) +
                                      y.T.dot(y) -
                                      mu_N.T.dot(S_N.dot(mu_N))))

    def predict(self, X, variance=False):
        r"""Calculate posterior predictive values.

        Given a new set of test inputs, :math:`\tilde{\mathbf{X}}`, predict the
        output value. The predictions are T-distributed according to:

        .. math::

            p\left(\tilde{\mathbf{y}} \vert
                          \tilde{\mathbf{X}}, \mathcal{D} \right) =
            \mathcal{T}\left(\tilde{\mathbf{y}} \; \big\vert \;
                             \tilde{\mathbf{\Phi}}\mathbf{w}_N,
                             \frac{b_N}{a_N}
                             \left(\mathbf{I} +
                                   \tilde{\mathbf{\Phi}}\mathbf{V}_N\tilde{\mathbf{\Phi}}^T
                             \right),
                             2a_N
                       \right)

        Args:
          X (|ndarray|): (N x M) input query locations
            (:math:`\tilde{\mathbf{X}}`) to perform prediction.
          variance (|bool|, *optional*): set to |True| to return the 95%
            confidence intervals. Default is set to |False|.

        Returns:
          |ndarray| or |tuple|: If ``variance`` is set to |False| only the
            predicted values are returned as a (N x 1) array. If ``variance``
            is set to |True| a tuple is returned containing both the predicted
            values (N x 1) and the 95% confidence intervals (N x 1).

        """

        # Perform basis function expansion.
        phi = self.__design_matrix(X)

        # Calculate mean.
        #     Eq 7.76 ref [1]
        m_hat = np.dot(phi, self.__mu_N)

        # Calculate mean and variance.
        #     Eq 7.76 ref [1]
        if variance:
            # Note that the scaling parameter is not equal to the variance in
            # the general case. In the limit, as the number of degrees of
            # freedom reaches infinity, the scale parameter becomes equivalent
            # to the variance of a Gaussian.
            uw = np.dot(phi, np.linalg.solve(self.__S_N, phi.T))
            S_hat = (self.__beta_N/self.__alpha_N) * (np.eye(len(phi)) + uw)
            S_hat = np.sqrt(np.diag(S_hat))

            # Calculate a one sided 97.5%, t-distribution, confidence
            # interval. This corresponds to a 95% two-sided confidence
            # interval.
            #
            # For a tabulation of values see:
            #     http://en.wikipedia.org/wiki/Student%27s_t-distribution#Confidence_intervals
            #
            # Note: If the number of degrees of freedom is equal to one, the
            #       distribution is equivalent to the Cauchy distribution. As
            #       the number of degrees of freedom approaches infinite, the
            #       distribution approaches a Gaussian distribution.
            #
            ci = scipy.stats.t.ppf(0.975, 2 * self.__alpha_N)

            return (m_hat, ci * S_hat[:, np.newaxis])

        else:
            return m_hat

    def evidence(self):
        r"""Return log marginal likelihood of the data (model evidence).

        The log marginal likelihood is calculated by taking the log of the
        following equation:

        .. math::

            \renewcommand{\det} [1]{{\begin{vmatrix}#1\end{vmatrix}}}

            p\left(\mathcal{D} \right) = \frac{1}{2\pi^\frac{N}{2}}
                                         \frac{\det{V_N}^\frac{1}{2}}
                                              {\det{V_0}^\frac{1}{2}}
                                         \frac{b_0^{a_0}}
                                              {b_N^{a_N}}
                                         \frac{\Gamma\left(a_N\right)}
                                              {\Gamma\left(a_0\right)}

        Note that the default prior is an improper, uninformative prior. The
        marginal likelihood equation, specified above, equation is undefined
        for improper priors. To approximate the marginal likelihood in this
        situation, the prior sufficient statistics :math:`\left(V_0, a_0,
        b_0\right)` are selectively ignored if they are unset. If all prior
        sufficient statistics are unset (default) the marginal likelihood
        equation is approximated as:

        .. math::

            \renewcommand{\det} [1]{{\begin{vmatrix}#1\end{vmatrix}}}

            p\left(\mathcal{D} \right) = \frac{1}{2\pi^\frac{N}{2}}
                                         \det{V_N}^\frac{1}{2}
                                         \frac{1}
                                              {b_N^{a_N}}
                                         \Gamma\left(a_N\right)

        Although this equation returns an approximate marginal likelihood, it
        can still be used for model selection. The omitted terms, which cannot
        be evaluated, create a constant which scales the final result. During
        model selection this constant will be identical across all models and
        can safely be ignored.

        Returns:
          |float|: The log marginal likelihood is returned. If the object has
            not been initialised with data, |None| is returned.

        """

        #     Eq 3.118 ref [2]
        #     Eq  203  ref [3]

        # The likelihood can be broken into simpler components:
        #
        #     pdf = A * B * C * D
        #
        # where:
        #
        #     A = 1 / (2 * pi)^(N/2)
        #     B = (b_0 ^ a_0) / (b_N ^ a_N)
        #     C = gamma(a_N) / gamma(a_0)
        #     D = det(S_N)^(1/2) / det(S_0)^(1/2)
        #
        # Using log probabilities:
        #
        #     pdf = A + B + C + D
        #
        # where:
        #
        #     log(A) = -0.5 * N * ln(2 * pi)
        #     lob(B) = a_0 * ln(b_0) - a_N * ln(b_N)
        #     log(C) = gammaln(a_N) - gammaln(a_0)
        #     log(D) = ln(det(S_N)^0.5) - ln(det(S_0)^0.5)
        #

        # Ensure the model has been updated with data.
        if not self.__initialised:
            return None

        # Create local copy of sufficient statistics for legibility.
        N = self.__N
        S_0 = self.__S_0
        a_0 = self.__alpha_0
        b_0 = self.__beta_0
        S_N = np.linalg.inv(self.__S_N)
        a_N = self.__alpha_N
        b_N = self.__beta_N

        A = -0.5 * N * np.log(2 * np.pi)

        # Prior value specified.
        if b_0 is not None:
            B = a_0 * np.log(b_0) - a_N * np.log(b_N)

        # Approximate uninformative prior.
        else:
            B = -a_N * np.log(b_N)

        # Prior value specified.
        if a_0 is not None:
            C = gammaln(a_N) - gammaln(a_0)

        # Approximate uninformative prior.
        else:
            C = gammaln(a_N)

        # Prior value specified.
        if S_0 is not None:
            S_0 = np.linalg.inv(S_0)
            D = 0.5 * np.log(np.linalg.det(S_N)) - \
                0.5 * np.log(np.linalg.det(S_0))

        # Approximate uninformative prior.
        else:
            D = 0.5 * np.log(np.linalg.det(S_N))

        return A + B + C + D

    def random(self, samples=1):
        r"""Draw a random model from the posterior distribution.

        The model parameters are T-distributed according to the following
        posterior marginal:

        .. math::

            p\left(\mathbf{w} \vert \mathcal{D} \right) =
            \mathcal{T}\left(
                           \mathbf{w}_N, \frac{b_N}{a_N}\mathbf{V}_N, 2a_N
                       \right)

        Args:
          samples (|int|, *optional*): number of random samples to return.

        Returns:
          |ndarray|: Return (NxD) random samples from the model weights
            posterior. Each row is a D-dimensional vector of random model
            weights.


        """

        # The posterior over the model weights is a Student-T distribution. To
        # generate random models, sample from the posterior marginals.
        #
        #     Eq 7.75 ref [1]

        # Note: Currently the 'scipy.stats.multivariate_normal' object does not
        #       permit (MxMxN) covariance matrices. As a result multiple
        #       observations can only be drawn from ONE multivariate
        #       normal. The code in this method uses broadcasting to vectorise
        #       sampling from multiple, multivariate normals.

        # Draw random samples from the inverse gamma distribution (1x1xN).
        r = scipy.stats.invgamma.rvs(self.__alpha_N,
                                     scale=self.__beta_N,
                                     size=samples).reshape((1, 1, samples))

        # Create multiple multivariate scale matrices from random gamma samples
        # (DxDxN).
        sigma = np.linalg.inv(self.__S_N)
        sigma = r * np.repeat(sigma[:, :, np.newaxis], samples, axis=2)

        # Draw random samples from the standard univariate normal distribution
        # (1xDxN).
        rn = np.random.normal(size=(1, self.__D, samples))

        # Create N random samples (1xD) drawn from multiple, random and unique
        # multivariate normal distributions.
        L = np.rollaxis(np.linalg.cholesky(sigma.T).T, 0, 2)
        sigma = inner1d(np.rollaxis(rn, 0, 2).T, L.T)

        # Return (NxD) samples drawn from multivariate-normal, inverse-gamma
        # distribution.
        return self.__mu_N.T + sigma
