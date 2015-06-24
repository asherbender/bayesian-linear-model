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

# Copyright 2015 Asher Bender
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.stats
from scipy.special import gammaln
from numpy.core.umath_tests import inner1d

# --------------------------------------------------------------------------- #
#                              Module Functions
# --------------------------------------------------------------------------- #


def _update(X, y, mu, S, alpha, beta):
    r"""Update sufficient statistics of the Normal-inverse-gamma distribution.

    .. math::

        \mathbf{w}_N &= \mathbf{V_N}\left(\mathbf{V_0}^{-1}\mathbf{w}_0 +
                                          \mathbf{X}^T\mathbf{y}\right)        \\
        \mathbf{V_N} &= \left(\mathbf{V_0}^{-1} + \mathbf{X}^T\mathbf{X}\right)^{-1} \\
        a_N          &= a_0 + \frac{n}{2}                                \\
        b_N          &= b_0 + \frac{k}{2}
                        \left(\mathbf{w}_0^T\mathbf{V}_0^{-1}\mathbf{w}_0 +
                              \mathbf{y}^T\mathbf{y} -
                              \mathbf{w}_N^T\mathbf{V}_N^{-1}\mathbf{w}_N
                        \right)

    Args:
      X (|ndarray|): (N x M) model inputs.
      y (|ndarray|): (N x 1) target outputs.
      mu (|ndarray|): Mean (:math:`\mathbf{w}_0`) of the normal
        distribution.
      S (|ndarray|): Dispersion (:math:`\mathbf{V}_0`) of the normal
        distribution.
      alpha (|float|): Shape parameter (:math:`a_0`) of the inverse Gamma
        distribution.
      beta (|float|): Scale parameter (:math:`b_0`) of the inverse Gamma
        distribution.

    Returns:
      |tuple|: The updated sufficient statistics are return as a tuple (mu, S,
           alpha, beta)

    """

    # Store prior parameters.
    mu_0 = mu
    S_0 = S

    # Update precision (Eq 7.71 ref [1], modified for precision).
    S = S_0 + np.dot(X.T, X)

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
    b = S_0.dot(mu_0) + X.T.dot(y)
    L = np.linalg.cholesky(S)
    mu = np.linalg.solve(L.T, np.linalg.solve(L, b))

    # Update shape parameter (Eq 7.72 ref [1]).
    N = X.shape[0]
    alpha += N / 2.0

    # Update scale parameter (Eq 7.73 ref [1]).
    beta += float(0.5 * (mu_0.T.dot(S_0.dot(mu_0)) +
                         y.T.dot(y) -
                         mu.T.dot(S.dot(mu))))

    return mu, S, alpha, beta


def _uninformative_fit(X, y):
    r"""Initialise sufficient statistics using an uninformative prior.

    .. math::

        \mathbf{w}_N &= \mathbf{V_N}\mathbf{X}^T\mathbf{y}         \\
        \mathbf{V_N} &= \left(\mathbf{X}^T\mathbf{X}\right)^{-1}   \\
        a_N          &= \frac{N - D}{2}                            \\
        b_N          &= \frac{1}{2}
                        \left(\mathbf{y} - \mathbf{X}\mathbf{w}_N\right)^T
                        \left(\mathbf{y} - \mathbf{X}\mathbf{w}_N\right)

    Args:
      X (|ndarray|): (N x M) model inputs.
      y (|ndarray|): (N x 1) target outputs.

    Returns:
      |tuple|: The updated sufficient statistics are return as a tuple (mu, S,
           alpha, beta)

    """

    N, D = X.shape
    XX = np.dot(X.T, X)

    mu = np.linalg.solve(XX, np.dot(X.T, y))
    V = np.linalg.inv(XX)
    alpha = float(N - D) / 2.0
    beta = 0.5 * np.sum((y - np.dot(X, mu))**2)

    return mu, V, alpha, beta


def _predict_mean(X, mu):
    """Calculate posterior predictive mean.

    Args:
      X (|ndarray|): (N x M) input query locations (:math:`\tilde{\mathbf{X}}`)
          to perform prediction.
      mu (|ndarray|): Mean (:math:`\mathbf{w}_0`) of the normal
        distribution.

    Returns:
      |ndarray|: posterior mean

    """

    # Calculate mean.
    #     Eq 7.76 ref [1]
    return np.dot(X, mu)


def _predict_variance(X, S, alpha, beta):
    """Calculate posterior predictive variance.

    Args:
      X (|ndarray|): (N x M) input query locations (:math:`\tilde{\mathbf{X}}`)
          to perform prediction.
      S (|ndarray|): Dispersion (:math:`\mathbf{V}_0`) of the normal
        distribution.
      alpha (|float|): Shape parameter (:math:`a_0`) of the inverse Gamma
        distribution.
      beta (|float|): Scale parameter (:math:`b_0`) of the inverse Gamma
        distribution.

    Returns:
      |ndarray|: posterior variance

    """

    # Note that the scaling parameter is not equal to the variance in
    # the general case. In the limit, as the number of degrees of
    # freedom reaches infinity, the scale parameter becomes equivalent
    # to the variance of a Gaussian.
    uw = np.dot(X, np.linalg.solve(S, X.T))
    S_hat = (beta / alpha) * (np.eye(len(X)) + uw)
    S_hat = np.sqrt(np.diag(S_hat))

    return S_hat


def _posterior_likelihood(y, m_hat, S_hat, alpha):
    """Calculate posterior predictive data likelihood.

    Args:
      y (|ndarray|): (N x 1) output query locations.
      m_hat (|ndarray|): Predicted mean.
      S_hat (|ndarray|): Predicted variance.
      S (|ndarray|): Dispersion (:math:`\mathbf{V}_0`) of the normal
        distribution.
      alpha (|float|): Shape parameter (:math:`a_0`) of the inverse Gamma
        distribution.

    Returns:
      |ndarray|: posterior variance

    """

    q = scipy.stats.t.pdf(y, df=2 * alpha, loc=m_hat, scale=S_hat)

    return q


def _evidence(N, S_N, alpha_N, beta_N, S_0=None, alpha_0=None, beta_0=None):
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

    Args:
      N (|int|): Number of observations.
      S_N (|ndarray|): Dispersion (:math:`\mathbf{V}_0`) of the normal
        distribution.
      alpha_N (|float|): Shape parameter (:math:`a_0`) of the inverse Gamma
        distribution.
      beta_N (|float|): Scale parameter (:math:`b_0`) of the inverse Gamma
        distribution.
      S_0 (|ndarray|, *optional*): Prior dispersion (:math:`\mathbf{V}_0`) of
        the normal distribution. Set to |None| to use uninformative value.
      alpha_0 (|float|, *optional*): Prior shape parameter (:math:`a_0`) of the
        inverse Gamma distribution. Set to |None| to use uninformative value.
      beta_0 (|float|, *optional*): Prior scale parameter (:math:`b_0`) of the
        inverse Gamma distribution. Set to |None| to use uninformative value.

    Returns:
      |float|: The log marginal likelihood is returned.

    """

    # The likelihood can be broken into simpler components:
    # (Eq 3.118 ref [2], Eq 203 ref [3])
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

    A = -0.5 * N * np.log(2 * np.pi)

    # Prior value specified.
    if beta_0 is not None:
        B = alpha_0 * np.log(beta_0) - alpha_N * np.log(beta_N)

    # Approximate uninformative prior.
    else:
        B = -alpha_N * np.log(beta_N)

    # Prior value specified.
    if alpha_0 is not None:
        C = gammaln(alpha_N) - gammaln(alpha_0)

    # Approximate uninformative prior.
    else:
        C = gammaln(alpha_N)

    # Prior value specified.
    S_N = np.linalg.inv(S_N)
    if S_0 is not None:
        S_0 = np.linalg.inv(S_0)
        D = 0.5 * np.log(np.linalg.det(S_N)) - \
            0.5 * np.log(np.linalg.det(S_0))

    # Approximate uninformative prior.
    else:
        D = 0.5 * np.log(np.linalg.det(S_N))

    return A + B + C + D

# --------------------------------------------------------------------------- #
#                               Module Objects
# --------------------------------------------------------------------------- #


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

    The sufficient statistics will be initialised during the first call to
    :py:meth:`.update` where the dimensionality of the problem can be inferred
    from the data.

    If the dimensionality of the problem ``D``, ``location`` or ``dispersion``
    are specified, initialisation will occur immediately. Uninformative values
    will be used for any unspecified parameters. Initialising the sufficient
    statistics immediately has the minor advantage of performing error checking
    before the first call to :py:meth:`.update`.

    Args:
      basis (|callable|): Function for performing basis function expansion on
        the input data.
      D (|int|, *optional*): Dimensionality of problem, after basis function
        expansion. If this value is supplied, it will be used for error
        checking when the sufficient statistics are initialised. If it is not
        supplied, the dimensionality of the problem will be inferred from
        either ``location``, ``dispersion`` or the first call to
        :py:meth:`.update`.
      location (|ndarray|, *optional*): Prior mean (:math:`\mathbf{w}_0`) of
        the normal distribution. Set to |None| to use uninformative value.
      dispersion (|ndarray|, *optional*): Prior dispersion
        (:math:`\mathbf{V}_0`) of the normal distribution. Set to |None| to use
        uninformative value.
      shape (|float|, *optional*): Prior shape parameter (:math:`a_0`) of the
        inverse Gamma distribution. Set to |None| to use uninformative value.
      scale (|float|, *optional*): Prior scale parameter (:math:`b_0`) of the
        inverse Gamma distribution. Set to |None| to use uninformative value.

    Raises:
      ~exceptions.Exception: If any of the input parameters are invalid.

    """

    def __init__(self, basis, D=None, location=None, dispersion=None,
                 shape=None, scale=None):

        # Ensure the basis function expansion is a callable function.
        self.__basis = basis
        if not callable(basis):
            msg = "The input 'basis' must be a callable function."
            raise Exception(msg)

        # Number of observations.
        self.__D = D
        self.__N = 0

        # Store prior.
        self.__mu_0 = location
        self.__S_0 = dispersion
        self.__alpha_0 = shape
        self.__beta_0 = scale

        # Work with precision if variance was provided.
        try:
            self.__S_0 = np.linalg.inv(self.__S_0)
        except:
            pass

        # Reset sufficient statistics.
        self.__mu_N = None
        self.__S_N = None
        self.__alpha_N = None
        self.__beta_N = None

        # Sufficient statistics have not been validated. Flag object as
        # uninitialised.
        self.__initialised = False

        # Attempt to initialise object from user input (either the
        # dimensionality 'D' or the sufficient statistics). If the object can
        # be initialise and there is an error, hault early. If the object
        # cannot be initialised, wait until a call to :py:meth:update (do not
        # throw error).
        try:
            self.__initialise(D=self.__D)
        except:
            raise

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

    def __initialise(self, D=None):
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

        # Infer dimensionality...
        if D is None:
            # From the location parameter.
            if isinstance(self.__mu_0, np.ndarray) and self.__mu_0.ndim == 2:
                self.__D = self.__mu_0.shape[0]

            # From the dispersion parameters.
            elif isinstance(self.__S_0, np.ndarray) and self.__S_0.ndim == 2:
                self.__D = self.__S_0.shape[0]

            # Cannot infer the dimensionality, it has not been specified. Exit
            # initialisation.
            else:
                return None

        # Check the input dimensionality is a positive scalar.
        elif not np.isscalar(D) or D <= 0:
            msg = 'The input dimension must be a positive scalar.'
            raise Exception(msg)

        # Store dimensionality of the data.
        elif self.__D is None:
            self.__D = int(D)

        # If the location parameter has not been set, use an uninformative
        # value (Eq 7.78 ref [1]).
        if self.__mu_0 is None:
            self.__mu_N = np.zeros((self.__D, 1))

        # Check that the location parameter is an array.
        elif (not isinstance(self.__mu_0, np.ndarray) or self.__mu_0.ndim != 2
              or self.__mu_0.shape[1] != 1):
            msg = 'The location parameter must be a (D x 1) numpy array.'
            raise Exception(msg)

        # Check the location parameter has the same dimensional as the input
        # data (after basis function expansion).
        elif self.__mu_0.shape[0] != self.__D:
            msg = 'The location parameter is a ({0[0]} x {0[1]}) matrix. '
            msg += 'The problem is {1}-dimensional. The location parameter '
            msg += 'must be a ({1} x 1) matrix.'
            raise Exception(msg.format(self.__mu_0.shape, self.__D))

        # User location is valid. Set location to specified value.
        else:
            self.__mu_N = self.__mu_0

        # If the dispersion parameter has not been set, use an uninformative
        # value (Eq 7.79 ref [1]).
        if self.__S_0 is None:
            self.__S_N = np.zeros((self.__D, self.__D))

        # Check that the dispersion parameter is an array.
        elif not isinstance(self.__S_0, np.ndarray) or self.__S_0.ndim != 2:
            msg = 'The dispersion parameter must be a (D x D) numpy array.'
            raise Exception(msg)

        # Check the dispersion parameter has the same dimensional as the input
        # data (after basis function expansion).
        elif ((self.__S_0.shape[0] != self.__D) and
              (self.__S_0.shape[1] != self.__D)):
            msg = 'The dispersion parameter is a ({0[0]} x {0[1]}) matrix. '
            msg += 'The design matrix (input data after basis function '
            msg += 'expansion) is {1}-dimensional. The dispersion parameter '
            msg += 'must be a ({1} x {1}) matrix.'
            raise Exception(msg.format(self.__S_0.shape, self.__D))

        # Convert covariance into precision.
        else:
            self.__S_N = self.__S_0

        # Use uninformative shape (Eq 7.80 ref [1]).
        if self.__alpha_0 is None:
            self.__alpha_N = -float(self.__D) / 2.0

        # Check the shape parameter is greater than zero.
        elif not np.isscalar(self.__alpha_0) or self.__alpha_0 < 0:
            msg = 'The shape parameter must be greater than or equal to zero.'
            raise Exception(msg)

        # User shape is valid. Set shape to specified value.
        else:
            self.__alpha_N = self.__alpha_0

        # Use uninformative scale (Eq 7.81 and 7.82 ref [1]).
        if self.__beta_0 is None:
            self.__beta_N = 0

        # Check the scale parameter is greater than zero.
        elif not np.isscalar(self.__beta_0) or self.__beta_0 < 0:
            msg = 'The scale parameter must be greater than or equal to zero.'
            raise Exception(msg)

        # User scale is valid. Set scale to specified value.
        else:
            self.__beta_N = self.__beta_0

        # The sufficient statistics have been validated. Prevent object from
        # checking the sufficient statistics again.
        self.__initialised = True

    def reset(self):
        """Reset sufficient statistics to prior values."""

        # Force the model to reset the sufficient statistics.
        self.__initialised = False

        # Erase current sufficient statistics.
        self.__mu_N = None
        self.__S_N = None
        self.__alpha_N = None
        self.__beta_N = None

        # Attempt to initialise the sufficient statistics from the prior.
        try:
            self.__initialise()
        except:
            raise

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

        # Get size of input data.
        N, D = phi.shape
        self.__N += N

        # Check sufficient statistics are valid (only once).
        if not self.__initialised:
            self.__initialise(D)

            # Ensure distribution is defined (i.e. D < N - 1). See:
            #
            #     Maruyama, Y. and E. George (2008). A g-prior extension
            #     for p > n. Technical report, U. Tokyo.
            #
            if self.__alpha_0 is None and self.__D >= (N - 1):
                msg = 'Update is only defined for D < N - 1. Initialise with '
                msg += 'more than {0} observations.'
                raise Exception(msg.format(self.__D + 1))

        # Check dimensions of input data.
        if self.__D != D:
            msg = 'The input data, after basis function expansion, is '
            msg += '{0}-dimensional. Expected {1}-dimensional data.'
            raise Exception(msg.format(D, self.__D))

        # Update sufficient statistics.
        self.__mu_N, self.__S_N, self.__alpha_N, self.__beta_N = \
            _update(phi, y, self.__mu_N, self.__S_N,
                    self.__alpha_N, self.__beta_N)

    def predict(self, X, y=None, variance=False):
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

        The data likelihood can also be requested by specifying both a set of
        test inputs, :math:`\tilde{\mathbf{X}}` and a set of test output,
        :math:`\tilde{\mathbf{y}}`, locations.

        Args:
          X (|ndarray|): (N x M) input query locations
            (:math:`\tilde{\mathbf{X}}`) to perform prediction.
          y (|ndarray|, *optional*): (K x 1) output query locations
            (:math:`\tilde{\mathbf{y}}`) to request data likelihood.
          variance (|bool|, *optional*): set to |True| to return the 95%
            confidence intervals. Default is set to |False|.

        Returns:
          |ndarray| or |tuple|:
            * |ndarray|: By default only the predicted means are returned as a
              (N x 1) array.
            * (|ndarray|, |ndarray|): If ``variance`` is set to |True| a tuple
              is returned containing both the predicted means (N x 1) and the
              95% confidence intervals (N x 1).
            * (|ndarray|, |ndarray|, |ndarray|): If ``y`` is set, the value of
              ``variance`` is ignored and the predicted means and 95%
              confidence intervals are returned. The final returned value is
              either a (N x 1) array or (K x N) matrix of likelihood values. If
              ``X`` and ``y`` are the same length (N = K), the result is
              returned as an array. If ``X`` and ``y`` are NOT the same length
              (N != K), a matrix is returned where each row represents an
              element in ``y`` and each column represents a row in ``X``.

        Raises:
          ~exceptions.Exception: If the sufficient statistics have not been
            initialised with observed data. Call :py:meth:`.update` first.

        """

        # Ensure the sufficient statistics have been initialised.
        if not self.__initialised:
            msg = 'The sufficient statistics need to be initialised before '
            msg += "calling 'predict()'. Run 'update()' first."
            raise Exception(msg)

        # Perform basis function expansion.
        phi = self.__design_matrix(X)

        # Calculate mean.
        m_hat = _predict_mean(phi, self.__mu_N)

        # Calculate variance.
        if (y is not None) or variance:
            S_hat = _predict_variance(phi,
                                      self.__S_N,
                                      self.__alpha_N,
                                      self.__beta_N)

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

            # Return mean and 95% confidence interval.
            if y is None:
                return (m_hat, ci * S_hat[:, np.newaxis])

            # Return mean, 95% confidence interval and likelihood.
            else:
                N = phi.shape[0]
                K = y.size

                # Return array.
                if N == K:
                    q = _posterior_likelihood(y.squeeze(),
                                              m_hat.squeeze(),
                                              S_hat.squeeze(),
                                              self.__alpha_N)

                # Return matrix result.
                else:
                    q = _posterior_likelihood(y.reshape((K, 1)),
                                              m_hat.reshape((1, N)),
                                              S_hat.reshape((1, N)),
                                              self.__alpha_N)

                return (m_hat, ci * S_hat[:, np.newaxis], q)

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
          |float|: The log marginal likelihood is returned.

        Raises:
          ~exceptions.Exception: If the sufficient statistics have not been
            initialised with observed data. Call :py:meth:`.update` first.

        """

        # Ensure the sufficient statistics have been initialised.
        if not self.__initialised:
            msg = 'The sufficient statistics need to be initialised before '
            msg += "calling 'evidence()'. Run 'update()' first."
            raise Exception(msg)

        return _evidence(self.__N,
                         self.__S_N, self.__alpha_N, self.__beta_N,
                         self.__S_0, self.__alpha_0, self.__beta_0)

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

        Raises:
          ~exceptions.Exception: If the sufficient statistics have not been
            initialised with observed data. Call :py:meth:`.update` first.

        """

        # Ensure the sufficient statistics have been initialised.
        if not self.__initialised:
            msg = 'The sufficient statistics need to be initialised before '
            msg += "calling 'random()'. Run 'update()' first."
            raise Exception(msg)

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
