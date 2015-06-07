import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from scipy.special import gammaln


def plot(lines, xlabel='X input domain', ylabel='Y output domain'):
    """Plot line information stored as a dictionary.

    Args:
      lines (dict or list): Dictionary or list of dictionaries containing plot
          information. The dictionaries must contain the 'x' key. Y-data is
          stored using the 'y' key or the 'model' key. 'model' is a callable
          function accepting only one argument: the x-input domain. All
          remaining key-value pairs are passed on to the plot function.
      xlabel (str, optional): Label for x-axis.
      ylabel (str, optional): Label for y-axis.

    """

    # Convert input into a list of dictionaries if only one dictionary is
    # supplied.
    if isinstance(lines, dict):
        lines = (lines, )

    # Iterate through lines in the supplied list.
    for data in lines:
        x = data['x']
        del(data['x'])

        # Create/get y values.
        if 'model' in data:
            y = data['model'](x)
            del(data['model'])
        elif 'y' in data:
            y = data['y']
            del(data['y'])

        # Plot data.
        plt.plot(x, y, **data)

    # Label axes.
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.grid()


def mvst_pdf(x, mu, scale, dof):
    """PDF of multivariate student-T distribution.

    Args:
      x (ndarray): (nxd) input locations.
      mu (ndarray): (dx1) mean/location of the distribution.
      scale (ndarray): (dxd) scale/dispersion of the distribution.
      dof (float): degrees of freedom in the multivariate student-T.

    Returns:
      ndarray: multivariate student-T likelihood of inputs

    """

    # The distribution can be broken into simpler components:
    #
    #     pdf = A * B * C
    #
    # where:
    #
    #     A = gamma((v + D)/2) / [ gamma(v/2) * (v*pi)^(D/2) ]
    #     B = det(S)^(1/2)
    #     C = (1 + m / v)^-((v + d)/2)
    #     m = (x-u)'S(x-u)
    #
    # Using log probabilities:
    #
    #     pdf = A + B + C
    #
    # where:
    #
    #     log(A) = gammaln((v + D)/2) - gammaln(v / 2) - (D / 2) * log(v * pi)
    #     lob(B) = - (1/2) * log(det(S))
    #     log(C) = - (v + d)/2) * log(1 + m / v)
    #

    # Calculate innovation (centre data around the mean).
    D = x.shape[1]
    v = x - mu

    A = gammaln((dof + D)/2.) - gammaln(dof/2.) - (D/2.) * np.log(dof * np.pi)

    # Use Cholesky factorization to calculate the determinant and the inner
    # products (avoid directly inverting the scale matrix).
    #
    #     L = chol(S)
    #     log(det(S)) = 2*sum(log(diag(S)))
    #
    L = np.linalg.cholesky(scale)
    B = -np.sum(np.log(np.diag(L)))

    x = np.linalg.solve(L, v.T).T
    m = np.sum(x * x, axis=1)
    C = -((dof + D)/2.) * np.log(1 + m / dof)

    return np.exp(A + B + C)


def plot_igamma_pdf(blm, min_gamma=1e-6, max_gamma=1.0, samples=1000):
    """Plot inverse gamma distribution from Bayesian linear model.

    Args:
      blm (BayesianLinearModel): Bayesian linear model object.
      min_gamma (float, optional): Minimum value to use in the input domain.
      max_gamma (float, optional): Maximum value to use in the input domain.
      samples (int, optional): Number of samples to use in the plot.

    """

    # Create domain for plotting inverse gamma PDF.
    gamma_domain = np.linspace(min_gamma, max_gamma, samples)

    # Calculate PDF of inverse gamma (Eq 7.74).
    #
    # Note: The prior has been defined over the variance. Take the square
    #       root of the variance to display standard deviations.
    pdf = scipy.stats.invgamma.pdf(gamma_domain, blm.shape, scale=blm.scale)
    plt.plot(np.sqrt(gamma_domain), pdf)
    plt.xlabel('noise (std)')
    plt.ylabel('likelihood')
    plt.xlim([min_gamma, max_gamma])


def plot_mvst_pdf(blm, min_d=-1.0, max_d=1.0, samples=1000):
    """Plot multivariate student-T distribution from Bayesian linear model.

    Args:
      blm (BayesianLinearModel): Bayesian linear model object.
      min_d (float, optional): Minimum value to use in the X/Y domain.
      max_d (float, optional): Maximum value to use in the X/Y domain.
      samples (int, optional): Number of samples to use on each axes of the
          plot.

    """

    # Create local copy of sufficient statistics for convenience.
    mu, V, alpha, beta = blm.location, blm.dispersion, blm.shape, blm.scale

    # Create domain for plotting multivariate student-T PDF.
    student_domain = np.linspace(min_d, max_d, samples)
    x_grid, y_grid = np.meshgrid(student_domain, student_domain)
    student_grid = np.hstack((x_grid.reshape(x_grid.size, 1),
                              y_grid.reshape(y_grid.size, 1)))

    # Calculate PDF of multivariate student-T (Eq 7.75).
    pdf = mvst_pdf(student_grid,
                   mu.T,
                   (beta/alpha) * V,
                   2 * alpha)

    # Plot prior/posterior.
    pdf = pdf.reshape(samples, samples)
    plt.imshow(pdf, origin='lower', extent=[min_d, max_d, min_d, max_d])
    plt.xlabel('Intercept')
    plt.ylabel('Slope')


def plot_random_models(blm, model, lines=50, min_x=-2.0, max_x=2.0,
                       samples=1000):
    """Plot random models drawn from the Bayesian linear model.

    Draw random values from the distribution over the model parameters
    (multivariate-normal, inverse Gamma distribution).

    Args:
      blm (BayesianLinearModel): Bayesian linear model object.
      model (callable): Callable function accepting two arguments: the x-input
          domain and the model parameters.
      lines (int, optional): Number of random models to plot.
      min_x (float, optional): Minimum value to use in the input domain.
      max_x (float, optional): Maximum value to use in the input domain.
      samples (int, optional): Number of samples to in the input domain.

    """

    # Create domain for plotting sampled models.
    x_domain = np.linspace(min_x, max_x, samples)[:, np.newaxis]

    # Plot random lines.
    params = blm.random(samples=lines)
    for i in range(lines):
        y_sampled = model(x_domain, params[i, :])
        plt.plot(x_domain, y_sampled, color=[0.6, 0.6, 0.6])

    mu, S2 = blm.predict(x_domain, variance=True)
    plt.plot(x_domain, mu + S2, 'r--', linewidth=2)
    plt.plot(x_domain, mu, 'r', linewidth=3)
    plt.plot(x_domain, mu - S2, 'r--', linewidth=2)

    plt.xlim([min_x, max_x])
    plt.xlabel('x')
    plt.ylabel('y')


def plot_update(x, y, model, blm):
    """Plot Bayesian linear model update.

    Plot an 'update' to the Bayesian linear model by displaying the
    distribution over model parameters as a subplot containing three plots:

        1) the posterior distribution over the model noise (inverse-Gamma)
        2) the posterior distribution over model parameters (multivariate
           student-T)
        3) the observed data including random models drawn from the posterior

    Args:
      x (ndarray): observed x data.
      y (ndarray): observed y data.
      model (callable): Callable function accepting two arguments: the x-input
          domain and the model parameters.
      blm (BayesianLinearModel): Bayesian linear model object.

    """

    plt.figure(figsize=(18, 6))
    ax = plt.subplot(1, 3, 1)
    plot_igamma_pdf(blm)
    ax.axis('auto')

    ax = plt.subplot(1, 3, 2)
    plot_mvst_pdf(blm)
    ax.axis('auto')

    ax = plt.subplot(1, 3, 3)
    plot_random_models(blm, model)
    plt.plot(x, y, 'k.', markersize=8)
    ax.axis('auto')
    plt.ylim([-2, 2])


def plot_model_fit(MSE, ML):
    """Plot model selection using maximum likelihood and Bayesian methods.

    Args:
      MSE (array): Mean squared error values from maximum likelihood estimation.
      ML (array): Log marginal likelihood values from Bayesian model selection.

    """

    # Prepare subplots.
    num_plots = len(MSE)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    f.subplots_adjust(hspace=5)

    # Note: in MSE plots, the model with the highest complexity will win,
    # subtract a small amount so the vertical line is distinguishable from the
    # plot boarder.
    ax1.plot(np.arange(num_plots), MSE)
    ax1.set_title('Maximum likelihood')
    ax1.set_xlabel('number of degrees')
    ax1.set_ylabel('Mean squared error')
    ax1.axvline(np.argmin(MSE) - 0.05, color='r', linewidth='3')
    ax1.grid('on')

    ax2.plot(np.arange(num_plots), ML)
    ax2.set_title('Bayesian linear model')
    ax2.set_xlabel('number of degrees')
    ax2.set_ylabel('Log marginal likelihood')
    ax2.axvline(np.argmax(ML), color='r', linewidth='3')
    ax2.grid('on')
