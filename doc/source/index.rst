Bayesian Linear Model
===================================

Consider a linear model that produces target outputs, :math:`y_i`, by linearly
combining the model parameters, :math:`w_i`, and the input variables,
:math:`x_i`:

.. math::

   y\left(\mathbf{x}, \mathbf{w}\right) = w_0 +
                                          w_1x_1 +
                                          \ldots +
                                          w_Dx_D.


The linear model can be extended to produce outputs that are a linear
combination of fixed nonlinear functions of the input variables. These functions
are known as *basis* functions. The linear model can now be expressed as

.. math::

   y\left(\mathbf{x}, \mathbf{w}\right) = w_0\phi_0\left(\mathbf{x}\right) +
                                          w_1\phi_1\left(\mathbf{x}\right) +
                                          \ldots +
                                          w_D\phi_D\left(\mathbf{x}\right).

For a set of observations perturbed by noise,

.. math::

    \mathbf{y} = \mathbf{\Phi}\mathbf{w} + \mathbf{\epsilon}

where:

* :math:`\mathbf{y} = \{y_1, \ldots, y_N\}^T` are the observed scalar
  outputs.
* :math:`\mathbf{X} = \{\mathbf{x}_1, \ldots, \mathbf{x}_N\}^T` are the
  :math:`M`-dimensional inputs.
* :math:`\mathbf{w} = \{w_1, \ldots, w_N\}^T` is the vector of model
  parameters.
* :math:`\mathbf{\Phi}` is the design matrix where the input data,
  :math:`\mathbf{X}`, has been passed through the vector of nonlinear
  basis functions :math:`\mathbf{\phi} = \{\phi_1, \ldots, \phi_D\}`.
* :math:`\mathbf{\epsilon}` is independent and identically distributed
  (i.i.d.) Gaussian noise with a zero mean and a variance of
  :math:`\sigma^2`.

Under the Bayesian linear model, observations are modelled using the following
likelihood:

.. math::

    p(\mathbf{y} \vert \mathbf{X}, \mathbf{w}, \sigma^2) =
        \mathcal{N}\left(\mathbf{y} \vert
                         \mathbf{\Phi}\mathbf{w},
                         \sigma^2\mathbf{I}_N
                   \right).

Given the Gaussian likelihood function, the natural conjugate prior takes the
form of a Normal-inverse-gamma distribution such that:

.. math::

    p(\mathbf{w}, \sigma^2) = \mathcal{N}\left(\mathbf{w} \vert
                                               \mathbf{w}_0, \mathbf{V_0}
                                         \right)
                              \mathcal{IG}\left(\sigma^2 \vert a_0, b_0 \right)
                              .

This distribution is defined over the unknown model weights, :math:`\mathbf{w}`,
and output noise, :math:`\sigma^2`. Since the Normal-inverse-gamma distribution
is a conjugate prior, for the Gaussian likelihood, the posterior also takes the
form of a Normal-inverse-gamma distribution. Conjugacy allows the distribution
over the model weights and output noise, :math:`\left(\mathbf{w},
\sigma^2\right)`, to be :func:`sequentially updated
<linear_model.BayesianLinearModel.update>` with data using closed form
expressions. Similarly, :func:`inference
<linear_model.BayesianLinearModel.predict>` and :func:`model selection
<linear_model.BayesianLinearModel.evidence>` can be carried out with efficient
closed form expressions.

..
   Note that the entire module could be documented automagically using the
   ':members:' option in 'automodule'. The manual approach adopted here has been
   pursued to force sphinx into inserting horizontal lines in-between the class
   methods. The empty 'automodule' call is used to load the documentation/macros
   specified in the doc-string of the module. The class is documented by
   subsequent calls to 'autoclass' and 'automethod'.

.. currentmodule:: linear_model
.. automodule:: linear_model

.. raw:: html

    <hr>

.. autoclass:: BayesianLinearModel

   .. raw:: html

       <hr>

   .. automethod:: update

   .. raw:: html

       <hr>

   .. automethod:: empirical_bayes

   .. raw:: html

       <hr>

   .. automethod:: predict

   .. raw:: html

       <hr>

   .. automethod:: evidence

   .. raw:: html

       <hr>

   .. automethod:: random

   .. raw:: html

       <hr>

   .. automethod:: reset



.. raw:: html

    <hr>

References
===================================

.. _[1]: http://www.cs.ubc.ca/~murphyk/MLbook/
.. _[2]: http://research.microsoft.com/en-us/um/people/cmbishop/prml/
.. _[3]: http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

`[1]`_ **Murphy, K. P.**, *Machine learning: A probabilistic perspective*,
       The MIT Press, 2012

`[2]`_ **Bishop, C. M**, *Pattern Recognition and Machine Learning (Information Science and Statistics)*,
       Jordan, M.; Kleinberg, J. & Scholkopf, B. (Eds.), Springer, 2006

`[3]`_ **Murphy, K. P.**, *Conjugate Bayesian analysis of the Gaussian distribution*,
       Department of Computer Science, The University of British Columbia, 2007
