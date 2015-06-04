Bayesian Linear Model
================================================================================

**Author**: Asher Bender

**Date**: June 2015

**License**: `Apache License Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0>`_

Overview
--------------------------------------------------------------------------------

This code implements the `Bayesian linear model
<http://en.wikipedia.org/wiki/Bayesian_linear_regression>`_ [:ref:`1, 2, 3
<references>`]. Treating linear models under a Bayesian framework allows:

* parameter estimation (learning coefficients of the linear model)
* performing prediction
* model selection

These features are demonstrated in the figure below where the task is to learn a
polynomial approximation of a noisy `sine <http://en.wikipedia.org/wiki/Sine>`_
function:

.. _example-figure:

.. figure:: ./example/example.png
   :scale: 100 %
   :alt: output of example code
   :align: center

The top subplot shows the log marginal likelihood after fitting polynomials of
increasing complexity (degrees) to the data. The model with the highest log
marginal likelihood is marked by the vertical red line. The benefit of Bayesian
model selection, over maximum likelihood methods, is that maximising the log
marginal likelihood (model evidence) tends to avoid over-fitting during model
selection. This is due to a model complexity penalty in the marginal likelihood
equation that preferences simpler models. The optimal model will provide a
balance between data-fit and model complexity, leading to better generalisation.

The bottom subplot shows the noisy sine data (black dots) and predictions from
the model (solid red line), including a 95% confidence bound (dashed red
line). The model used in the bottom plot is the model recommended in the top
plot.

The code used to produce this figure is provided in the :ref:`Example
<example-code>` section.

Dependencies
--------------------------------------------------------------------------------

The following libraries are used in the Bayesian linear model module:

* Python
* Numpy
* Scipy
* Sphinx

The following libraries are used in the example code but are *not* requirements
of the Bayesian linear model module:

* Matplotlib
* Sklearn

Installation
--------------------------------------------------------------------------------

This code supports installation using pip (via `setuptools
<https://pypi.python.org/pypi/setuptools>`_). To install from the git
repository:

.. code-block:: bash

    git clone https://github.com/asherbender/bayesian-linear-model
    cd bayesian-linear-model
    sudo pip install .

To generate the Sphinx documentation:

.. code-block:: bash

    cd doc/
    make html

The entry point of the documentation can then be found at:

.. code-block:: bash

    build/html/index.html

To uninstall the package:

.. code-block:: bash

    pip uninstall linear_model


.. _example-code:

Example
--------------------------------------------------------------------------------

The following code is a short demonstration of how to use the
``BayesianLinearModel()`` class for model selection and inference. This code was
used to produce the :ref:`example figure <example-figure>`:

.. literalinclude:: ./example/example.py
   :language: python
   :name: example_code

.. _references:

References
--------------------------------------------------------------------------------

The Bayesian linear model module was created using the following references:

.. _[1]: http://www.cs.ubc.ca/~murphyk/MLbook/
.. _[2]: http://research.microsoft.com/en-us/um/people/cmbishop/prml/
.. _[3]: http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

`[1]`_ Murphy, K. P., Machine learning: A probabilistic perspective,
       The MIT Press, 2012

`[2]`_ Bishop, C. M, Pattern Recognition and Machine Learning (Information Science and Statistics),
       Jordan, M.; Kleinberg, J. & Scholkopf, B. (Eds.), Springer, 2006

`[3]`_ Murphy, K. P., Conjugate Bayesian analysis of the Gaussian distribution,
       Department of Computer Science, The University of British Columbia, 2007
