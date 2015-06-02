from setuptools import setup

setup(
    name='linear_model',
    version='1.0',
    author='Asher Bender',
    author_email='a.bender.dev@gmail.com',
    description=('Implementation of the Bayesian linear model.'),
    py_modules=['linear_model'],
    install_requires=[
        'numpy',                # Tested on 1.9.2
        'scipy',                # Tested on 0.15.1
        'Sphinx',               # Tested on 1.3.1
    ]
)
