# rust-mlearn
Machine Learning Tools in Rust

## Introduction
This package is meant to provide some machine learning models and tools for the Rust language,
currently using the ndarray package as a matrix and linear algebra backend.

Currently most of these tasks are done using Python with C, C++, and/or Cython packages for the computationally-intensive parts.
There is limited support for machine learning in Rust, so this package will include various basic models. It is not intended to 
be as flexible as something like tensorflow but instead I hope to define better APIs and some native implementations of some of
the more common models.

This package is also intentionally not using a scikit-learn derived interface for models. Instead, the preferred treatment is to 
build a model from a factory class or function. Models can do transformations and predictions but training and initialization
code should be separated from the models to avoid some of the problems seen in sklearn (such as there being a number of different
linear regression models with different training algorithms).

I am also just starting to learn Rust, so it's likely that some design choices can be improved to match the correct style for the
language. Feel free to comment.

## Current Features

* Linear regression with a matrix inversion (Newton's method) solver and optional L2 regularization
* Logistic regression with a naive Newton's method solver and optional L2 regularization
* A couple basic evaluation metrics


## TODOs

There are lots of things to do before releasing this, including
* Settle on a good high level API
* Add in more optimization methods - gradient descent, SGD, different Newton's method variants, simulated annealing
  * Can we make use of 3rd party C/Fortran optimization packages to not have to code everything from scratch?
* Figure out better initialization schemes - randomized methods, maybe some MCMC to walk through parameter space 
* Add in more models - some tractable ones could be some basic Naive Bayes models, linear or quadratic discriminants, linear SVC, basic decision trees
* Add in more traits: multi-valued regression, multilabel classification, transformers
