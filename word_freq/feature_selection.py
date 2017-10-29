#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=================================================================
Selecting dimensionality reduction with Pipeline and GridSearchCV
=================================================================

This example constructs a pipeline that does dimensionality
reduction followed by prediction with a support vector
classifier. It demonstrates the use of ``GridSearchCV`` and
``Pipeline`` to optimize over different classes of estimators in a
single CV run -- unsupervised ``PCA`` and ``NMF`` dimensionality
reductions are compared to univariate feature selection during
the grid search.

Additionally, ``Pipeline`` can be instantiated with the ``memory``
argument to memoize the transformers within the pipeline, avoiding to fit
again the same transformers over and over.

Note that the use of ``memory`` to enable caching becomes interesting when the
fitting of a transformer is costly.
"""

###############################################################################
# Illustration of ``Pipeline`` and ``GridSearchCV``
###############################################################################
# This section illustrates the use of a ``Pipeline`` with
# ``GridSearchCV``

# Authors: Robert McGibbon, Joel Nothman, Guillaume Lemaitre

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from operator import itemgetter
def feature_selection(datas, labels):
    if len(datas)==0:
        print('datas is null')
        return
    dim = len(datas[0])
    pipe = Pipeline([
                ('reduce_dim', SelectKBest(chi2)),
                ('classify',SVC())
        ])

    N_FEATURES_OPTIONS = [int((i+1)*0.1*dim) for i in range(10)]
    C_OPTIONS = [1, 10, 100, 1000]
    param_grid = [{
    'reduce_dim': [SelectKBest(chi2)],
    'reduce_dim__k': N_FEATURES_OPTIONS,
    'classify__C': C_OPTIONS,
    'classify__kernel':['linear'],},
    {
    'reduce_dim': [SelectKBest(chi2)],
    'reduce_dim__k': N_FEATURES_OPTIONS,
    'classify__C': C_OPTIONS,
    'classify__kernel':['rbf'],
    'classify__gamma':[1e-3,1e-4]
    }
    ]
    reducer_labels = ['KBest(chi2)']

    grid = GridSearchCV(pipe, cv=3, n_jobs=-1, param_grid=param_grid)
    grid.fit(datas, labels)
    print("Best score: %0.3f" % grid.best_score_)
    print("Best parameters set found on development set:")  
    print()
    print(grid.best_params_)  
    print()  
    print("Grid scores on development set:")  
    print()  
    for params, mean_score, scores in grid.grid_scores_:  
        print("%0.3f (+/-%0.03f) for %r"  
            % (mean_score, scores.std() * 2, params))  
    print()  

    print("Detailed classification report:")  
    print()  
    print("The model is trained on the full development set.")  
    print("The scores are computed on the full evaluation set.")  
    print() 

    reduce_features = grid.best_params_['reduce_dim__k']
    C = grid.best_params_['classify__C']
    kernel = grid.best_params_['classify__kernel']
    gamma = grid.best_params_['classify__gamma']
    scores = grid.best_params_['reduce_dim'].scores_
    pvalues = grid.best_params_['reduce_dim'].pvalues_

    print('len:%d'%len(scores))
    return reduce_features, C, kernel, gamma, scores, pvalues
