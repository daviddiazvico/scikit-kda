"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import numpy as np
import pickle
from scipy.stats import uniform
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from skkda.base import KernelDiscriminantAnalysis


np.random.seed(0)


###############################################################################
#  Scikit-learn integration tests
###############################################################################


def test_sklearn():
    """Tests general compatibility with Scikit-learn."""
    data = load_iris()
    predictor = Pipeline([('StandardScaler', StandardScaler()),
                          ('KernelDiscriminantAnalysis', KernelDiscriminantAnalysis()),
                          ('DummyClassifier', DummyClassifier())])
    predictor.fit(data.data, data.target)
    preds = predictor.predict(data.data)
    assert isinstance(preds, np.ndarray)
    score = predictor.score(data.data, data.target)
    assert isinstance(score, float)


def test_hyperparametersearchcv():
    """Tests compatibility with Scikit-learn's hyperparameter search CV."""
    data = load_iris()
    for search, space in ((GridSearchCV,
                           {'KernelDiscriminantAnalysis__lmb': [0.0, 0.5, 1.0]}),
                          (RandomizedSearchCV,
                           {'KernelDiscriminantAnalysis__lmb': uniform(0.0,
                                                                    1.0)})):
        predictor = Pipeline([('StandardScaler', StandardScaler()),
                              ('KernelDiscriminantAnalysis', KernelDiscriminantAnalysis()),
                              ('DummyClassifier', DummyClassifier())])
        predictor = search(predictor, space)
        assert isinstance(predictor, search)
        predictor.fit(data.data, data.target)
        preds = predictor.predict(data.data)
        assert isinstance(preds, np.ndarray)
        score = predictor.score(data.data, data.target)
        assert isinstance(score, float)


def test_ensemble():
    """Tests compatibility with Scikit-learn's ensembles."""
    data = load_iris()
    predictor = Pipeline([('StandardScaler', StandardScaler()),
                          ('KernelDiscriminantAnalysis', KernelDiscriminantAnalysis()),
                          ('DummyClassifier', DummyClassifier())])
    predictor = BaggingClassifier(base_estimator=predictor, n_estimators=3)
    assert isinstance(predictor, BaggingClassifier)
    predictor.fit(data.data, data.target)
    assert len(predictor.estimators_) == 3
    preds = predictor.predict(data.data)
    assert isinstance(preds, np.ndarray)
    score = predictor.score(data.data, data.target)
    assert isinstance(score, float)


###############################################################################
#  Serialization test
###############################################################################


def test_serialization():
    """Tests serialization capability."""
    data = load_iris()
    predictor = Pipeline([('StandardScaler', StandardScaler()),
                          ('KernelDiscriminantAnalysis', KernelDiscriminantAnalysis()),
                          ('DummyClassifier', DummyClassifier())])
    predictor.fit(data.data, data.target)
    serialized_predictor = pickle.dumps(predictor)
    deserialized_predictor = pickle.loads(serialized_predictor)
    preds = deserialized_predictor.predict(data.data)
    assert isinstance(preds, np.ndarray)
    score = deserialized_predictor.score(data.data, data.target)
    assert isinstance(score, float)
