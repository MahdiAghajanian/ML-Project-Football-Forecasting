from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if not (HERE / "p2_reimplementation").exists():
    subprocess.check_call([sys.executable, str(HERE / "prepare_p2_reimplementation.py")])

from p2_reimplementation import NGBClassifier, NGBRegressor
from p2_reimplementation.distns import MultivariateNormal, Normal, k_categorical
from ngboost.manifold import manifold
from p2_reimplementation.scores import LogScore

SEED = 42


def base():
    return DecisionTreeRegressor(max_depth=2, min_samples_leaf=5, random_state=SEED)


def test_classifier_clone_pipeline_and_determinism():
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(120, 10)); X[0, 0] = np.nan
    y = rng.integers(0, 3, size=120)
    est = Pipeline([("prep", SimpleImputer(strategy="median", add_indicator=True)),
                    ("model", NGBClassifier(Dist=k_categorical(3), Score=LogScore, Base=base(),
                                            n_estimators=12, learning_rate=.04, random_state=SEED,
                                            verbose=False))])
    a = clone(est).fit(X, y); b = clone(est).fit(X, y)
    pa = a.predict_proba(X[:15]); pb = b.predict_proba(X[:15])
    assert pa.shape == (15, 3)
    assert np.allclose(pa.sum(1), 1.0)
    assert np.all(np.isfinite(pa))
    assert np.allclose(pa, pb)


def test_normal_gradient_and_fisher_against_finite_difference():
    params = np.array([[0.4, -0.2], [np.log(1.3), np.log(.8)]])
    y = np.array([1.1, -0.7])
    NormalManifold = manifold(LogScore, Normal)
    D = NormalManifold(params)
    analytic = D.d_score(y)
    eps = 1e-6
    for obs in range(2):
        for j in range(2):
            plus = params.copy(); minus = params.copy()
            plus[j, obs] += eps; minus[j, obs] -= eps
            sp = NormalManifold(plus).score(y)[obs]
            sm = NormalManifold(minus).score(y)[obs]
            assert abs((sp - sm) / (2 * eps) - analytic[obs, j]) < 1e-5
    fisher = D.metric()
    assert np.allclose(fisher, np.swapaxes(fisher, 1, 2))
    assert np.all(np.linalg.eigvalsh(fisher) > 0)
    natural = D.grad(y, natural=True)
    direct = np.linalg.solve(fisher, analytic[..., None])[..., 0]
    assert np.allclose(natural, direct)


def test_normal_and_multivariate_prediction_and_serialization(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 8))
    y = .7 * X[:, 0] + rng.normal(size=100)
    reg = NGBRegressor(Dist=Normal, Score=LogScore, Base=base(), n_estimators=10,
                       learning_rate=.04, random_state=SEED, verbose=False).fit(X, y)
    d = reg.pred_dist(X[:10])
    assert np.all(np.asarray(d.params["scale"]) > 0)
    path = tmp_path / "normal.joblib"; joblib.dump(reg, path); loaded = joblib.load(path)
    assert np.allclose(reg.predict(X[:10]), loaded.predict(X[:10]))

    Y = np.column_stack([.4 * X[:, 0], -.3 * X[:, 1]]) + rng.multivariate_normal(
        [0, 0], [[1.0, .25], [.25, .8]], size=len(X))
    mv = NGBRegressor(Dist=MultivariateNormal(2), Score=LogScore, Base=base(),
                      n_estimators=8, learning_rate=.03, random_state=SEED, verbose=False).fit(X, Y)
    md = mv.pred_dist(X[:8]); cov = np.asarray(md.cov)
    assert cov.shape == (8, 2, 2)
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    assert np.all(np.isfinite(md.logpdf(Y[:8])))


def test_natural_gradient_switch_changes_training_path():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(100, 6)); y = rng.integers(0, 3, size=100)
    kw = dict(Dist=k_categorical(3), Score=LogScore, Base=base(), n_estimators=8,
              learning_rate=.05, random_state=SEED, verbose=False)
    nat = NGBClassifier(natural_gradient=True, **kw).fit(X, y)
    eu = NGBClassifier(natural_gradient=False, **kw).fit(X, y)
    assert not np.allclose(nat.predict_proba(X[:20]), eu.predict_proba(X[:20]))
