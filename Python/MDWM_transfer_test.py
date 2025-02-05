# flake8: noqa
from self_py_fun.ExistMLFun import MDWM
import numpy as np
import pytest


def generate_cov(Nt, Ne):
    """Generate a set of cavariances matrices for test purpose."""
    rs = np.random.RandomState(1234)
    diags = 2.0 + 0.1 * rs.randn(Nt, Ne)
    A = 2 * rs.rand(Ne, Ne) - 1
    A /= np.atleast_2d(np.sqrt(np.sum(A ** 2, 1))).T
    covmats = np.empty((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = np.dot(np.dot(A, np.diag(diags[i])), A.T)
    return covmats


def test_MDWM_init_noargs():
    m = MDWM()


def test_MDWM_init_L():
    m = MDWM(L=0.5)


def test_MDWM_init_MDM():
    m = MDWM(metric='riemann')


def test_MDWM_init_error_MDM():
    with pytest.raises(TypeError):
        MDWM(metric=42)

    with pytest.raises(ValueError):
        MDWM(L=-0.5)


def test_MDWM_fit():
    X = generate_cov(10, 3)
    y = np.array([0, 1]).repeat(5)
    X_domain = generate_cov(10, 3)
    y_domain = np.array([0, 1]).repeat(5)

    mdwm = MDWM(L=0.5)
    mdwm.fit(X, y, X_domain, y_domain)
    mdwm.predict(X)
    mdwm.predict_proba(X)

    mdwm = MDWM(L=0.5, n_jobs=2)
    mdwm.fit(X, y, X_domain, y_domain)
    mdwm.predict(X)


if __name__ == "__main__":
    test_MDWM_fit()
