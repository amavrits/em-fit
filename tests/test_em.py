"""Tests for the EM mixture fitter.

The recovery tests lean on a fixed seed and generous tolerances: EM finds a
local optimum, so they check that the fit lands near the truth, not that it
reproduces exact numbers.
"""

import numpy as np
import pytest

from em import EM, LOGLIKE_FNS
from em.loglike_fns import get_family


def sorted_by_weight(model):
    """Return (weights, params) with components in a canonical order."""
    order = np.argsort(model.params_[:, 0])
    return model.weights_[order], model.params_[order]


@pytest.fixture
def two_normals():
    rng = np.random.default_rng(20240905)
    x = np.concatenate([rng.normal(-3.0, 0.8, 4_000), rng.normal(2.0, 1.5, 6_000)])
    return x


# --- families ----------------------------------------------------------------

@pytest.mark.parametrize(
    "name,params",
    [
        ("normal", [1.0, 2.0]),
        ("lognormal", [0.5, 0.4]),
        ("exponential", [3.0]),
        ("poisson", [4.0]),
        ("gamma", [2.5, 1.7]),
    ],
)
def test_weighted_mle_recovers_parameters(name, params):
    """With uniform weights the weighted MLE is the ordinary MLE."""
    family = get_family(name)
    truth = np.asarray(params)
    x = family.rvs(truth, 200_000, np.random.default_rng(0))
    estimate = family.mle(x, np.ones_like(x), 1e-9, truth)
    assert np.allclose(estimate, truth, rtol=0.02)


DENSITY_FAMILIES = sorted(
    n for n, f in LOGLIKE_FNS.items() if not f.conditional and not f.multivariate
)


@pytest.mark.parametrize("name", DENSITY_FAMILIES)
def test_logpdf_integrates_to_one(name):
    """Each density normalizes: a crude quadrature (or sum) hits 1.

    Conditional and multivariate families are excluded: p(y | x) integrates to
    1 over y at each x, and a multivariate density needs a grid per dimension.
    Both are covered by their own tests.
    """
    family = get_family(name)
    params = {
        "normal": [0.0, 1.0],
        "lognormal": [0.0, 0.5],
        "exponential": [1.5],
        "poisson": [3.0],
        "gamma": [2.0, 1.5],
    }[name]
    params = np.asarray(params)
    if family.support == "count":
        k = np.arange(0, 60.0)
        assert np.isclose(np.sum(np.exp(family.logpdf(k, params))), 1.0, atol=1e-6)
    else:
        low = 1e-9 if family.support == "positive" else -40.0
        grid = np.linspace(low, 40.0, 400_001)
        assert np.isclose(np.trapezoid(np.exp(family.logpdf(grid, params)), grid), 1.0, atol=1e-4)


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="Unknown loglikelihood function"):
        EM("weibull")


def test_family_rejects_out_of_support_data():
    with pytest.raises(ValueError, match="strictly positive"):
        EM("gamma").train(np.array([1.0, -2.0, 3.0]), n_groups=1)
    with pytest.raises(ValueError, match="integer counts"):
        EM("poisson").train(np.array([1.0, 2.5, 3.0]), n_groups=1)


# --- fitting -----------------------------------------------------------------

def test_recovers_two_normal_components(two_normals):
    model = EM("normal", seed=0).train(two_normals, n_groups=2, n_init=5)
    weights, params = sorted_by_weight(model)
    assert model.converged_
    assert np.allclose(weights, [0.4, 0.6], atol=0.03)
    assert np.allclose(params[:, 0], [-3.0, 2.0], atol=0.1)
    assert np.allclose(params[:, 1], [0.8, 1.5], atol=0.1)


def test_loglikelihood_increases_monotonically(two_normals):
    """The defining property of EM: every iteration is an improvement."""
    model = EM("normal", seed=0).train(two_normals, n_groups=3, n_init=1)
    history = np.asarray(model.loglike_history_)
    assert np.all(np.diff(history) >= -1e-8)


def test_single_component_matches_plain_mle(two_normals):
    model = EM("normal", seed=0).train(two_normals, n_groups=1)
    assert np.allclose(model.params_[0], [two_normals.mean(), two_normals.std()], rtol=1e-5)


def test_more_restarts_never_hurt(two_normals):
    """n_init keeps the best fit, so the score cannot go down."""
    one = EM("normal", seed=0).train(two_normals, n_groups=4, n_init=1).score()
    many = EM("normal", seed=0).train(two_normals, n_groups=4, n_init=6).score()
    assert many >= one - 1e-9


def test_bic_selects_the_true_number_of_groups(two_normals):
    scores = {k: EM("normal", seed=0).train(two_normals, k, n_init=4).bic() for k in range(1, 5)}
    assert min(scores, key=scores.get) == 2


def test_weights_and_responsibilities_are_normalized(two_normals):
    model = EM("normal", seed=0).train(two_normals, n_groups=3, n_init=2)
    assert np.isclose(model.weights_.sum(), 1.0)
    assert np.allclose(model.predict_proba().sum(axis=1), 1.0)


def test_degenerate_data_does_not_blow_up():
    """Duplicated points would drive sigma to 0; the reg floor holds it back."""
    x = np.concatenate([np.zeros(50), np.ones(50) * 10.0])
    model = EM("normal", seed=0).train(x, n_groups=2, n_init=3)
    assert np.all(np.isfinite(model.params_))
    assert np.all(model.params_[:, 1] > 0.0)
    assert np.isfinite(model.score())


def test_gamma_mixture_recovers_shapes():
    rng = np.random.default_rng(11)
    x = np.concatenate([rng.gamma(2.0, 1.0, 4_000), rng.gamma(9.0, 2.0, 4_000)])
    model = EM("gamma", seed=0).train(x, n_groups=2, n_init=5)
    _, params = sorted_by_weight(model)
    assert np.allclose(params[:, 0], [2.0, 9.0], rtol=0.2)
    assert np.allclose(params[:, 1], [1.0, 2.0], rtol=0.2)


# --- inference ---------------------------------------------------------------

def test_classify_matches_generating_component(two_normals):
    model = EM("normal", seed=0).train(two_normals, n_groups=2, n_init=5)
    truth = np.concatenate([np.zeros(4_000), np.ones(6_000)])
    labels = model.classify()
    accuracy = max((labels == truth).mean(), (labels != truth).mean())  # labels are arbitrary
    assert accuracy > 0.95


def test_classify_is_the_argmax_of_the_posterior(two_normals):
    """classify() returns labels; predict_proba() returns the posterior."""
    model = EM("normal", seed=0).train(two_normals, n_groups=2, n_init=2)
    posterior = model.predict_proba()
    labels = model.classify()
    assert posterior.shape == (two_normals.size, 2)
    assert labels.shape == two_normals.shape
    assert np.array_equal(labels, np.argmax(posterior, axis=1))


def test_loglike_with_and_without_priors(two_normals):
    model = EM("normal", seed=0).train(two_normals, n_groups=2, n_init=2)
    mixture = model.loglike(with_priors=True)
    components = model.loglike(with_priors=False)
    assert mixture.shape == two_normals.shape
    assert components.shape == (two_normals.size, 2)
    from scipy.special import logsumexp
    expected = logsumexp(components + np.log(model.weights_), axis=1)
    assert np.allclose(mixture, expected)


def test_passing_x_explicitly_matches_the_cached_fit(two_normals):
    model = EM("normal", seed=0).train(two_normals, n_groups=2, n_init=2)
    assert np.allclose(model.predict_proba(), model.predict_proba(two_normals))
    assert np.allclose(model.loglike(), model.loglike(two_normals))


def test_accepts_column_vector(two_normals):
    column = EM("normal", seed=0).train(two_normals[:, None], n_groups=2, n_init=2)
    flat = EM("normal", seed=0).train(two_normals, n_groups=2, n_init=2)
    assert np.allclose(column.params_, flat.params_)


def test_sample_round_trip(two_normals):
    model = EM("normal", seed=0).train(two_normals, n_groups=2, n_init=5)
    x, labels = model.sample(20_000, seed=3)
    assert x.shape == labels.shape == (20_000,)
    refit = EM("normal", seed=0).train(x, n_groups=2, n_init=5)
    assert np.allclose(np.sort(refit.params_[:, 0]), np.sort(model.params_[:, 0]), atol=0.15)


# --- custom log-likelihoods --------------------------------------------------

def laplace_loglike(x, p):
    return -np.log(2.0 * p[1]) - np.abs(x - p[0]) / p[1]


def test_custom_loglike_fits_a_laplace_mixture():
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.laplace(0.0, 1.0, 3_000), rng.laplace(8.0, 2.0, 3_000)])
    model = EM(
        laplace_loglike,
        n_params=2,
        param_bounds=[(None, None), (1e-6, None)],
        init_params=[0.0, 1.0],
        seed=0,
    ).train(x, n_groups=2, n_init=3)
    assert model.custom_loglike
    _, params = sorted_by_weight(model)
    assert np.allclose(params[:, 0], [0.0, 8.0], atol=0.2)
    assert np.allclose(params[:, 1], [1.0, 2.0], atol=0.2)


def test_init_params_is_used_as_the_starting_guess():
    """Regression: init_params was validated but never reached the optimizer."""
    seen = []

    def recording_loglike(x, p):
        seen.append(np.asarray(p, dtype=float).copy())
        return laplace_loglike(x, p)

    rng = np.random.default_rng(5)
    x = np.concatenate([rng.laplace(0.0, 1.0, 500), rng.laplace(8.0, 2.0, 500)])
    EM(
        recording_loglike,
        n_params=2,
        param_bounds=[(None, None), (1e-6, None)],
        init_params=[5.0, 3.0],
        seed=0,
    ).train(x, n_groups=2, n_iters=2)
    assert np.allclose(seen[0], [5.0, 3.0])


def test_custom_loglike_requires_n_params():
    with pytest.raises(ValueError, match="n_params is required"):
        EM(laplace_loglike)


def test_custom_loglike_must_be_elementwise():
    model = EM(lambda x, p: np.sum(-((x - p[0]) ** 2)), n_params=1, seed=0)
    with pytest.raises(ValueError, match="elementwise"):
        model.train(np.arange(10.0), n_groups=2)


def test_non_callable_non_string_rejected():
    with pytest.raises(TypeError, match="family name or a callable"):
        EM(42)


# --- API guards --------------------------------------------------------------

def test_methods_require_a_fit():
    model = EM("normal")
    with pytest.raises(RuntimeError, match="not fitted"):
        model.classify(np.arange(5.0))
    with pytest.raises(RuntimeError, match="not fitted"):
        model.loglike(np.arange(5.0))


def test_train_rejects_bad_input():
    x = np.arange(10.0)
    with pytest.raises(TypeError, match="requires X"):
        EM("normal").train(None, n_groups=2)
    with pytest.raises(ValueError, match="at least 1"):
        EM("normal").train(x, n_groups=0)
    with pytest.raises(ValueError, match="exceeds"):
        EM("normal").train(x, n_groups=99)
    with pytest.raises(ValueError, match="NaN"):
        EM("normal").train(np.array([1.0, np.nan, 3.0]), n_groups=1)
    with pytest.raises(ValueError, match="empty"):
        EM("normal").train(np.array([]), n_groups=1)
    with pytest.raises(ValueError, match="univariate"):
        EM("normal").train(np.zeros((10, 3)), n_groups=2)


def test_seed_makes_restarts_reproducible(two_normals):
    a = EM("normal", seed=42).train(two_normals, n_groups=3, n_init=4)
    b = EM("normal", seed=42).train(two_normals, n_groups=3, n_init=4)
    assert np.allclose(a.params_, b.params_)
    assert np.allclose(a.weights_, b.weights_)


def test_context_manager_releases_data_but_keeps_the_fit(two_normals):
    with EM("normal", seed=0) as model:
        model.train(two_normals, n_groups=2, n_init=2)
        params = model.params_.copy()
    assert model.X_ is None
    assert np.allclose(model.params_, params)
    with pytest.raises(RuntimeError, match="No training data"):
        model.classify()
    assert model.classify(two_normals).shape == two_normals.shape


def test_n_iters_caps_the_run(two_normals):
    model = EM("normal", seed=0).train(two_normals, n_groups=2, n_iters=3)
    assert model.n_iter_ == 3
    assert not model.converged_


# --- the specialized logsumexp -----------------------------------------------

def test_rows_logsumexp_matches_scipy():
    """The hand-rolled reduction must agree with scipy's on every shape."""
    from scipy.special import logsumexp

    from em.em import _rows_logsumexp

    rng = np.random.default_rng(0)
    cases = {
        "ordinary": rng.normal(size=(200, 3)) * 30,
        "single component": rng.normal(size=(50, 1)),
        "extreme underflow": np.full((5, 4), -1e300),
        "identical entries": np.zeros((10, 3)),
        "partial -inf": np.array([[-np.inf, -2.0], [-1.0, -np.inf]]),
        "all -inf row": np.array([[-np.inf, -np.inf], [-1.0, -2.0]]),
    }
    for name, a in cases.items():
        assert np.allclose(_rows_logsumexp(a), logsumexp(a, axis=1)), name


def test_rows_logsumexp_is_quiet_on_minus_inf():
    """An all -inf row is -inf, not nan, and raises no warning."""
    import warnings

    from em.em import _rows_logsumexp

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _rows_logsumexp(np.array([[-np.inf, -np.inf], [-1.0, -2.0]]))
    assert out[0] == -np.inf
    assert np.isfinite(out[1])


# --- mixture of regressions --------------------------------------------------

@pytest.fixture
def two_lines():
    """y = 2 + 3x and y = 10 - 1.5x over the same x-range, 1500 points each."""
    rng = np.random.default_rng(4)
    n = 1_500
    xa, xb = rng.uniform(-3, 3, n), rng.uniform(-3, 3, n)
    ya = 2.0 + 3.0 * xa + rng.normal(0, 0.6, n)
    yb = 10.0 - 1.5 * xb + rng.normal(0, 0.9, n)
    return np.concatenate([xa, xb]), np.concatenate([ya, yb]), n


def test_recovers_two_regression_lines(two_lines):
    x, y, _ = two_lines
    model = EM("linear-regression", seed=0).train(x, n_groups=2, n_init=8, y=y)
    order = np.argsort(model.params_[:, 0])          # by intercept
    params = model.params_[order]
    assert model.converged_
    assert np.allclose(params[0], [2.0, 3.0, 0.6], atol=0.15)
    assert np.allclose(params[1], [10.0, -1.5, 0.9], atol=0.15)
    assert np.allclose(model.weights_, 0.5, atol=0.03)


def test_regression_classify_recovers_the_generating_line(two_lines):
    x, y, n = two_lines
    model = EM("linear-regression", seed=0).train(x, n_groups=2, n_init=8, y=y)
    truth = np.concatenate([np.zeros(n), np.ones(n)])
    labels = model.classify()
    assert max((labels == truth).mean(), (labels != truth).mean()) > 0.9


def test_regression_with_several_features():
    rng = np.random.default_rng(7)
    n = 1_500
    X = rng.uniform(-2, 2, size=(2 * n, 2))
    first = np.arange(2 * n) < n
    y = np.where(first, 1.0 + 2.0 * X[:, 0] - X[:, 1], -4.0 - X[:, 0] + 3.0 * X[:, 1])
    y = y + rng.normal(0, 0.5, 2 * n)
    model = EM("linear-regression", seed=0).train(X, n_groups=2, n_init=10, y=y)
    params = model.params_[np.argsort(model.params_[:, 0])]
    assert np.allclose(params[0], [-4.0, -1.0, 3.0, 0.5], atol=0.15)
    assert np.allclose(params[1], [1.0, 2.0, -1.0, 0.5], atol=0.15)


def test_n_free_params_tracks_the_feature_count():
    """A regression has d + 2 parameters, so AIC/BIC must follow the width."""
    rng = np.random.default_rng(0)
    n = 400
    for d in (1, 2, 4):
        X = rng.uniform(-2, 2, size=(n, d))
        y = X.sum(axis=1) + rng.normal(0, 0.5, n)
        model = EM("linear-regression", seed=0).train(X, n_groups=2, n_init=2, y=y)
        assert model.params_.shape == (2, d + 2)
        assert model.n_free_params() == (2 - 1) + 2 * (d + 2)
        assert np.isfinite(model.bic())


def test_conditional_family_requires_y(two_lines):
    x, y, _ = two_lines
    with pytest.raises(ValueError, match=r"models p\(y \| x\)"):
        EM("linear-regression").train(x, n_groups=2)


def test_density_family_rejects_y(two_lines):
    x, y, _ = two_lines
    with pytest.raises(ValueError, match="y is not used"):
        EM("normal").train(x, n_groups=2, y=y)


def test_regression_rejects_mismatched_lengths(two_lines):
    x, y, _ = two_lines
    with pytest.raises(ValueError, match="rows but y has"):
        EM("linear-regression").train(x, n_groups=2, y=y[:-5])


def test_regression_rejects_a_different_width_at_predict_time():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(300, 2))
    y = X.sum(axis=1) + rng.normal(0, 0.5, 300)
    model = EM("linear-regression", seed=0).train(X, n_groups=2, n_init=2, y=y)
    with pytest.raises(ValueError, match="fitted on 2 features"):
        model.predict_proba(X[:, :1], y=y)


def test_sample_is_unavailable_for_a_conditional_family(two_lines):
    x, y, _ = two_lines
    model = EM("linear-regression", seed=0).train(x, n_groups=2, n_init=2, y=y)
    with pytest.raises(NotImplementedError, match="needs predictors"):
        model.sample(10)


def test_regression_loglike_shapes(two_lines):
    x, y, _ = two_lines
    model = EM("linear-regression", seed=0).train(x, n_groups=2, n_init=2, y=y)
    assert model.loglike().shape == x.shape
    assert model.loglike(with_priors=False).shape == (x.size, 2)
    assert np.allclose(model.loglike(x, y=y), model.loglike())


# --- multivariate normal -----------------------------------------------------

def mvn_params(mean, cov):
    """Pack a mean and covariance into the flat parameter vector."""
    cov = np.asarray(cov, dtype=float)
    return np.concatenate([np.asarray(mean, dtype=float), cov[np.tril_indices(cov.shape[0])]])


@pytest.fixture
def two_blobs():
    """Two 2-D Gaussians with different, correlated covariances."""
    rng = np.random.default_rng(11)
    a = rng.multivariate_normal([-2.0, 1.0], [[1.0, 0.7], [0.7, 1.0]], size=3_000)
    b = rng.multivariate_normal([3.0, 2.0], [[2.0, -1.1], [-1.1, 1.5]], size=2_000)
    return np.vstack([a, b])


def test_mvn_logpdf_matches_scipy():
    from scipy.stats import multivariate_normal

    family = get_family("multivariate-normal")
    mean, cov = [1.0, -2.0], [[2.0, 0.9], [0.9, 1.0]]
    x = np.random.default_rng(0).multivariate_normal(mean, cov, size=500)
    assert np.allclose(family.logpdf(x, mvn_params(mean, cov)), multivariate_normal(mean, cov).logpdf(x))


def test_mvn_density_integrates_to_one():
    """Two-dimensional quadrature over a grid wide enough to hold the mass."""
    family = get_family("multivariate-normal")
    p = mvn_params([0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]])
    axis = np.linspace(-8.0, 8.0, 601)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    density = np.exp(family.logpdf(np.column_stack([grid_x.ravel(), grid_y.ravel()]), p))
    total = np.trapezoid(np.trapezoid(density.reshape(grid_x.shape), axis, axis=1), axis)
    assert np.isclose(total, 1.0, atol=1e-4)


def test_mvn_weighted_mle_recovers_parameters():
    family = get_family("multivariate-normal")
    truth = mvn_params([1.0, -2.0], [[2.0, 0.9], [0.9, 1.0]])
    x = family.rvs(truth, 200_000, np.random.default_rng(0))
    assert np.allclose(family.mle(x, np.ones(len(x)), 1e-9, truth), truth, atol=0.02)


def test_recovers_two_2d_components(two_blobs):
    model = EM("multivariate-normal", seed=0).train(two_blobs, n_groups=2, n_init=5)
    order = np.argsort(model.params_[:, 0])
    params = model.params_[order]
    assert model.converged_
    assert np.allclose(model.weights_[order], [0.6, 0.4], atol=0.03)
    assert np.allclose(params[0], mvn_params([-2.0, 1.0], [[1.0, 0.7], [0.7, 1.0]]), atol=0.12)
    assert np.allclose(params[1], mvn_params([3.0, 2.0], [[2.0, -1.1], [-1.1, 1.5]]), atol=0.15)


def test_2d_classification_matches_the_generating_blob(two_blobs):
    model = EM("multivariate-normal", seed=0).train(two_blobs, n_groups=2, n_init=5)
    truth = np.concatenate([np.zeros(3_000), np.ones(2_000)])
    labels = model.classify()
    assert max((labels == truth).mean(), (labels != truth).mean()) > 0.95


def test_mvn_free_params_counts_only_the_triangle():
    """A d-dimensional normal has d + d(d+1)/2 free parameters, not d + d^2."""
    rng = np.random.default_rng(0)
    for d in (1, 2, 4):
        x = rng.normal(size=(600, d))
        model = EM("multivariate-normal", seed=0).train(x, n_groups=2, n_init=2)
        expected = d + d * (d + 1) // 2
        assert model.params_.shape == (2, expected)
        assert model.n_free_params() == (2 - 1) + 2 * expected
        assert np.isfinite(model.bic())


def test_2d_sample_round_trip(two_blobs):
    model = EM("multivariate-normal", seed=0).train(two_blobs, n_groups=2, n_init=5)
    x, labels = model.sample(20_000, seed=3)
    assert x.shape == (20_000, 2)
    assert labels.shape == (20_000,)
    refit = EM("multivariate-normal", seed=0).train(x, n_groups=2, n_init=5)
    assert np.allclose(
        np.sort(refit.params_[:, 0]), np.sort(model.params_[:, 0]), atol=0.15
    )


def test_mvn_rejects_a_different_width_at_predict_time(two_blobs):
    model = EM("multivariate-normal", seed=0).train(two_blobs, n_groups=2, n_init=2)
    with pytest.raises(ValueError, match="fitted on 2 features"):
        model.predict_proba(two_blobs[:, :1])


# --- anchors (semi-supervised EM) --------------------------------------------

@pytest.fixture
def overlapping_pair():
    """Two unit normals 1.2 apart, with the generating component of each point."""
    rng = np.random.default_rng(31)
    n = 3_000
    x = np.concatenate([rng.normal(0.0, 1.0, n), rng.normal(1.2, 1.0, n)])
    truth = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return x, truth


def anchor(truth, fraction, seed=0):
    """Reveal a random fraction of the true labels; -1 everywhere else."""
    rng = np.random.default_rng(seed)
    labels = np.full(truth.shape, -1)
    picked = rng.choice(truth.size, size=int(fraction * truth.size), replace=False)
    labels[picked] = truth[picked]
    return labels


def test_anchors_fix_the_component_indices(overlapping_pair):
    """Component k is the one the anchors call k, so labels stop being arbitrary."""
    x, truth = overlapping_pair
    labels = anchor(truth, 0.03)
    model = EM("normal", seed=0).train(x, n_groups=2, n_init=4, labels=labels)
    assert model.params_[0, 0] < model.params_[1, 0]
    assert np.allclose(model.params_[:, 0], [0.0, 1.2], atol=0.15)
    assert (model.classify() == truth).mean() > 0.7  # no need to try both orderings

    swapped = np.where(labels >= 0, 1 - labels, -1)
    other = EM("normal", seed=0).train(x, n_groups=2, n_init=4, labels=swapped)
    assert np.allclose(other.params_[::-1], model.params_)
    assert np.allclose(other.weights_[::-1], model.weights_)


def test_anchored_rows_are_pinned_in_the_training_posterior(overlapping_pair):
    x, truth = overlapping_pair
    labels = anchor(truth, 0.05)
    model = EM("normal", seed=0).train(x, n_groups=2, n_init=2, labels=labels)
    rows = np.flatnonzero(labels >= 0)
    cached = model.predict_proba()
    assert np.allclose(cached[rows, labels[rows]], 1.0)
    assert np.allclose(cached.sum(axis=1), 1.0)
    # The unconstrained posterior for the same points is not degenerate.
    free = model.predict_proba(x)
    assert not np.allclose(free[rows, labels[rows]], 1.0)
    assert np.array_equal(model.labels_, labels)


def test_anchored_loglikelihood_increases_monotonically(overlapping_pair):
    """Pinning is a proper likelihood, so EM's guarantee still holds."""
    x, truth = overlapping_pair
    model = EM("normal", seed=0).train(x, n_groups=2, n_init=1, labels=anchor(truth, 0.1))
    history = np.asarray(model.loglike_history_)
    assert np.all(np.diff(history) >= -1e-8)
    assert model.converged_


def test_anchored_objective_matches_a_hand_computation(overlapping_pair):
    x, truth = overlapping_pair
    labels = anchor(truth, 0.1)
    model = EM("normal", seed=0).train(x, n_groups=2, n_init=1, labels=labels)
    log_joint = model.loglike(with_priors=False) + np.log(model.weights_)
    rows = np.flatnonzero(labels >= 0)
    expected = model.loglike().copy()
    expected[rows] = log_joint[rows, labels[rows]]
    _, total = model.e_step(x, model.weights_, model.params_, labels)
    assert np.isclose(total, expected.sum())
    assert np.isclose(model.loglike_history_[-1], expected.sum())


def test_anchors_rescue_a_fit_the_data_alone_cannot_identify():
    """Two components with the same mean and different spread.

    Unsupervised EM has no way to tell which index is 'wide'; anchors decide.
    """
    rng = np.random.default_rng(8)
    n = 2_500
    x = np.concatenate([rng.normal(0.0, 1.0, n), rng.normal(0.0, 4.0, n)])
    truth = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    model = EM("normal", seed=0).train(x, n_groups=2, n_init=3, labels=anchor(truth, 0.05))
    assert np.allclose(model.params_[:, 1], [1.0, 4.0], atol=0.3)


def test_fully_anchored_fit_is_the_supervised_mle(overlapping_pair):
    """With every point labeled there is nothing to infer: one M-step, exact."""
    x, truth = overlapping_pair
    model = EM("normal", seed=0).train(x, n_groups=2, labels=truth)
    for k in (0, 1):
        part = x[truth == k]
        assert np.allclose(model.params_[k], [part.mean(), part.std()], rtol=1e-6)
    assert np.allclose(model.weights_, np.bincount(truth) / truth.size)


def test_unlabeled_everywhere_is_plain_em(overlapping_pair):
    x, _ = overlapping_pair
    plain = EM("normal", seed=0).train(x, n_groups=2, n_init=2)
    empty = EM("normal", seed=0).train(x, n_groups=2, n_init=2, labels=np.full(x.size, -1))
    assert np.allclose(plain.params_, empty.params_)
    assert empty.labels_ is None


def test_labels_accept_nan_and_none_for_unknown(overlapping_pair):
    x, truth = overlapping_pair
    labels = anchor(truth, 0.05)
    reference = EM("normal", seed=0).train(x, n_groups=2, labels=labels)
    as_float = np.where(labels < 0, np.nan, labels).astype(float)
    as_object = [None if v < 0 else int(v) for v in labels]
    for variant in (as_float, as_object):
        model = EM("normal", seed=0).train(x, n_groups=2, labels=variant)
        assert np.allclose(model.params_, reference.params_)


def test_labels_are_validated(overlapping_pair):
    x, truth = overlapping_pair
    labels = anchor(truth, 0.05)
    with pytest.raises(ValueError, match="entries but X has"):
        EM("normal").train(x, n_groups=2, labels=labels[:-1])
    with pytest.raises(ValueError, match=r"lie in \[0, 2\)"):
        EM("normal").train(x, n_groups=2, labels=np.where(labels == 0, 2, labels))
    with pytest.raises(ValueError, match=r"lie in \[0, 2\)"):
        EM("normal").train(x, n_groups=2, labels=np.where(labels == 0, -2, labels))
    with pytest.raises(ValueError, match="integer component indices"):
        EM("normal").train(x, n_groups=2, labels=np.where(labels == 0, 0.5, labels).astype(float))
    with pytest.raises(ValueError, match="integer component indices"):
        EM("normal").train(x, n_groups=2, labels=["a"] * x.size)


def test_anchors_are_reproducible_and_reported(overlapping_pair):
    x, truth = overlapping_pair
    labels = anchor(truth, 0.05)
    a = EM("normal", seed=3).train(x, n_groups=2, n_init=3, labels=labels)
    b = EM("normal", seed=3).train(x, n_groups=2, n_init=3, labels=labels)
    assert np.allclose(a.params_, b.params_)
    assert f"{int((labels >= 0).sum())} anchors" in a.summary()


def test_anchors_seed_the_regression_lines(two_lines):
    """A few labeled points per line pin which index each line gets."""
    x, y, n = two_lines
    labels = np.full(2 * n, -1)
    labels[:15] = 0
    labels[n:n + 15] = 1
    model = EM("linear-regression", seed=0).train(x, n_groups=2, n_init=3, y=y, labels=labels)
    assert np.allclose(model.params_[0], [2.0, 3.0, 0.6], atol=0.15)
    assert np.allclose(model.params_[1], [10.0, -1.5, 0.9], atol=0.15)
    truth = np.concatenate([np.zeros(n), np.ones(n)])
    assert (model.classify() == truth).mean() > 0.9


def test_anchors_on_a_multivariate_mixture(two_blobs):
    labels = np.full(two_blobs.shape[0], -1)
    labels[:25] = 1        # the first blob is index 1 this time
    labels[3_000:3_025] = 0
    model = EM("multivariate-normal", seed=0).train(two_blobs, n_groups=2, n_init=3, labels=labels)
    assert np.allclose(model.params_[1, :2], [-2.0, 1.0], atol=0.12)
    assert np.allclose(model.params_[0, :2], [3.0, 2.0], atol=0.15)


def test_anchors_with_a_custom_loglike(overlapping_pair):
    x, truth = overlapping_pair
    model = EM(
        laplace_loglike,
        n_params=2,
        param_bounds=[(None, None), (1e-6, None)],
        init_params=[0.0, 1.0],
        seed=0,
    ).train(x, n_groups=2, n_init=2, labels=anchor(truth, 0.05))
    assert model.params_[0, 0] < model.params_[1, 0]
    assert np.all(np.isfinite(model.params_))


def test_context_manager_drops_the_anchors(overlapping_pair):
    x, truth = overlapping_pair
    with EM("normal", seed=0) as model:
        model.train(x, n_groups=2, labels=anchor(truth, 0.05))
    assert model.labels_ is None
    assert model.X_ is None
