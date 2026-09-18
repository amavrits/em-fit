# em-fit

Expectation-maximization for finite mixture models.

Fit a mixture of `k` components to your data, get back the mixing weights, the
per-component parameters, and an assignment of every observation to a
component. Seven families ship built in — five univariate densities, a full
multivariate normal, and a linear regression that turns EM into clusterwise
regression.

**Any density you can write as a Python function works as a component.** Pass a
`loglike_fn(x, p)` and EM fits a mixture of it, with the M-step handled for you.
`sklearn.mixture` has no equivalent hook — it fits Gaussians and nothing else —
so this is the main reason to reach for `em`. See
[Custom log-likelihoods](#custom-log-likelihoods).

```python
import numpy as np
from em import EM

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(-3, 0.8, 4_000), rng.normal(2, 1.5, 6_000)])

model = EM("normal", seed=0).train(x, n_groups=2, n_init=5)
print(model.summary())
# EM(normal) with 2 groups, converged
#   group 1: weight=0.5969, mu=2.051, sigma=1.473
#   group 0: weight=0.4031, mu=-3.003, sigma=0.8039
#   mean loglike = -2.198174

labels = model.classify()          # hard assignment, shape (n_samples,)
posterior = model.predict_proba()  # responsibilities, shape (n_samples, 2)
```

## Install

```bash
pip install em-fit       # or: uv add em-fit
```

The package is imported as `em`. To work on it from a clone:

```bash
uv sync          # runtime + dev + examples, project installed editable
uv run pytest    # 70 tests
```

## Built-in families

| name | parameters | data |
|---|---|---|
| `normal` | `mu`, `sigma` | real, 1-D |
| `lognormal` | `mu`, `sigma` (of `log x`) | positive, 1-D |
| `exponential` | `rate` | positive, 1-D |
| `poisson` | `lam` | non-negative integers, 1-D |
| `gamma` | `shape`, `scale` | positive, 1-D |
| `multivariate-normal` | `mu1..mud`, lower triangle of the covariance | real, `d`-D |
| `linear-regression` | `intercept`, `beta1..betad`, `sigma` | `(X, y)`, conditional |

Each carries a *weighted* MLE, so the M-step is solved directly rather than by
generic optimization — exactly for the first four, by Newton on the shape for
`gamma`, by a Cholesky factorization for `multivariate-normal`, and by weighted
least squares for `linear-regression`. Data outside a family's support is
rejected up front.

## Multivariate mixtures

`multivariate-normal` takes `(n_samples, n_features)` and fits a full
covariance per component. Only the lower triangle is stored, so the parameter
vector has exactly `d + d(d+1)/2` entries and the count that AIC/BIC use is the
number of *free* parameters, not a padded square matrix:

```python
model = EM("multivariate-normal", seed=0).train(X, n_groups=3, n_init=5)
print(model.summary())
#   group 1: weight=0.4082, mu1=-2.507, mu2=1.034, cov11=1.031, cov21=0.7341, cov22=0.9643
```

Component covariances have to stay positive definite; `reg` is added to the
diagonal at every M-step to keep them there.

## Mixtures of regressions

`linear-regression` is a *conditional* family: it models `p(y | x)` rather than
a density over `x`, so it needs paired data and takes an explicit `y`. This is
clusterwise (switching) regression — several linear relationships hiding in one
scatter, recovered together with the assignment of points to lines.

```python
model = EM("linear-regression", seed=0).train(X, n_groups=2, n_init=8, y=y)
print(model.summary())
#   group 1: weight=0.5032, intercept=0.04102, beta1=-2.996, sigma=0.6924
#   group 0: weight=0.4968, intercept=0.00969, beta1=3.003,  sigma=0.6951

labels = model.classify(y=y)   # which line each point came from
```

`X` is `(n_samples,)` or `(n_samples, n_features)`; `y` is `(n_samples,)`.
Because a responsibility is `r_ik ∝ w_k · N(y_i | x_iᵀβ_k, σ_k)`, every
inference method needs `y` too — `classify`, `predict_proba`, `loglike`,
`score`, `aic` and `bic` all take it as a keyword.

A few consequences worth knowing:

- **Parameter count follows the data width.** A regression has `d + 2`
  parameters, so `n_free_params()` — and therefore AIC/BIC — depends on how
  many features you passed. Same for `multivariate-normal`. The univariate
  density families are fixed-width.
- **Passing `y` to a density family raises**, and omitting it for a conditional
  family raises. Neither is silently accepted.
- **Initialization is different.** No ordering of `x` separates regression
  lines, so the quantile split is meaningless here; components are seeded by
  fitting random subsamples. Mixtures of regressions are badly multimodal, so
  `n_init` matters much more than it does for the density families.
- **`sample()` is unavailable.** Drawing from `p(y | x)` needs predictors to
  condition on; generate `X` yourself and draw `y ~ Normal(X @ beta, sigma)`.

## Anchors: semi-supervised EM

If the component of some observations is already known, pass those labels as
*anchors*. Every other entry is `-1` (or `NaN`, or `None`):

```python
labels = np.full(x.size, -1)
labels[known_idx] = known_component        # a few percent is plenty

model = EM("normal", seed=0).train(x, n_groups=2, n_init=4, labels=labels)
print(model.summary())
# EM(normal) with 2 groups, converged, 200 anchors
```

An anchored observation has its responsibility pinned to its component in every
E-step, and contributes the complete-data term `log w_k + log f(x | p_k)` to
the objective instead of the marginal `log Σ_k`. That objective is still a
likelihood, so the trace stays monotone and convergence is unchanged. It works
with every family, including custom log-likelihoods and mixtures of regressions.

What a few anchors buy:

- **Named components.** Component `k` *is* the one the anchors call `k`, so
  `classify()` needs no relabeling afterwards and a group index means the same
  thing across fits and datasets.
- **The right optimum.** Where the unlabeled likelihood has several near-equal
  optima (heavily overlapping components, mixtures of regressions), the anchors
  pick the one that agrees with the labels. In `examples/mixture_anchored.py`
  two unit normals 1.5 apart put plain EM on a lopsided optimum at 74% accuracy
  in 697 iterations; 2% anchors take it to within half a point of the
  77.2% Bayes rate, in 203.
- **Identifiability the data cannot supply.** Two components with the same
  mean and different spread, or a small component under a large one, can only
  be told apart by convention without labels.

Every initialization is aligned to the anchors before the first M-step: the
quantile or k-means++ clusters are permuted to agree with the labels as much as
possible, and for a regression each anchored component is seeded from its own
anchors.

Two things to know. `predict_proba()` with no argument returns the training
responsibilities, in which anchors are exactly one-hot; pass `X` explicitly to
get the unconstrained posterior the fitted mixture assigns them. And
`loglike()`, `score()`, `aic()` and `bic()` are the marginal mixture density
regardless of labels, so they stay comparable across label sets; the
semi-supervised objective the fit maximized is `loglike_history_`, and is what
`summary()` reports. A fully labeled sample skips inference entirely and gives
the supervised per-component MLE.

## Choosing the number of groups

EM maximizes the likelihood, which always improves with more components, so the
component count is chosen outside the fit:

```python
scores = {k: EM("normal", seed=0).train(x, k, n_init=4).bic() for k in range(1, 6)}
best = min(scores, key=scores.get)
```

`aic()`, `bic()`, `score()` (mean log-likelihood) and `n_free_params()` are all
available on a fitted model.

## Custom log-likelihoods

Pass a callable `loglike_fn(x, p)` that returns the log-density of every
observation in `x` under one component with parameter vector `p`. The number of
parameters cannot be inferred from a function, so `n_params` is required; bounds
and a starting guess are optional but make the numerical M-step behave.

```python
def laplace(x, p):
    return -np.log(2 * p[1]) - np.abs(x - p[0]) / p[1]

model = EM(
    laplace,
    n_params=2,
    param_bounds=[(None, None), (1e-6, None)],  # scale must stay positive
    init_params=[0.0, 1.0],
    seed=0,
).train(x, n_groups=2, n_init=3)
```

The callable must evaluate elementwise over `x` and return an array shaped like
it; this is checked before fitting starts. Built-in families use their closed
form, custom ones are maximized with L-BFGS-B.

Everything else keeps working: `classify`, `predict_proba`, `aic`/`bic`,
restarts and the convergence trace do not care where the log-density came from.
That means heavy tails, truncated or censored supports, circular data, or a
likelihood specific to your instrument are all a dozen lines away, rather than
a fork of somebody's mixture library.

## Notes on the fit

- **Local optima.** EM converges to a local maximum. The first restart starts
  from a deterministic ordered split; further restarts (`n_init > 1`) use
  k-means++ seeding, and the best fit wins. Raise `n_init` when components
  overlap.
- **Reproducibility.** `seed` fixes the restarts, so a given `(data, seed,
  n_init)` always gives the same fit.
- **Degenerate components.** A component collapsing onto a single point sends
  the likelihood to infinity; the `reg` floor on the variance/scale prevents it.
- **Convergence.** Declared when the mean log-likelihood changes by less than
  `tol`. `converged_`, `n_iter_` and `loglike_history_` record what happened;
  the history is monotonically non-decreasing, which is the property that makes
  EM EM.
- **The univariate families stay univariate.** `normal`, `lognormal`,
  `exponential`, `poisson` and `gamma` take `(n_samples,)` or `(n_samples, 1)`.
  For `d > 1` use `multivariate-normal`, which carries the covariance
  parameterization the univariate families do not have.

## Why this implementation

**It fits what `sklearn.mixture` will not.** The component family is the reason
to reach for `em`. sklearn fits Gaussians; here a component is any log-density
you can write as a Python function, alongside seven built-ins — a univariate
normal, four non-Gaussian densities, a full multivariate normal, and a
conditional regression family that turns EM into clusterwise regression. Heavy tails,
counts, positive-only data, a truncated or censored support, a likelihood
specific to your instrument: each is a dozen lines rather than a fork of
somebody's mixture library.

**The M-step is solved, not searched.** Every built-in carries a *weighted* MLE
in closed form — exact for `normal`, `lognormal`, `exponential` and `poisson`,
Newton on the shape for `gamma`, a Cholesky factorization for
`multivariate-normal`, weighted least squares for `linear-regression`. A
generic optimizer only appears for a custom callable, and even then it is
seeded from a pooled fit of the whole sample rather than from `init_params`
directly, so no component starts from a wild guess.

**The E-step is specialized.** `scipy.special.logsumexp` carries machinery this
loop never uses — complex input, `b` weights, an optional sign return, a masked
search for the largest real part. The row-wise version in `em/em.py` drops all
of it and measures **2.5–2.8× faster** on the shapes an E-step actually sees
(20k×3 through 100k×4). It runs once per iteration per restart, so on the
large-`n` cases it is a large share of the total.

**Parameter counts are honest.** A `d`-dimensional covariance is stored as its
lower triangle, so a component has exactly `d + d(d+1)/2` parameters and
`n_free_params()` counts free ones rather than the cells of a padded square
matrix. AIC and BIC are computed off that number, which is what makes the
model-selection panels in the examples trustworthy.

**Reproducible by construction.** `seed` fixes the restarts, so a given
`(data, seed, n_init)` always gives the same fit — both example scripts are
bit-identical across runs. `converged_`, `n_iter_` and `loglike_history_`
record what the fit did, and `e_step`/`m_step` are public, so a suspicious
result can be stepped through by hand.

**Small surface.** ~1,000 lines over two modules, depending on numpy, scipy and
tqdm. No pandas, no sklearn at runtime; matplotlib and scikit-learn live in the
`examples` and `bench` groups.

### Benchmark on the example datasets

`benchmarks/on_examples.py` runs the head-to-head on the real data the examples
fit, where there is no ground truth to recover, so the questions are whether
the two implementations reach the *same* answer and what that costs:

```bash
uv run --group bench python benchmarks/on_examples.py
```

Matched settings (`tol=1e-8`, `reg=1e-6`, `n_init=24`, k-means++ seeding, full
covariance), best of 3, on an M-series laptop:

| dataset | shape | groups by BIC | BIC | mean loglike | labels agree | one fit at chosen `k` | whole BIC sweep |
|---|---|---|---|---|---|---|---|
| galaxy velocities | 82 × 1 | **3** / **3** | 441.612 / 441.612 | −2.4778 / −2.4778 | 1.000 | **0.018 s** / 0.059 s | **1.00 s** / 3.66 s |
| Old Faithful | 272 × 2 | **2** / **2** | 2322.19 / 2322.19 | −4.1554 / −4.1554 | 1.000 | **0.027 s** / 0.029 s | 6.30 s / **4.88 s** |

Cells are `em` / sklearn. The first four columns are the ones that matter and
they are ties: both implementations choose the same component count by BIC,
land on the same log-likelihood to six figures, and produce **identical
labellings** — adjusted Rand 1.000, not 0.999. These are the same two fits.

The timings split by dimension, and the per-`k` breakdown says why:

- **1-D is `em`'s case.** On the galaxy velocities it is 2.5–4× faster at every
  candidate `k`. The univariate M-step is closed-form arithmetic on two scalars
  per component, with no matrix work to amortize.
- **2-D is a draw at the `k` you keep, and a loss past it.** On Old Faithful
  `em` wins at `k ≤ 3` (0.004/0.027/0.44 s against 0.008/0.028/0.48 s) and
  loses from `k = 4` on (1.76/2.01/2.12 s against 1.42/1.40/1.60 s), which is
  what drags the sweep. The M-step loops over components in Python and takes a
  Cholesky per component per iteration, where sklearn vectorizes the covariance
  update across all of them; at n=272 that loop overhead is the whole
  difference. The sweep total is therefore dominated by the four-to-six-group
  fits — exactly the ones BIC rejects and nobody keeps.

So: `em` is faster on the fit you actually run in both examples, and slower
only when fitting 2-D models the data does not support. The synthetic
benchmark in [Benchmark](#benchmark) covers the large-`n` regime, where it wins
every case.

`mixture_regression.py` has no row here because there is nothing to compare
against — `sklearn.mixture` does not fit mixtures of regressions.

## Examples

```bash
uv run python examples/mixture_1d.py          # 1-D density mixture, real data
uv run python examples/mixture_2d.py          # 2-D density mixture, real data
uv run python examples/mixture_regression.py  # mixture of regressions
uv run python examples/mixture_anchored.py    # semi-supervised, with anchors
```

Each writes a PNG next to itself; pass `--show` to open a window instead.

- **`mixture_1d.py`** fits a three-component normal mixture and plots the
  fitted density, AIC/BIC model selection, the log-likelihood trace, and the
  responsibility curves. All three components come back to two decimals.
- **`mixture_2d.py`** does the same in two dimensions with three correlated
  Gaussians, including one with a strong negative correlation. The panels are
  the points with 2σ ellipses, AIC/BIC, the trace, and the fitted density as
  filled contours. Recovers `[-2.5, 1.0]`, `[3.0, 2.5]` and `[0.5, -3.5]` along
  with their covariances.
- **`mixture_regression.py`** hides two crossing lines (`y = 3x` and `y = -3x`,
  σ=0.7) in one X-shaped scatter. A single regression through all of it finds
  slope **−0.028** and σ=**5.31** — the two lines cancel, and the fit is
  nothing. The two-component mixture recovers **+3.003** and **−2.996** with
  σ≈**0.69**, and assigns 97.1% of the points to the right line. The fourth
  panel maps assignment *certainty*, which is where the crossing shows up:
  near the origin no model could tell the lines apart.
- **`mixture_anchored.py`** hides two unit normals 1.5 apart and reveals the
  label of 2% of the points. The panels are the data with the anchors as a
  rug, accuracy on the *unlabeled* points as the anchored fraction grows from
  0% to 20% against the Bayes rate, the semi-supervised log-likelihood trace,
  and the fitted `P(group 1 | x)` against the true posterior. Plain EM stops at
  **74.0%**; from 1% anchored the fit sits at the **77.2%** Bayes rate, with the
  component indices fixed by the labels rather than by luck.

Plotting is not a runtime dependency; matplotlib lives in the `examples`
dependency group, which `uv sync` installs by default.

## Compared to scikit-learn

Both maximize the same objective and find the same optimum (see the benchmark
below); they differ in what they will fit at all.

| | `em` | `sklearn.mixture` |
|---|---|---|
| Gaussian mixtures, 1-D and `d`-D | yes | yes |
| Non-Gaussian families (`lognormal`, `exponential`, `poisson`, `gamma`) | yes | no |
| **Custom log-density** | **yes, pass a callable** | no, only by subclassing the private `BaseMixture` |
| Mixtures of regressions | yes | no |
| Semi-supervised fit (anchors: partial labels) | yes | no |
| Covariance types | full | full, tied, diag, spherical |
| Bayesian / Dirichlet-process mixtures | no | yes |
| Warm start, explicit initial means | no | yes |
| `Pipeline` / `GridSearchCV` | no | yes |
| AIC / BIC | yes | yes |
| Fit time | faster in all six synthetic cases below and on 1-D real data; mixed on small 2-D | baseline |

Use sklearn for a Gaussian mixture, especially inside an existing pipeline or
when you need constrained covariances. Use `em` when the components are not
Gaussian, when you want to write the density yourself, when some observations
come with known labels, or when you are fitting a mixture of regressions.

## Benchmark

There are two benchmark scripts.
[Benchmark on the example datasets](#benchmark-on-the-example-datasets) covers
the real data, where no truth is known and the question is agreement. This one
covers synthetic mixtures, where the true means *are* known and recovery is
measurable.

`benchmarks/vs_sklearn.py` runs `em` head-to-head against
`sklearn.mixture.GaussianMixture`, the standard reference implementation. Both
maximize the same objective, so the achieved log-likelihood is directly
comparable; recovery error and adjusted Rand index say whether that likelihood
found the right answer, and the timing says what it cost. Settings are matched
(`tol`, variance floor, `n_init`, k-means++ seeding, full covariance).

```bash
uv run --group bench python benchmarks/vs_sklearn.py
```

scikit-learn lives in the `bench` group, which `uv sync` does not install by
default. Six cases, all `n = 20,000` or more, `n_init=5`, best-of-2 timing, on
an M-series laptop:

| case | mean loglike | mean abs error | adjusted Rand | fit time |
|---|---|---|---|---|
| 1-D, well separated (n=20k, k=3) | **em** | **em** | **em** | **em** |
| 1-D, overlapping (n=20k, k=4) | **em** | **em** | **em** | **em** |
| 2-D, well separated (n=20k, k=3) | tie | **em** | tie | **em** |
| 2-D, overlapping (n=20k, k=5) | tie | sklearn | **em** | **em** |
| 5-D (n=20k, k=4) | tie | **em** | tie | **em** |
| large 2-D (n=100k, k=4) | tie | sklearn | sklearn | **em** |

Fit time, `em` / sklearn: median **0.74x**, range 0.07–0.89x. This is the
large-`n` regime, where the per-iteration array work dominates and the Python
loop over components does not; on the 272-point 2-D example that ordering
reverses at four or more groups.

The log-likelihood column is the point: the two agree to five or six figures
everywhere, and where they part `em` is ahead. Neither implementation is
leaving fit on the table, so the speed difference is not bought with a worse
answer.

The two cells `em` loses are worth reading carefully, because in both the
log-likelihood is an exact tie. In `2-D, overlapping` the two fits' component
means agree to 0.005 — the same solution, and 0.3914 vs 0.3889 is where they
differ in the last digits. Same for `large 2-D`: means agree to 0.004, and the
adjusted Rand gap of 0.0003 is a few dozen points out of 100,000. Neither is a
quality difference; both are two runs sitting on one optimum.

Read the adjusted Rand column as a property of the data rather than of the
code. At `1-D, overlapping` the four components sit almost on top of each other,
so ~0.06 is close to all the label information the sample contains, and both
implementations get it. A high number means separable components, not a better
algorithm.

The timing gap is mostly seeding. The `5-D` row is the extreme: switching
sklearn's `init_params` to `"random"` takes it from 4.81s to 0.67s at the same
29-31 iterations, so ~85% of that case is its k-means++ implementation rather
than the EM loop.

A note on the recovery metrics. `mean_error` pairs fitted components to
true ones with `linear_sum_assignment`, not greedily. Greedy pairing — give
each fitted component its nearest unclaimed true mean — reports a property of
the matcher rather than of the fit: it inflated `2-D, overlapping` from 0.391 to
0.657 and flipped the winner of `1-D, overlapping` the wrong way.

## API

| | |
|---|---|
| `EM(loglike_fn, *, n_params, param_bounds, init_params, reg, tol, seed)` | construct |
| `.train(X, n_groups, n_iters=1000, n_init=1, *, y=None, labels=None)` | fit; returns `self`. `labels` are anchors, `-1` where unknown |
| `.classify(X=None, *, y=None)` | integer labels |
| `.predict_proba(X=None, *, y=None)` | responsibilities |
| `.loglike(X=None, with_priors=True, *, y=None)` | mixture log-density, or per-component if `with_priors=False` |
| `.score / .aic / .bic / .n_free_params` | model comparison |
| `.sample(n_samples, seed=None)` | draw from the fitted mixture |
| `.summary()` | printable fit report |
| `.e_step(X, weights, params, labels=None) / .m_step` | the two halves, exposed for inspection |

`X=None` reuses the training data. Fitted state lives in `weights_`, `params_`,
`converged_`, `n_iter_`, `loglike_history_` and, when anchors were given,
`labels_`. Used as a context manager, `EM`
releases the cached training data on exit and keeps the fitted parameters.

## License

MIT. See `LICENSE`.
