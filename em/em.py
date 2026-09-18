"""Expectation-maximization for finite mixture models."""
from __future__ import annotations
from collections.abc import Callable, Sequence
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from tqdm import tqdm
from em.loglike_fns import Family, LOGLIKE_FNS, get_family

__all__ = ["EM"]

_TINY = 1e-300


def _rows_logsumexp(a: np.ndarray) -> np.ndarray:
    """``logsumexp`` along axis 1, specialized for the E-step.

    ``scipy.special.logsumexp`` is generic — complex input, ``b`` weights, an
    optional sign return, a masked search for the largest real part — and none
    of it applies here. Dropping that machinery is worth about 2.5x, and this
    call is the bulk of a fit.

    Shifting by the row max is what keeps the exponentials in range. A row that
    is entirely ``-inf`` (a component some custom ``loglike_fn`` rules out for
    every observation) would make that shift ``-inf - -inf = nan``, so those
    rows are shifted by 0 instead and fall out as ``-inf``, matching scipy.
    """
    largest = a.max(axis=1, keepdims=True)
    shift = np.where(np.isfinite(largest), largest, 0.0)
    with np.errstate(divide="ignore"):
        return np.log(np.exp(a - shift).sum(axis=1)) + shift[:, 0]


class EM:
    """Fit a mixture of ``n_groups`` components by expectation-maximization.

    The component family is either the name of a built-in (see
    :data:`em.loglike_fns.LOGLIKE_FNS`) or a callable ``loglike_fn(x, p)``
    returning the log-density of every observation in ``x`` under one component
    with parameter vector ``p``. Built-ins carry a closed-form weighted MLE, so
    their M-step is exact; a custom callable is maximized numerically with
    L-BFGS-B, which needs ``n_params`` and benefits from ``param_bounds``.

    Observations whose component is already known can be passed to
    :meth:`train` as *anchors* (``labels=``). Their responsibilities are pinned
    to the known component in every E-step, which turns the fit into
    semi-supervised EM: a handful of labeled points is enough to name the
    components, steer the fit away from a wrong local optimum, and separate
    components the unlabeled data alone could not tell apart.

    Args:
        loglike_fn: Family name, or a custom ``(x, p) -> log-density`` callable.
        n_params: Length of ``p``. Required for a custom callable.
        param_bounds: Per-parameter ``(low, high)`` bounds for the numerical
            M-step; ``None`` on either side means unbounded.
        init_params: Starting parameter vector for the numerical M-step.
        reg: Variance/scale floor. Keeps a component from collapsing onto a
            single observation and driving the likelihood to infinity.
        tol: Convergence threshold on the change in mean log-likelihood.
        seed: Seed for the random restarts.

    Example:
        >>> model = EM("normal").train(x, n_groups=2)
        >>> labels = model.classify(x)
    """

    def __init__(
        self,
        loglike_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | str,
        *,
        n_params: int | None = None,
        param_bounds: Sequence[tuple[float | None, float | None]] | None = None,
        init_params: np.ndarray | Sequence[float] | None = None,
        reg: float = 1e-6,
        tol: float = 1e-8,
        seed: int | None = None,
    ) -> None:

        if isinstance(loglike_fn, str):
            if loglike_fn in LOGLIKE_FNS:
                self.family = get_family(loglike_fn)
                self.loglike_fn = self.family.logpdf
                self.custom_loglike = False
            else:
                raise ValueError("Unknown loglikelihood function.")
        else:
            if not callable(loglike_fn):
                raise TypeError("loglike_fn must be a family name or a callable.")
            if n_params is None:
                raise ValueError(
                    "n_params is required for a custom loglike_fn: the number of "
                    "parameters cannot be inferred from the callable."
                )
            self.family = self._custom_family(loglike_fn, n_params, param_bounds, init_params)
            self.loglike_fn = loglike_fn
            self.custom_loglike = True

        self.reg = float(reg)
        self.tol = float(tol)
        self.seed = seed

        self.weights_: np.ndarray | None = None
        self.params_: np.ndarray | None = None
        self.loglike_history_: list[float] = []
        self.converged_ = False
        self.n_iter_ = 0
        self.X_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.n_features_ = 1
        self._resp: np.ndarray | None = None

    # -- construction ---------------------------------------------------------

    @staticmethod
    def _custom_family(
        fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        n_params: int,
        bounds: Sequence[tuple[float | None, float | None]] | None,
        init: np.ndarray | Sequence[float] | None,
    ) -> Family:
        """Wrap a user callable so the EM loop can treat it like a built-in."""
        n_params = int(n_params)
        if n_params < 1:
            raise ValueError("n_params must be at least 1.")
        if bounds is not None and len(bounds) != n_params:
            raise ValueError(f"param_bounds must have {n_params} entries, got {len(bounds)}.")

        start = np.ones(n_params) if init is None else np.asarray(init, dtype=float).ravel()
        if start.size != n_params:
            raise ValueError(f"init_params must have {n_params} entries, got {start.size}.")

        def mle(x: np.ndarray, w: np.ndarray, reg: float, p0: np.ndarray) -> np.ndarray:
            if np.sum(w) <= 0.0:
                return np.asarray(p0, dtype=float)

            def neg_loglike(p: np.ndarray) -> float:
                value = -float(np.dot(w, fn(x, p)))
                return value if np.isfinite(value) else np.inf

            result = minimize(neg_loglike, np.asarray(p0, dtype=float), method="L-BFGS-B", bounds=bounds)
            candidate = np.asarray(result.x, dtype=float)
            if not np.all(np.isfinite(candidate)):
                return np.asarray(p0, dtype=float)
            # L-BFGS-B can report failure while still having improved on p0.
            return candidate if neg_loglike(candidate) <= neg_loglike(p0) else np.asarray(p0, dtype=float)

        def rvs(p: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
            raise NotImplementedError("sample() is only available for built-in families.")

        return Family(
            name="custom",
            param_names=tuple(f"p{i}" for i in range(n_params)),
            logpdf=fn,
            mle=mle,
            rvs=rvs,
            support="real",
            start=start,
        )

    # -- EM steps -------------------------------------------------------------

    def e_step(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        params: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Responsibilities and the log-likelihood.

        Args:
            labels: Optional anchors, shape ``(n_samples,)``: the known
                component of each observation, or ``-1`` where it is unknown.
                An anchored row gets responsibility 1 on its component and 0
                elsewhere, and contributes ``log w_k + log f(x | p_k)`` to the
                total instead of the marginal ``log sum_k``.

        Returns:
            ``(log_resp, total_loglike)``, where ``log_resp`` has shape
            ``(n_samples, n_groups)`` and rows summing to 1 in probability space.
        """
        log_joint = self._log_joint(X, weights, params)
        log_norm = _rows_logsumexp(log_joint)
        log_resp = log_joint - log_norm[:, None]
        if labels is not None:
            rows = np.flatnonzero(labels >= 0)
            if rows.size:
                known = labels[rows]
                # The anchor's own term is its complete-data log-likelihood,
                # so the objective stays a proper likelihood and EM's
                # monotonicity survives the pinning.
                log_norm[rows] = log_joint[rows, known]
                log_resp[rows] = -np.inf
                log_resp[rows, known] = 0.0
        return log_resp, float(np.sum(log_norm))

    def m_step(self, X: np.ndarray, log_resp: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Re-fit the mixing weights and per-component parameters.

        Returns:
            ``(weights, params)`` maximizing the expected complete-data
            log-likelihood under the responsibilities from :meth:`e_step`.
        """
        resp = np.exp(log_resp)
        mass = resp.sum(axis=0)
        weights = mass / X.shape[0]
        new_params = np.array(
            [self.family.mle(X, resp[:, k], self.reg, params[k]) for k in range(params.shape[0])],
            dtype=float,
        )
        return weights, new_params

    def train(
        self,
        X: np.ndarray | None,
        n_groups: int,
        n_iters: int = 1_000,
        n_init: int = 1,
        *,
        y: np.ndarray | None = None,
        labels: np.ndarray | Sequence | None = None,
    ) -> "EM":
        """Fit the mixture to ``X``.

        The first restart starts from a deterministic quantile split; any
        further restarts use k-means++ seeding, and the fit with the highest
        log-likelihood wins. EM only finds a local optimum, so ``n_init > 1`` is
        worth it whenever the components overlap.

        Args:
            X: Observations, shape ``(n_samples,)`` or ``(n_samples, 1)``.
            n_groups: Number of mixture components.
            n_iters: Maximum EM iterations per restart.
            n_init: Number of restarts.
            y: Targets, shape ``(n_samples,)``. Required for a conditional
                family such as ``"linear-regression"``, which models
                ``p(y | x)``; rejected for the density families.
            labels: Anchors, shape ``(n_samples,)``: the component index of
                every observation whose component is known, and ``-1`` (or
                ``NaN``/``None``) where it is not. Anchored observations keep
                their component through the whole fit, and component ``k`` is
                by definition the one the anchors labeled ``k``, so the fitted
                indices are no longer arbitrary. Every initialization is
                aligned to the anchors before the first M-step.

        Returns:
            ``self``, so calls can be chained.
        """
        x = self._check_data(X, y, allow_none=False)
        if self.family.conditional:
            self.n_features_ = x.shape[1] - 1
        elif self.family.multivariate:
            self.n_features_ = x.shape[1]
        else:
            self.n_features_ = 1
        if self.custom_loglike:
            self._validate_custom_loglike(x)
        n_groups = int(n_groups)
        if n_groups < 1:
            raise ValueError("n_groups must be at least 1.")
        if n_groups > x.shape[0]:
            raise ValueError(f"n_groups={n_groups} exceeds the {x.shape[0]} observations.")
        if n_iters < 1:
            raise ValueError("n_iters must be at least 1.")
        if n_init < 1:
            raise ValueError("n_init must be at least 1.")
        anchors = self._check_labels(labels, x.shape[0], n_groups)

        rng = np.random.default_rng(self.seed)
        best: dict | None = None
        failures: list[str] = []
        # One bar for the whole fit: n_init is a real target, where n_iters is
        # only a cap that a converging run never reaches.
        with tqdm(total=n_init, desc="EM restarts") as bar:
            for attempt in range(n_init):
                try:
                    fit = self._fit_once(
                        x, n_groups, n_iters, rng, deterministic=(attempt == 0), bar=bar,
                        labels=anchors,
                    )
                except (FloatingPointError, ValueError) as exc:
                    failures.append(str(exc))
                else:
                    if best is None or fit["loglike"] > best["loglike"]:
                        best = fit
                bar.update(1)

        if best is None:
            raise RuntimeError(
                "Every EM restart failed. Last error: " + (failures[-1] if failures else "unknown")
            )

        self.X_ = x
        self.labels_ = anchors
        self.weights_ = best["weights"]
        self.params_ = best["params"]
        self.loglike_history_ = best["history"]
        self.converged_ = best["converged"]
        self.n_iter_ = best["n_iter"]
        self._resp = np.exp(best["log_resp"])
        return self

    def _fit_once(
        self,
        x: np.ndarray,
        n_groups: int,
        n_iters: int,
        rng: np.random.Generator,
        deterministic: bool,
        bar: "tqdm | None" = None,
        labels: np.ndarray | None = None,
    ) -> dict:
        """Run EM to convergence from one initialization.

        ``bar`` is the restart-level progress bar owned by :meth:`train`; this
        method reports into it but does not create or close one.
        """
        n_params = self.family.n_params_for(self.n_features_)
        if self.family.conditional:
            # No ordering of x separates regression lines, so a quantile split
            # is meaningless here: seed each component from its own random
            # subsample instead, which gives genuinely different starting fits.
            weights = np.full(n_groups, 1.0 / n_groups)
            params = self._init_lines(x, n_groups, n_params, rng, labels)
            return self._em_loop(x, weights, params, n_iters, bar, labels)

        # Anchored rows are exact zeros off their component, and log(0) is the
        # -inf the E-step would produce for them anyway.
        with np.errstate(divide="ignore"):
            log_resp = np.log(self._init_resp(x, n_groups, rng, deterministic, labels))
        if self.custom_loglike:
            # Fit the whole sample once, so every component's numerical M-step
            # starts from a sane point rather than from init_params directly.
            pooled = self.family.mle(x, np.ones_like(x), self.reg, self.family.start)
            params = np.tile(pooled, (n_groups, 1))
        else:
            params = np.zeros((n_groups, n_params))
        weights, params = self.m_step(x, log_resp, params)
        return self._em_loop(x, weights, params, n_iters, bar, labels)

    def _em_loop(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        params: np.ndarray,
        n_iters: int,
        bar: "tqdm | None",
        labels: np.ndarray | None = None,
    ) -> dict:
        """Alternate E and M from the given start until convergence."""
        history: list[float] = []
        previous = -np.inf
        converged = False
        n_iter = 0
        for n_iter in range(1, n_iters + 1):
            log_resp, loglike = self.e_step(x, weights, params, labels)
            if not np.isfinite(loglike):
                raise ValueError("The log-likelihood became non-finite; check the data and the family.")
            history.append(loglike)
            if bar is not None and n_iter % 25 == 0:
                bar.set_postfix(iter=n_iter, loglike=f"{loglike / x.shape[0]:.5f}")
            if abs(loglike - previous) / x.shape[0] <= self.tol:
                converged = True
                break
            previous = loglike
            weights, params = self.m_step(x, log_resp, params)
            if not np.all(np.isfinite(params)):
                raise ValueError("The M-step produced non-finite parameters.")

        if bar is not None:
            bar.set_postfix(iter=n_iter, loglike=f"{history[-1] / x.shape[0]:.5f}")

        return {
            "weights": weights,
            "params": params,
            "log_resp": log_resp,
            "loglike": history[-1],
            "history": history,
            "converged": converged,
            "n_iter": n_iter,
        }

    def _init_resp(
        self,
        x: np.ndarray,
        n_groups: int,
        rng: np.random.Generator,
        deterministic: bool,
        labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build starting responsibilities: a quantile split, else k-means++.

        With anchors, the split's cluster indices are first permuted to agree
        with the anchors as much as possible, and the anchored rows are then
        pinned. Without the permutation the anchors and the split would name
        the same cluster differently, and the first M-step would blend two
        clusters into one component.
        """
        rows = x[:, None] if x.ndim == 1 else x
        n_rows = rows.shape[0]
        if deterministic:
            clusters = self._ordered_split(rows, n_groups)
        else:
            centers = self._kmeanspp(rows, n_groups, rng)
            distance = np.sum((rows[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            clusters = np.argmin(distance, axis=1)
        if labels is not None:
            clusters = self._align_to_anchors(clusters, labels, n_groups)

        resp = np.zeros((n_rows, n_groups))
        resp[np.arange(n_rows), clusters] = 1.0

        # Soften, so that no component starts with exactly zero mass.
        smoothing = 1e-2
        resp = (1.0 - smoothing) * resp + smoothing / n_groups
        if labels is not None:
            anchored = np.flatnonzero(labels >= 0)
            resp[anchored] = 0.0
            resp[anchored, labels[anchored]] = 1.0
        return resp

    @staticmethod
    def _align_to_anchors(clusters: np.ndarray, labels: np.ndarray, n_groups: int) -> np.ndarray:
        """Relabel ``clusters`` to maximize agreement with the anchors.

        A cluster index is arbitrary; an anchor's is not. The assignment that
        matches the most anchored rows is a linear assignment problem on the
        cluster-by-label contingency table.
        """
        anchored = labels >= 0
        table = np.zeros((n_groups, n_groups))
        np.add.at(table, (clusters[anchored], labels[anchored]), 1.0)
        cluster_index, component = linear_sum_assignment(table, maximize=True)
        mapping = np.empty(n_groups, dtype=int)
        mapping[cluster_index] = component
        return mapping[clusters]

    def _init_lines(
        self,
        data: np.ndarray,
        n_groups: int,
        n_params: int,
        rng: np.random.Generator,
        labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Seed a mixture of regressions by fitting random subsamples.

        A random *partition* of the data would give every component the same
        line, since each part is a uniform subsample of the whole. Small random
        subsets differ enough to break that symmetry.

        A component with anchors is seeded from them instead: they are the
        best available guess at its line. If there are too few to determine
        one, random rows fill the subsample up to its usual size.
        """
        n_rows = data.shape[0]
        size = int(min(n_rows, max(n_params + 1, n_rows // (2 * n_groups))))
        params = []
        for k in range(n_groups):
            w = np.zeros(n_rows)
            own = np.flatnonzero(labels == k) if labels is not None else np.empty(0, dtype=int)
            w[own] = 1.0
            missing = size - own.size
            if missing > 0:
                pool = np.setdiff1d(np.arange(n_rows), own, assume_unique=True)
                w[rng.choice(pool, size=min(missing, pool.size), replace=False)] = 1.0
            params.append(self.family.mle(data, w, self.reg, np.zeros(n_params)))
        return np.asarray(params, dtype=float)

    @staticmethod
    def _ordered_split(rows: np.ndarray, n_groups: int) -> np.ndarray:
        """Equal-size split along the direction the data varies most.

        In one dimension this is just sorting by x. In more, the first
        principal direction is the closest analogue: the axis along which an
        equal split separates the data best.
        """
        if rows.shape[1] == 1:
            score = rows[:, 0]
        else:
            centered = rows - rows.mean(axis=0)
            _, _, components = np.linalg.svd(centered, full_matrices=False)
            score = centered @ components[0]
        labels = np.empty(rows.shape[0], dtype=int)
        for k, index in enumerate(np.array_split(np.argsort(score), n_groups)):
            labels[index] = k
        return labels

    @staticmethod
    def _kmeanspp(rows: np.ndarray, n_groups: int, rng: np.random.Generator) -> np.ndarray:
        """k-means++ seeding over rows, in any number of dimensions."""
        n_rows = rows.shape[0]
        centers = [rows[rng.integers(n_rows)]]
        closest = np.sum((rows - centers[0]) ** 2, axis=1)
        for _ in range(1, n_groups):
            total = float(closest.sum())
            if total <= 0.0:
                centers.append(rows[rng.integers(n_rows)])
            else:
                centers.append(rows[rng.choice(n_rows, p=closest / total)])
            closest = np.minimum(closest, np.sum((rows - centers[-1]) ** 2, axis=1))
        return np.asarray(centers)

    # -- inference ------------------------------------------------------------

    def classify(self, X: np.ndarray | None = None, *, y: np.ndarray | None = None) -> np.ndarray:
        """Assign each observation to its most probable component.

        Args:
            X: Observations, or ``None`` to reuse the training data.
            y: Targets, for a conditional family.

        Returns:
            Integer labels of shape ``(n_samples,)``. For the full posterior
            use :meth:`predict_proba`.
        """
        return np.argmax(self.predict_proba(X, y=y), axis=1)

    def predict_proba(
        self, X: np.ndarray | None = None, *, y: np.ndarray | None = None
    ) -> np.ndarray:
        """Posterior probability of each component for each observation.

        With ``X=None`` this is the responsibility matrix of the final
        E-step, in which anchored observations are pinned to their component.
        Pass the training data explicitly for the unconstrained posterior the
        fitted mixture assigns to them.
        """
        self._check_fitted()
        x = self._check_data(X, y, allow_none=True)
        if X is None and self._resp is not None:
            return self._resp
        log_resp, _ = self.e_step(x, self.weights_, self.params_)
        return np.exp(log_resp)

    def loglike(
        self, X: np.ndarray | None = None, with_priors: bool = True, *, y: np.ndarray | None = None
    ) -> np.ndarray:
        """Per-observation log-likelihood.

        Args:
            X: Observations, or ``None`` to reuse the training data.
            with_priors: If true, return the mixture log-density
                ``log sum_k w_k f(x | p_k)`` with shape ``(n_samples,)``. If
                false, return the per-component log-densities *without* the
                mixing weights, shape ``(n_samples, n_groups)``.

        Anchors do not enter here: this is the marginal density of the fitted
        mixture, whatever labels were used to fit it. The semi-supervised
        objective the fit maximized is in ``loglike_history_``.
        """
        self._check_fitted()
        x = self._check_data(X, y, allow_none=True)
        if not with_priors:
            return self._log_components(x, self.params_)
        return _rows_logsumexp(self._log_joint(x, self.weights_, self.params_))

    def score(self, X: np.ndarray | None = None, *, y: np.ndarray | None = None) -> float:
        """Mean log-likelihood per observation."""
        return float(np.mean(self.loglike(X, y=y)))

    def n_free_params(self) -> int:
        """Number of free parameters: the mixing weights plus the components."""
        self._check_fitted()
        n_groups = self.weights_.size
        return (n_groups - 1) + n_groups * self.family.n_params_for(self.n_features_)

    def aic(self, X: np.ndarray | None = None, *, y: np.ndarray | None = None) -> float:
        """Akaike information criterion; lower is better."""
        return 2.0 * self.n_free_params() - 2.0 * float(np.sum(self.loglike(X, y=y)))

    def bic(self, X: np.ndarray | None = None, *, y: np.ndarray | None = None) -> float:
        """Bayesian information criterion; lower is better."""
        x = self._check_data(X, y, allow_none=True)
        n = x.shape[0]
        return self.n_free_params() * np.log(n) - 2.0 * float(np.sum(self.loglike(X, y=y)))

    def sample(self, n_samples: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Draw from the fitted mixture.

        Returns:
            ``(x, labels)`` — the samples and the component each came from.
        """
        self._check_fitted()
        rng = np.random.default_rng(seed)
        labels = rng.choice(self.weights_.size, size=int(n_samples), p=self.weights_)
        shape = (int(n_samples), self.n_features_) if self.family.multivariate else (int(n_samples),)
        x = np.empty(shape)
        for k in range(self.weights_.size):
            mask = labels == k
            if np.any(mask):
                x[mask] = self.family.rvs(self.params_[k], int(mask.sum()), rng)
        return x, labels

    def summary(self) -> str:
        """One line per component: weight and fitted parameters."""
        self._check_fitted()
        status = "converged" if self.converged_ else f"stopped at {self.n_iter_} iterations"
        if self.labels_ is not None and np.any(self.labels_ >= 0):
            status += f", {int(np.sum(self.labels_ >= 0))} anchors"
        lines = [f"EM({self.family.name}) with {self.weights_.size} groups, {status}"]
        order = np.argsort(-self.weights_)
        for k in order:
            lines.append(f"  group {k}: weight={self.weights_[k]:.4f}, {self.family.describe(self.params_[k], self.n_features_)}")
        lines.append(f"  mean loglike = {self.loglike_history_[-1] / self.X_.shape[0]:.6f}")
        return "\n".join(lines)

    # -- internals ------------------------------------------------------------

    def _validate_custom_loglike(self, x: np.ndarray) -> None:
        """Probe a user callable before fitting, so shape errors surface here."""
        probe = x[: min(x.shape[0], 8)]
        self._log_components(probe, self.family.start[None, :])

    def _log_components(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Component log-densities, shape ``(n_samples, n_groups)``."""
        columns = []
        for p in params:
            column = np.asarray(self.family.logpdf(x, p), dtype=float)
            if column.shape != (x.shape[0],):
                raise ValueError(
                    f"loglike_fn returned shape {column.shape}, expected {(x.shape[0],)}: it "
                    "must evaluate the log-density elementwise over x."
                )
            columns.append(column)
        return np.column_stack(columns)

    def _log_joint(self, x: np.ndarray, weights: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Component log-densities plus the log mixing weights."""
        return self._log_components(x, params) + np.log(np.maximum(weights, _TINY))[None, :]

    def _check_data(
        self, X: np.ndarray | None, y: np.ndarray | None, allow_none: bool
    ) -> np.ndarray:
        """Validate the inputs and pack them into the array the family expects.

        A density family gets a 1-D array of observations. A conditional family
        gets ``(n_samples, n_features + 1)`` with the target in the last column
        — the packing is an implementation detail, never the public contract.
        """
        if X is None:
            if not allow_none:
                raise TypeError("train() requires X; None is only accepted after fitting.")
            if y is not None:
                raise ValueError("y was given without X.")
            if self.X_ is None:
                raise RuntimeError("No training data stored; pass X explicitly.")
            return self.X_

        x = np.asarray(X, dtype=float)
        if x.size == 0:
            raise ValueError("X is empty.")
        if not np.all(np.isfinite(x)):
            raise ValueError("X contains NaN or infinite values.")

        if self.family.conditional:
            if y is None:
                raise ValueError(
                    f"The '{self.family.name}' family models p(y | x); pass y=."
                )
            if x.ndim == 1:
                x = x[:, None]
            if x.ndim != 2:
                raise ValueError(f"X must be 1-D or 2-D; got shape {np.shape(X)}.")
            target = np.asarray(y, dtype=float).ravel()
            if target.shape[0] != x.shape[0]:
                raise ValueError(
                    f"X has {x.shape[0]} rows but y has {target.shape[0]} entries."
                )
            if not np.all(np.isfinite(target)):
                raise ValueError("y contains NaN or infinite values.")
            if self.params_ is not None and x.shape[1] != self.n_features_:
                raise ValueError(
                    f"This model was fitted on {self.n_features_} features; got {x.shape[1]}."
                )
            data = np.column_stack([x, target])
            self.family.validate(data)
            return data

        if y is not None:
            raise ValueError(
                f"The '{self.family.name}' family is a density over x; y is not used."
            )
        if self.family.multivariate:
            if x.ndim == 1:
                x = x[:, None]
            if x.ndim != 2:
                raise ValueError(f"X must be 1-D or 2-D; got shape {np.shape(X)}.")
            if self.params_ is not None and x.shape[1] != self.n_features_:
                raise ValueError(
                    f"This model was fitted on {self.n_features_} features; got {x.shape[1]}."
                )
            self.family.validate(x)
            return x
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.ravel()
        if x.ndim != 1:
            raise ValueError(
                f"X must be 1-D or (n_samples, 1); got shape {np.shape(X)}. EM here is univariate."
            )
        self.family.validate(x)
        return x

    @staticmethod
    def _check_labels(
        labels: np.ndarray | Sequence | None, n_samples: int, n_groups: int
    ) -> np.ndarray | None:
        """Validate anchors into an int array with ``-1`` for the unlabeled.

        Accepts ``-1``, ``NaN`` or ``None`` as "unknown", so a labels column
        from a dataframe drops in whichever convention it uses.
        """
        if labels is None:
            return None
        raw = np.asarray(labels)
        if raw.dtype == object:
            raw = np.array([-1 if v is None else v for v in raw.ravel()], dtype=float).reshape(raw.shape)
        if raw.dtype.kind == "f":
            raw = np.where(np.isnan(raw), -1.0, raw)
            if not np.all(np.isfinite(raw)) or not np.all(raw == np.round(raw)):
                raise ValueError("labels must be integer component indices, -1, NaN or None.")
        elif raw.dtype.kind not in "iub":
            raise ValueError("labels must be integer component indices, -1, NaN or None.")
        out = raw.astype(int).ravel()
        if out.shape[0] != n_samples:
            raise ValueError(f"labels has {out.shape[0]} entries but X has {n_samples} rows.")
        if np.any(out < -1) or np.any(out >= n_groups):
            raise ValueError(f"labels must lie in [0, {n_groups}) or be -1 for unlabeled observations.")
        if not np.any(out >= 0):
            return None
        return out

    def _check_fitted(self) -> None:
        if self.params_ is None:
            raise RuntimeError("This EM instance is not fitted yet; call train(X, n_groups) first.")

    def __repr__(self) -> str:
        if self.params_ is None:
            return f"EM(family={self.family.name!r}, unfitted)"
        return f"EM(family={self.family.name!r}, n_groups={self.weights_.size}, converged={self.converged_})"

    def __enter__(self) -> "EM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Drop the cached per-observation arrays; the fitted parameters stay.
        self.X_ = None
        self.labels_ = None
        self._resp = None
        return False
