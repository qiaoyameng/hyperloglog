"""
Joint cardinality estimation for pairs of HyperLogLog sketches.

The naive inclusion-exclusion estimate |A| + |B| - |A u B| subtracts
three noisy cardinality estimates and frequently goes negative when the
true overlap is small compared to the estimation error.  This module
instead uses the joint distribution of the *paired registers* of two
compatible sketches to estimate the three Venn components |A\\B|,
|B\\A| and |A n B| directly, via the joint maximum-likelihood estimator
of Otmar Ertl, "New cardinality estimation algorithms for HyperLogLog
sketches" (arXiv:1702.01284), section 6.

Model: the elements of A\\B, B\\A and A n B hashed into one register
form three independent Poisson streams with rates n_a/m, n_b/m and
n_ab/m.  A register holds the maximum "rho" of its stream, so for a
stream with rate lam, P(M <= k) = exp(-lam * 2**-k).  The register
pairs (M_a[i], M_b[i]) are i.i.d. across registers, giving a concave
log-likelihood in (n_a, n_b, n_ab) which is maximized by coordinate
ascent with safeguarded bisection on the analytic score.

The solver reports its convergence state; if it does not converge (or
the maximum is unbounded, e.g. saturated registers) a
JointEstimationError is raised instead of silently returning zero.
"""

import numpy as np

from .hll import HyperLogLog, get_estimate

__all__ = ['JointEstimate', 'JointEstimationError', 'joint_estimate']

# cardinality estimates are meaningless beyond the 64-bit hash space
_MAX_LOG2_CARD = 62
_LAM_MIN = 1e-13
_GRAD_TOL_REL = 1e-7
_BISECT_ITER = 60


class JointEstimationError(ArithmeticError):
    """
    Raised when the joint estimate cannot be computed: the numerical
    solver did not converge, or the likelihood is unbounded (e.g.
    saturated registers).  Never silently returns zero instead.
    """


class JointEstimate(object):
    """
    Result of joint_estimate(): the three Venn components of two sets,
    plus diagnostics of the numerical solver.

    a_only: estimate of |A \\ B|
    b_only: estimate of |B \\ A|
    both:   estimate of |A n B| (the intersection)
    """

    __slots__ = ('a_only', 'b_only', 'both', 'converged', 'iterations',
                 'residual', 'method')

    def __init__(self, a_only, b_only, both, converged, iterations,
                 residual, method):
        self.a_only = a_only
        self.b_only = b_only
        self.both = both
        self.converged = converged
        self.iterations = iterations
        self.residual = residual
        self.method = method

    @property
    def intersection(self):
        """Estimate of |A n B| (same as `both`)."""
        return self.both

    @property
    def union(self):
        """Estimate of |A u B| as the sum of the three components."""
        return self.a_only + self.b_only + self.both

    @property
    def a_card(self):
        """Estimate of |A| implied by the joint fit."""
        return self.a_only + self.both

    @property
    def b_card(self):
        """Estimate of |B| implied by the joint fit."""
        return self.b_only + self.both

    def __repr__(self):
        return ('JointEstimate(a_only=%r, b_only=%r, both=%r, '
                'converged=%r, iterations=%r, residual=%r, method=%r)'
                % (self.a_only, self.b_only, self.both, self.converged,
                   self.iterations, self.residual, self.method))


def _distributions(lam, t):
    # F[k] = P(M <= k) = exp(-lam * 2**-k), with t[q+1] = 0 so F[q+1] = 1
    F = np.exp(-lam * t)
    dF = -t * F
    K = len(t)
    f = np.empty(K)
    df = np.empty(K)
    f[0] = F[0]
    df[0] = dF[0]
    # f[k] = F[k] - F[k-1] = F[k] * (1 - exp(-lam * t[k])) for 1 <= k <= q;
    # expm1 keeps tiny probabilities accurate instead of underflowing to 0
    small = -np.expm1(-lam * t)
    f[1:] = F[1:] * small[1:]
    f[-1] = small[-2]  # f[q+1] = 1 - F[q]
    df[1:] = dF[1:] - dF[:-1]
    return F, f, dF, df


def _loglike_grad(la, lb, lab, t, diag, gt, lt):
    """
    Joint log-likelihood and its gradient w.r.t. (la, lb, lab).

    diag: (k, count) cells with M_a == M_b == k
    gt:   (k, l, count) cells with M_a == k > l == M_b
    lt:   (k, l, count) cells with M_a == k < l == M_b
    """
    Fa, fa, dFa, dfa = _distributions(la, t)
    Fb, fb, dFb, dfb = _distributions(lb, t)
    Fab, fab, dFab, dfab = _distributions(lab, t)
    Fa_prev = np.concatenate(([0.0], Fa[:-1]))
    Fb_prev = np.concatenate(([0.0], Fb[:-1]))
    Fab_prev = np.concatenate(([0.0], Fab[:-1]))
    dFab_prev = np.concatenate(([0.0], dFab[:-1]))

    ll = 0.0
    ga = gb = gab = 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        if len(gt[2]):
            k, l, c = gt
            # P = fa[k] * (fb[l]*Fab[l] + Fb_prev[l]*fab[l])
            S = fb[l] * Fab[l] + Fb_prev[l] * fab[l]
            ll += np.dot(c, np.log(fa[k]) + np.log(S))
            ga += np.dot(c, dfa[k] / fa[k])
            gb += np.dot(c, (dfb[l] * Fab[l] + np.concatenate(([0.0], dFb[:-1]))[l] * fab[l]) / S)
            gab += np.dot(c, (fb[l] * dFab[l] + Fb_prev[l] * dfab[l]) / S)

        if len(lt[2]):
            k, l, c = lt
            # P = fb[l] * (fa[k]*Fab[k] + Fa_prev[k]*fab[k])
            S = fa[k] * Fab[k] + Fa_prev[k] * fab[k]
            ll += np.dot(c, np.log(fb[l]) + np.log(S))
            gb += np.dot(c, dfb[l] / fb[l])
            ga += np.dot(c, (dfa[k] * Fab[k] + np.concatenate(([0.0], dFa[:-1]))[k] * fab[k]) / S)
            gab += np.dot(c, (fa[k] * dFab[k] + Fa_prev[k] * dfab[k]) / S)

        if len(diag[1]):
            k, c = diag
            # P = fab[k]*Fa[k]*Fb[k] + Fab_prev[k]*fa[k]*fb[k]
            D = fab[k] * Fa[k] * Fb[k] + Fab_prev[k] * fa[k] * fb[k]
            ll += np.dot(c, np.log(D))
            ga += np.dot(c, (fab[k] * dFa[k] * Fb[k] + Fab_prev[k] * dfa[k] * fb[k]) / D)
            gb += np.dot(c, (fab[k] * Fa[k] * dFb[k] + Fab_prev[k] * fa[k] * dfb[k]) / D)
            gab += np.dot(c, (dfab[k] * Fa[k] * Fb[k] + dFab_prev[k] * fa[k] * fb[k]) / D)

    return ll, (ga, gb, gab)


def _solve_coordinate(which, lam, ctx, lam_cap):
    """
    Maximize the (concave) log-likelihood over one coordinate by
    bisection on its analytic score, which is monotonically decreasing.
    """
    def score(x):
        lam[which] = x
        return _loglike_grad(lam[0], lam[1], lam[2], *ctx)[1][which]

    if score(_LAM_MIN) <= 0.0:
        lam[which] = 0.0
        return

    lo, hi = _LAM_MIN, 1.0
    while score(hi) > 0.0:
        hi *= 2.0
        if hi > lam_cap:
            raise JointEstimationError(
                'joint likelihood is unbounded (saturated registers?)')

    for _ in range(_BISECT_ITER):
        mid = 0.5 * (lo + hi)
        if score(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * max(hi, 1.0):
            break

    lam[which] = 0.5 * (lo + hi)


def _solve_joint(ctx, m, init, max_iter):
    """
    Maximize the joint log-likelihood: Newton steps on the analytic
    score (Hessian by finite differences) with backtracking, falling
    back to coordinate-ascent sweeps whenever the Newton step is not
    an improvement or leaves the feasible region.
    Returns (lam, iterations, residual, converged).
    """
    lam = np.array(init, dtype=float)
    lam_cap = 2.0 ** _MAX_LOG2_CARD / m
    grad_tol = _GRAD_TOL_REL * m

    def loglike_grad(x):
        return _loglike_grad(x[0], x[1], x[2], *ctx)

    # a few coordinate sweeps to reach the neighborhood of the optimum
    work = list(lam)
    for _ in range(3):
        for which in range(3):
            _solve_coordinate(which, work, ctx, lam_cap)
    lam = np.array(work)

    residual = np.inf
    for it in range(1, max_iter + 1):
        ll, grad = loglike_grad(lam)
        grad = np.array(grad)
        # KKT residual: at a boundary point (lam_i == 0) a negative
        # score is optimal, only a positive one violates the conditions
        residual = max(abs(g) if x > 0.0 else max(g, 0.0)
                       for x, g in zip(lam, grad))
        if residual <= grad_tol:
            return lam, it + 3, residual, True

        # Hessian of the log-likelihood by central differences of the score
        hess = np.empty((3, 3))
        for i in range(3):
            h = max(lam[i], 1.0 / m) * 1e-6
            step_i = np.zeros(3)
            step_i[i] = h
            if lam[i] - h >= 0.0:
                _, gp = loglike_grad(lam + step_i)
                _, gm = loglike_grad(lam - step_i)
                hess[:, i] = (np.array(gp) - np.array(gm)) / (2.0 * h)
            else:
                _, g0 = loglike_grad(lam)
                _, gp = loglike_grad(lam + step_i)
                hess[:, i] = (np.array(gp) - np.array(g0)) / h
        hess = 0.5 * (hess + hess.T)

        newton_ok = False
        try:
            delta = -np.linalg.solve(hess, grad)
            if np.all(np.isfinite(delta)):
                # backtracking line search on the Newton direction
                scale = 1.0
                for _ in range(25):
                    cand = lam + scale * delta
                    if np.all(cand >= 0.0):
                        ll_cand, _ = loglike_grad(cand)
                        if np.isfinite(ll_cand) and ll_cand > ll:
                            lam = cand
                            newton_ok = True
                            break
                    scale *= 0.5
        except np.linalg.LinAlgError:
            pass

        if not newton_ok:
            work = list(lam)
            for which in range(3):
                _solve_coordinate(which, work, ctx, lam_cap)
            lam = np.array(work)

    return lam, max_iter + 3, residual, False


def _mle_1d(M, m, t):
    """
    1D maximum-likelihood cardinality of a single sketch's registers
    (Ertl's MLE).  Used for the degenerate cases of the joint estimate.
    """
    counts = np.bincount(np.asarray(M).ravel(), minlength=len(t)).astype(float)
    if counts[0] == m:
        return 0.0

    def score(lam):
        _, f, _, df = _distributions(lam, t)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(f > 0, df / f, 0.0)
        return float(np.dot(counts, r))

    if score(_LAM_MIN) <= 0.0:
        return 0.0

    lam_cap = 2.0 ** _MAX_LOG2_CARD / m
    lo, hi = _LAM_MIN, 1.0
    while score(hi) > 0.0:
        hi *= 2.0
        if hi > lam_cap:
            raise JointEstimationError(
                'likelihood is unbounded (saturated registers?)')

    for _ in range(_BISECT_ITER):
        mid = 0.5 * (lo + hi)
        if score(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * max(hi, 1.0):
            break

    return m * 0.5 * (lo + hi)


def _check_compatible(a, b):
    if not isinstance(a, HyperLogLog) or not isinstance(b, HyperLogLog):
        raise TypeError(
            'joint estimation requires two HyperLogLog sketches built '
            'with the same hash scheme')
    if a.p != b.p:
        raise ValueError(
            'sketches have incompatible precisions (p=%d vs p=%d); '
            'joint register distributions require equal p'
            % (a.p, b.p))


def _check_bounds(est):
    # basic set bounds: components non-negative, intersection not larger
    # than either set, union not smaller than either set
    tol = 1e-6 * max(1.0, est.union)
    if min(est.a_only, est.b_only, est.both) < -tol:
        raise JointEstimationError('negative Venn component: %r' % (est,))
    if est.both - min(est.a_card, est.b_card) > tol:
        raise JointEstimationError('intersection exceeds a set: %r' % (est,))
    if max(est.a_card, est.b_card) - est.union > tol:
        raise JointEstimationError('union smaller than a set: %r' % (est,))


def joint_estimate(a, b, max_iter=100):
    """
    Jointly estimate the Venn components of the sets behind two
    HyperLogLog sketches from their paired register distributions.

    Returns a JointEstimate with a_only ~ |A\\B|, b_only ~ |B\\A| and
    both ~ |A n B|, plus solver diagnostics (converged, iterations,
    residual).  Both sketches must use the same precision and hash
    scheme.  Empty, identical and saturated sketches are handled
    separately.  Raises JointEstimationError if the numerical solver
    fails - it never silently returns zero.
    """
    _check_compatible(a, b)

    m, p = a.m, a.p
    q = 64 - p
    t = 2.0 ** -np.arange(q + 2)
    t[q + 1] = 0.0

    ma, mb = a.M.astype(np.int64), b.M.astype(np.int64)
    a_empty = not ma.any()
    b_empty = not mb.any()

    if a_empty and b_empty:
        return JointEstimate(0.0, 0.0, 0.0, True, 0, 0.0, 'exact-empty')

    if np.array_equal(ma, mb):
        n = _mle_1d(ma, m, t)
        return JointEstimate(0.0, 0.0, n, True, 1, 0.0, 'exact-identical')

    if a_empty:
        n = _mle_1d(mb, m, t)
        return JointEstimate(0.0, n, 0.0, True, 1, 0.0, 'exact-one-empty')

    if b_empty:
        n = _mle_1d(ma, m, t)
        return JointEstimate(n, 0.0, 0.0, True, 1, 0.0, 'exact-one-empty')

    # joint histogram of paired registers
    K = q + 2
    joint = np.bincount(ma * K + mb, minlength=K * K).reshape(K, K)
    ks, ls = np.nonzero(joint)
    counts = joint[ks, ls].astype(float)

    is_diag = ks == ls
    is_gt = ks > ls
    diag = (ks[is_diag], counts[is_diag])
    gt = (ks[is_gt], ls[is_gt], counts[is_gt])
    lt = (ks[~is_diag & ~is_gt], ls[~is_diag & ~is_gt],
          counts[~is_diag & ~is_gt])
    ctx = (t, diag, gt, lt)

    # start from the inclusion-exclusion components (floored positive)
    ca, cb = a.card(), b.card()
    merged = np.maximum(ma, mb)
    cu = get_estimate(merged, m, p, a.alpha)
    ab0 = max(ca + cb - cu, 0.0)
    init = (max(ca - ab0, 0.25) / m,
            max(cb - ab0, 0.25) / m,
            max(ab0, 0.25) / m)

    lam, iterations, residual, converged = _solve_joint(ctx, m, init,
                                                        max_iter)

    if not converged or not np.isfinite(residual):
        raise JointEstimationError(
            'joint MLE did not converge after %d iterations '
            '(residual=%r); estimates are not reported' % (max_iter, residual))

    ll, _ = _loglike_grad(lam[0], lam[1], lam[2], *ctx)
    if not np.isfinite(ll):
        raise JointEstimationError('joint log-likelihood is not finite')

    est = JointEstimate(max(m * lam[0], 0.0), max(m * lam[1], 0.0),
                        max(m * lam[2], 0.0), True, iterations, residual,
                        'joint-mle')
    _check_bounds(est)
    return est
