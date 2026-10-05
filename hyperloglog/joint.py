"""
Joint cardinality estimation for a pair of HyperLogLog sketches.

Instead of the naive inclusion-exclusion formula
``card(A) + card(B) - card(A | B)`` (which may go negative), the
estimator here uses the joint distribution of the paired registers
``(A.M[j], B.M[j])`` and fits the per-region Poisson rates of
``A\\B``, ``B\\A`` and ``A intersection B`` by maximum likelihood, as
described in

    Otmar Ertl, "New cardinality estimation algorithms for HyperLogLog
    sketches" (2017), https://arxiv.org/abs/1706.07290

Under the Poissonization approximation each register sees independent
Poisson element counts with rates ``l1, l2, l3`` (elements per register)
for the three regions, and the joint register CDF is

    P(K_A <= k, K_B <= l) =
        exp(-l1 * 2**-k - l2 * 2**-l - l3 * 2**-min(k, l))

The cell probabilities ``p[k, l] = P(K_A = k, K_B = l)`` follow by
inclusion-exclusion over the four rectangle corners, and the
log-likelihood of the observed register-pair histogram is maximized
with a damped Newton iteration projected onto ``l1, l2, l3 >= 0``.
Non-negativity of the rates directly enforces the basic set bounds:
every component is >= 0 and ``intersection <= min(card_a, card_b)``,
``union >= max(card_a, card_b)`` hold for the composed estimates.
"""

import math

import numpy as np

from .hll import HyperLogLog, get_estimate


class JointEstimationError(Exception):
    """
    Raised when a joint estimate cannot be produced (incompatible
    sketches, saturated registers) or, in strict mode, when the
    numerical solver fails to converge.
    """


class JointEstimate(object):
    """
    Result of :func:`estimate_joint`.

    Components (joint MLE estimates, always >= 0):

    - ``intersection``: estimated |A ∩ B|
    - ``a_only``: estimated |A \\ B|
    - ``b_only``: estimated |B \\ A|

    Composed values (consistent with the components by construction):

    - ``card_a`` = ``a_only + intersection``
    - ``card_b`` = ``b_only + intersection``
    - ``union`` = ``a_only + b_only + intersection``

    Solver diagnostics:

    - ``converged``: False means the values are the best iterate found,
      not a trustworthy estimate; it is never silently exact
    - ``iterations``: Newton iterations performed
    - ``status``: machine-readable convergence/exactness status
    """

    __slots__ = ('intersection', 'a_only', 'b_only',
                 'card_a', 'card_b', 'union',
                 'converged', 'iterations', 'status')

    def __init__(self, intersection, a_only, b_only,
                 converged, iterations, status):
        self.intersection = intersection
        self.a_only = a_only
        self.b_only = b_only
        self.card_a = a_only + intersection
        self.card_b = b_only + intersection
        self.union = a_only + b_only + intersection
        self.converged = converged
        self.iterations = iterations
        self.status = status

    def __repr__(self):
        return ('JointEstimate(intersection=%r, a_only=%r, b_only=%r, '
                'converged=%r, iterations=%r, status=%r)' %
                (self.intersection, self.a_only, self.b_only,
                 self.converged, self.iterations, self.status))

    def as_tuple(self):
        return (self.intersection, self.a_only, self.b_only)


def _check_compatible(a, b):
    if not isinstance(a, HyperLogLog) or not isinstance(b, HyperLogLog):
        raise ValueError('joint estimation requires two HyperLogLog '
                         'sketches (same register semantics and hash scheme)')

    if a.p != b.p:
        raise ValueError('Counters precisions should be equal')

    # registers saturate at rho = 64 - p + 1 (w == 0); beyond that the
    # sketch carries no usable information for joint estimation
    max_rho = 64 - a.p + 1
    if a.M.max() >= max_rho or b.M.max() >= max_rho:
        raise JointEstimationError('saturated registers (rho = %d): the '
                                   'sketches cannot support joint estimation'
                                   % max_rho)


def _histogram(a, b):
    pairs, counts = np.unique(np.stack((a.M.astype(np.int64),
                                        b.M.astype(np.int64)), axis=1),
                              axis=0, return_counts=True)
    return pairs[:, 0], pairs[:, 1], counts.astype(float)


def _loglik_grad_hess(lam, ks, ls, cs):
    # log-likelihood, gradient and Hessian of the joint register model
    # at rates lam = (l1, l2, l3); see module docstring for the model
    l1, l2, l3 = lam
    loglik = 0.0
    grad = np.zeros(3)
    hess = np.zeros((3, 3))

    for k, l, c in zip(ks, ls, cs):
        p = 0.0
        dp = np.zeros(3)
        d2p = np.zeros((3, 3))

        for u, v, sign in ((k, l, 1.0), (k - 1, l, -1.0),
                           (k, l - 1, -1.0), (k - 1, l - 1, 1.0)):
            if u < 0 or v < 0:
                continue

            s = np.array((2.0 ** -u, 2.0 ** -v, 2.0 ** -min(u, v)))
            F = math.exp(-(l1 * s[0] + l2 * s[1] + l3 * s[2]))
            p += sign * F
            dp -= sign * F * s
            d2p += sign * F * np.outer(s, s)

        # guard against floating point underflow of a (mathematically
        # positive) cell probability
        p = max(p, 1e-300)

        loglik += c * math.log(p)
        w = c / p
        grad += w * dp
        hess += w * d2p - (w / p) * np.outer(dp, dp)

    return loglik, grad, hess


def _solve_mle(ks, ls, cs, lam, tol, max_iter):
    # damped Newton with projection onto lam >= 0; returns
    # (lam, converged, iterations, status)
    loglik, grad, hess = _loglik_grad_hess(lam, ks, ls, cs)

    for iteration in range(1, max_iter + 1):
        try:
            step = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = None

        if step is None or not np.all(np.isfinite(step)) \
                or np.dot(grad, step) <= 0:
            # Hessian not usable as a curvature matrix: fall back to a
            # gradient ascent direction
            norm = np.linalg.norm(grad)
            if norm == 0:
                return lam, True, iteration, 'ok'
            step = grad / norm

        # active set: do not push components at the boundary below zero
        step[(lam <= 0) & (step < 0)] = 0.0
        if not np.any(step):
            return lam, True, iteration, 'ok'

        # keep lam + t * step >= 0
        t = 1.0
        neg = step < 0
        if np.any(neg):
            t = min(t, 0.99 * np.min(-lam[neg] / step[neg]))

        # backtracking line search (Armijo)
        slope = np.dot(grad, step)
        while t > 1e-12:
            cand = lam + t * step
            cand_loglik = _loglik_grad_hess(cand, ks, ls, cs)[0]
            if cand_loglik >= loglik + 1e-4 * t * slope:
                break
            t *= 0.5

        if t <= 1e-12:
            return lam, False, iteration, 'line_search_failed'

        lam = lam + t * step
        loglik, grad, hess = _loglik_grad_hess(lam, ks, ls, cs)

        if np.max(np.abs(t * step) / (1.0 + np.abs(lam))) < tol:
            return lam, True, iteration, 'ok'

    return lam, False, max_iter, 'max_iter_exceeded'


def estimate_joint(a, b, tol=1e-10, max_iter=100, strict=False):
    """
    Estimates |A ∩ B|, |A \\ B| and |B \\ A| from a pair of HyperLogLog
    sketches via the joint distribution of their paired registers.

    Both sketches must be :class:`HyperLogLog` instances built with the
    same precision (and hence the same hash scheme); a ValueError is
    raised otherwise. Saturated sketches raise JointEstimationError.

    Returns a JointEstimate. If the numerical solver does not converge,
    ``converged`` is False and the components hold the best iterate
    (never a silent zero); with ``strict=True`` a JointEstimationError
    is raised instead.
    """
    _check_compatible(a, b)

    m = a.m

    # degenerate configurations with exact answers
    a_empty = not np.any(a.M)
    b_empty = not np.any(b.M)

    if a_empty and b_empty:
        return JointEstimate(0.0, 0.0, 0.0, True, 0, 'exact:empty')

    if a_empty or b_empty:
        card = b.card() if a_empty else a.card()
        only = (0.0, card) if a_empty else (card, 0.0)
        return JointEstimate(0.0, only[0], only[1], True, 0,
                             'exact:one_empty')

    if np.array_equal(a.M, b.M):
        # identical registers: the likelihood is maximized on the
        # boundary l1 = l2 = 0, i.e. everything lands in the intersection
        return JointEstimate(a.card(), 0.0, 0.0, True, 0,
                             'exact:identical')

    ks, ls, cs = _histogram(a, b)

    # starting point from the plain inclusion-exclusion estimates,
    # clamped into the feasible region
    card_a = a.card()
    card_b = b.card()
    card_u = get_estimate(np.maximum(a.M, b.M), m, a.p, a.alpha)

    inter = max(card_a + card_b - card_u, 0.0)
    lam = np.array([max(card_a - inter, 0.0),
                    max(card_b - inter, 0.0),
                    inter]) / m

    lam, converged, iterations, status = _solve_mle(ks, ls, cs, lam,
                                                    tol, max_iter)

    if not converged and strict:
        raise JointEstimationError('joint MLE did not converge (%s) '
                                   'after %d iterations'
                                   % (status, iterations))

    # rates are kept >= 0 by the solver, so the components satisfy the
    # basic set bounds by construction
    a_only, b_only, intersection = m * lam

    return JointEstimate(float(intersection), float(a_only), float(b_only),
                         converged, iterations, status)
