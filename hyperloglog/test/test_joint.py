#!/usr/bin/env python

from unittest import TestCase

from hyperloglog import HyperLogLog, SlidingHyperLogLog
from hyperloglog.joint import JointEstimationError, estimate_joint


def build(keys, error_rate=0.02):
    hll = HyperLogLog(error_rate)
    hll.add_bulk(keys)
    return hll


def inclusion_exclusion(a, b, error_rate=0.02):
    # naive |A| + |B| - |A u B| on the same sketches, as a baseline
    merged = HyperLogLog(error_rate)
    merged.update(a)
    merged.update(b)
    return a.card() + b.card() - merged.card()


class JointEstimateTestCase(TestCase):
    def test_overlapping_sets(self):
        # reproducible synthetic sets: |A| = 5000, |B| = 5000, |A n B| = 3000
        a = build(['k%d' % i for i in range(5000)])
        b = build(['k%d' % i for i in range(2000, 7000)])

        r = estimate_joint(a, b)

        self.assertTrue(r.converged)
        self.assertEqual(r.status, 'ok')

        # approximation error is within a few standard errors (2% each)
        self.assertAlmostEqual(r.intersection, 3000, delta=300)
        self.assertAlmostEqual(r.a_only, 2000, delta=300)
        self.assertAlmostEqual(r.b_only, 2000, delta=300)

        # composed values are consistent with the components
        self.assertAlmostEqual(r.card_a, r.a_only + r.intersection)
        self.assertAlmostEqual(r.card_b, r.b_only + r.intersection)
        self.assertAlmostEqual(r.union, r.a_only + r.b_only + r.intersection)

    def test_disjoint_sets_beats_inclusion_exclusion(self):
        # same sketches, same baseline: IE goes negative here while the
        # joint estimate stays on the feasible side of zero
        a = build(['a%d' % i for i in range(1000)], 0.05)
        b = build(['b%d' % i for i in range(1000)], 0.05)

        ie = inclusion_exclusion(a, b, 0.05)
        r = estimate_joint(a, b)

        self.assertLess(ie, 0)  # documents the baseline failure mode
        self.assertTrue(r.converged)
        self.assertGreaterEqual(r.intersection, 0)
        self.assertLess(r.intersection, 100)

    def test_basic_set_bounds(self):
        cases = [
            (['x%d' % i for i in range(4000)],
             ['x%d' % i for i in range(1000, 8000)]),
            (['p%d' % i for i in range(2000)],
             ['q%d' % i for i in range(9000)]),
            (['s%d' % i for i in range(6000)],
             ['s%d' % i for i in range(5900, 6100)]),
        ]

        for keys_a, keys_b in cases:
            a = build(keys_a)
            b = build(keys_b)
            r = estimate_joint(a, b)

            self.assertTrue(r.converged)
            # components non-negative
            self.assertGreaterEqual(r.intersection, 0)
            self.assertGreaterEqual(r.a_only, 0)
            self.assertGreaterEqual(r.b_only, 0)
            # intersection bounded by each side, union by the sum
            self.assertLessEqual(r.intersection, r.card_a)
            self.assertLessEqual(r.intersection, r.card_b)
            self.assertGreaterEqual(r.union, r.card_a)
            self.assertGreaterEqual(r.union, r.card_b)

    def test_both_empty(self):
        r = estimate_joint(HyperLogLog(0.02), HyperLogLog(0.02))

        self.assertTrue(r.converged)
        self.assertEqual(r.status, 'exact:empty')
        self.assertEqual(r.as_tuple(), (0.0, 0.0, 0.0))
        self.assertEqual(r.union, 0.0)

    def test_one_empty(self):
        a = build(['k%d' % i for i in range(3000)])
        r = estimate_joint(a, HyperLogLog(0.02))

        self.assertTrue(r.converged)
        self.assertEqual(r.status, 'exact:one_empty')
        self.assertEqual(r.intersection, 0.0)
        self.assertEqual(r.b_only, 0.0)
        self.assertAlmostEqual(r.a_only, a.card())
        self.assertAlmostEqual(r.union, r.card_a)

    def test_identical_sketches(self):
        a = build(['k%d' % i for i in range(4000)])
        b = build(['k%d' % i for i in range(4000)])

        r = estimate_joint(a, b)

        self.assertTrue(r.converged)
        self.assertEqual(r.status, 'exact:identical')
        self.assertEqual(r.a_only, 0.0)
        self.assertEqual(r.b_only, 0.0)
        self.assertAlmostEqual(r.intersection, a.card())
        self.assertAlmostEqual(r.union, r.intersection)

    def test_incompatible_precision(self):
        a = HyperLogLog(0.05)
        b = HyperLogLog(0.01)

        self.assertRaises(ValueError, estimate_joint, a, b)

    def test_incompatible_sketch_type(self):
        a = HyperLogLog(0.05)
        b = SlidingHyperLogLog(0.05, window=10)

        self.assertRaises(ValueError, estimate_joint, a, b)
        self.assertRaises(ValueError, estimate_joint, b, a)

    def test_saturated_registers(self):
        a = build(['k%d' % i for i in range(1000)], 0.05)
        saturated = HyperLogLog(0.05)
        saturated.M[:] = 64 - saturated.p + 1

        self.assertRaises(JointEstimationError, estimate_joint, saturated, a)
        self.assertRaises(JointEstimationError, estimate_joint, a, saturated)

    def test_non_convergence_is_reported_not_silent(self):
        a = build(['k%d' % i for i in range(5000)])
        b = build(['k%d' % i for i in range(2000, 7000)])

        r = estimate_joint(a, b, max_iter=0)

        # failure is flagged, and the best iterate is not a silent zero
        self.assertFalse(r.converged)
        self.assertEqual(r.status, 'max_iter_exceeded')
        self.assertGreater(r.intersection, 0)

        self.assertRaises(JointEstimationError,
                          estimate_joint, a, b, max_iter=0, strict=True)
