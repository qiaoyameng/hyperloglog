#!/usr/bin/env python

import random
from unittest import TestCase

from hyperloglog import HyperLogLog, SlidingHyperLogLog
from hyperloglog.joint import JointEstimationError, joint_estimate


def make_sketches(rng, n_a, n_b, n_ab, error_rate):
    shared = [rng.randrange(1 << 60) for _ in range(n_ab)]
    only_a = [rng.randrange(1 << 60) for _ in range(n_a)]
    only_b = [rng.randrange(1 << 60) for _ in range(n_b)]

    a = HyperLogLog(error_rate)
    b = HyperLogLog(error_rate)
    a.add_bulk(shared + only_a)
    b.add_bulk(shared + only_b)
    return a, b


class JointEstimateTestCase(TestCase):
    def test_incompatible_precision(self):
        a = HyperLogLog(0.05)
        b = HyperLogLog(0.01)
        a.add('x')
        b.add('x')
        self.assertRaises(ValueError, joint_estimate, a, b)

    def test_incompatible_type(self):
        a = HyperLogLog(0.05)
        a.add('x')
        s = SlidingHyperLogLog(0.05, 10)
        self.assertRaises(TypeError, joint_estimate, a, s)
        self.assertRaises(TypeError, joint_estimate, a, object())

    def test_both_empty(self):
        est = joint_estimate(HyperLogLog(0.05), HyperLogLog(0.05))
        self.assertEqual((est.a_only, est.b_only, est.both), (0.0, 0.0, 0.0))
        self.assertEqual(est.union, 0.0)
        self.assertTrue(est.converged)
        self.assertEqual(est.method, 'exact-empty')

    def test_one_empty(self):
        rng = random.Random(7)
        a, b = make_sketches(rng, 0, 1000, 0, 0.02)
        est = joint_estimate(a, b)
        self.assertEqual(est.a_only, 0.0)
        self.assertEqual(est.both, 0.0)
        self.assertAlmostEqual(est.b_only, 1000, delta=1000 * 0.1)
        self.assertEqual(est.method, 'exact-one-empty')

    def test_identical(self):
        rng = random.Random(11)
        a, _ = make_sketches(rng, 5000, 0, 0, 0.02)
        b = HyperLogLog(0.02)
        b.update(a)
        self.assertEqual(a, b)

        est = joint_estimate(a, b)
        self.assertEqual(est.a_only, 0.0)
        self.assertEqual(est.b_only, 0.0)
        self.assertAlmostEqual(est.both, 5000, delta=5000 * 0.1)
        self.assertEqual(est.union, est.both)
        self.assertEqual(est.method, 'exact-identical')

    def test_saturated_raises(self):
        a = HyperLogLog(0.05)
        a.M[:] = 64 - a.p + 1  # all registers saturated
        rng = random.Random(13)
        _, b = make_sketches(rng, 0, 1000, 0, 0.05)
        # unbounded likelihood must raise, never silently return zero
        self.assertRaises(JointEstimationError, joint_estimate, a, b)

    def test_saturated_identical_raises(self):
        a = HyperLogLog(0.05)
        a.M[:] = 64 - a.p + 1
        b = HyperLogLog(0.05)
        b.M[:] = 64 - b.p + 1
        self.assertRaises(JointEstimationError, joint_estimate, a, b)

    def test_joint_vs_truth(self):
        rng = random.Random(12345)
        n_a, n_b, n_ab = 15000, 25000, 5000
        a, b = make_sketches(rng, n_a, n_b, n_ab, 0.02)

        est = joint_estimate(a, b)
        self.assertTrue(est.converged)
        self.assertEqual(est.method, 'joint-mle')
        self.assertGreaterEqual(est.iterations, 1)
        self.assertGreaterEqual(est.residual, 0.0)

        # approximation error, not solver failure
        self.assertAlmostEqual(est.a_only, n_a, delta=n_a * 0.1)
        self.assertAlmostEqual(est.b_only, n_b, delta=n_b * 0.1)
        self.assertAlmostEqual(est.both, n_ab, delta=n_ab * 0.1)

        # basic set bounds on components and composed values
        self.assertGreaterEqual(min(est.a_only, est.b_only, est.both), 0.0)
        self.assertLessEqual(est.both, min(est.a_card, est.b_card))
        self.assertGreaterEqual(est.union, max(est.a_card, est.b_card))

    def test_method_entry_point(self):
        rng = random.Random(17)
        a, b = make_sketches(rng, 3000, 4000, 2000, 0.02)
        est = a.joint(b)
        self.assertTrue(est.converged)
        self.assertAlmostEqual(est.both, 2000, delta=2000 * 0.15)

    def test_compare_with_inclusion_exclusion(self):
        # small overlap compared to the estimation error: the naive
        # |A| + |B| - |A u B| estimate goes negative, the joint MLE
        # stays feasible and is on average much closer to the truth
        rng = random.Random(999)
        n_a, n_b, n_ab = 20000, 20000, 500
        ie_negative = 0
        joint_err = 0.0
        ie_err = 0.0
        trials = 24

        for _ in range(trials):
            a, b = make_sketches(rng, n_a, n_b, n_ab, 0.05)
            est = joint_estimate(a, b)
            self.assertTrue(est.converged)

            union = HyperLogLog(0.05)
            union.update(a)
            union.update(b)
            ie = a.card() + b.card() - union.card()

            ie_negative += ie < 0
            ie_err += abs(ie - n_ab)
            joint_err += abs(est.both - n_ab)

            # joint components always satisfy basic set bounds
            self.assertGreaterEqual(min(est.a_only, est.b_only, est.both), 0.0)
            self.assertLessEqual(est.both, min(est.a_card, est.b_card) + 1e-6)
            self.assertGreaterEqual(est.union + 1e-6, max(est.a_card, est.b_card))

        self.assertGreater(ie_negative, 0)
        self.assertLess(joint_err, ie_err)

    def test_does_not_mutate_inputs(self):
        rng = random.Random(19)
        a, b = make_sketches(rng, 3000, 4000, 2000, 0.02)
        ma = a.M.copy()
        mb = b.M.copy()
        joint_estimate(a, b)
        self.assertTrue((a.M == ma).all())
        self.assertTrue((b.M == mb).all())
