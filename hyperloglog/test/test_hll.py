#!/usr/bin/env python

import math
import os
import pickle
from unittest import TestCase

import numpy as np

from hyperloglog.hll import HyperLogLog, get_alpha, get_rho
from hyperloglog.const import biasData, tresholdData, rawEstimateData


class HyperLogLogTestCase(TestCase):
    def test_blobs(self):
        self.assertEqual(len(tresholdData), 18 - 3)

    def test_alpha(self):
        alpha = [get_alpha(b) for b in range(4, 10)]
        self.assertEqual(alpha, [0.673, 0.697, 0.709, 0.7152704932638152, 0.7182725932495458, 0.7197831133217303])

    def test_alpha_bad(self):
        self.assertRaises(ValueError, get_alpha, 1)
        self.assertRaises(ValueError, get_alpha, 17)

    def test_rho(self):
        self.assertEqual(get_rho(0, 32), 33)
        self.assertEqual(get_rho(1, 32), 32)
        self.assertEqual(get_rho(2, 32), 31)
        self.assertEqual(get_rho(3, 32), 31)
        self.assertEqual(get_rho(4, 32), 30)
        self.assertEqual(get_rho(5, 32), 30)
        self.assertEqual(get_rho(6, 32), 30)
        self.assertEqual(get_rho(7, 32), 30)
        self.assertEqual(get_rho(1 << 31, 32), 1)
        self.assertRaises(ValueError, get_rho, 1 << 32, 32)

    def test_init(self):
        s = HyperLogLog(0.05)
        self.assertEqual(s.p, 9)
        self.assertEqual(s.alpha, 0.7197831133217303)
        self.assertEqual(s.m, 512)
        self.assertEqual(len(s.M), 512)

    def test_add(self):
        s = HyperLogLog(0.05)

        for i in range(10):
            s.add(str(i))

        M = [(i, v) for i, v in enumerate(s.M) if v > 0]

        self.assertEqual(M, [(40, 1), (121, 1), (197, 4), (200, 4), (247, 2), (260, 3), (377, 2), (444, 1), (500, 3)])

    def test_add_bulk(self):
        s = HyperLogLog(0.05)

        s.add_bulk(str(i) for i in range(10))

        M = [(i, v) for i, v in enumerate(s.M) if v > 0]

        self.assertEqual(M, [(40, 1), (121, 1), (197, 4), (200, 4), (247, 2), (260, 3), (377, 2), (444, 1), (500, 3)])

    def test_calc_cardinality(self):
        clist = [1, 5, 10, 30, 60, 200, 1000, 10000, 60000]
        n = 30
        rel_err = 0.05

        for card in clist:
            s = 0.0
            for c in range(n):
                a = HyperLogLog(rel_err)

                for i in range(card):
                    # add bytes
                    a.add(os.urandom(20))

                s += a.card()

            z = (float(s) / n - card) / (rel_err * card / math.sqrt(n))
            self.assertLess(-3, z)
            self.assertGreater(3, z)

    def test_calc_cardinality_bulk(self):
        clist = [1, 5, 10, 30, 60, 200, 1000, 10000, 60000]
        n = 30
        rel_err = 0.05

        for card in clist:
            s = 0.0
            for c in range(n):
                a = HyperLogLog(rel_err)

                # add bytes
                a.add_bulk((os.urandom(20) for i in range(card)))

                s += a.card()

            z = (float(s) / n - card) / (rel_err * card / math.sqrt(n))
            self.assertLess(-3, z)
            self.assertGreater(3, z)


    def test_update(self):
        a = HyperLogLog(0.05)
        b = HyperLogLog(0.05)
        c = HyperLogLog(0.05)

        for i in range(2):
            a.add(str(i))
            c.add(str(i))

        for i in range(2, 4):
            b.add(str(i))
            c.add(str(i))

        a.update(b)

        self.assertNotEqual(a, b)
        self.assertNotEqual(b, c)
        self.assertEqual(a, c)


    def test_update_err(self):
        a = HyperLogLog(0.05)
        b = HyperLogLog(0.01)

        self.assertRaises(ValueError, a.update, b)

    def test_downsample_recomputes_registers(self):
        hashes = {}

        for removed in range(4):
            hashes['max-%d' % removed] = (removed << 4) | removed
            hashes['rank-58-%d' % removed] = (1 << 6) | (removed << 4) | (4 + removed)
            hashes['rank-1-%d' % removed] = (1 << 63) | (removed << 4) | (8 + removed)

        expected = [61, 60, 59, 59] + [58] * 4 + [1] * 4 + [0] * 4

        high = HyperLogLog(0.13, hash_func=hashes.get)
        low = HyperLogLog(0.26, hash_func=hashes.get)

        for value in hashes:
            high.add(value)
            low.add(value)

        self.assertEqual(high.p, 6)
        self.assertEqual(low.p, 4)

        original_high = high.M.copy()
        downsampled = high.downsample(4)

        np.testing.assert_array_equal(high.M, original_high)
        np.testing.assert_array_equal(downsampled.M, expected)
        np.testing.assert_array_equal(downsampled.M, low.M)
        self.assertIsNot(downsampled.M, high.M)

    def test_downsample_common_precision_update(self):
        hashes = {}

        for removed in range(4):
            hashes['max-%d' % removed] = (removed << 4) | removed
            hashes['rank-58-%d' % removed] = (1 << 6) | (removed << 4) | (4 + removed)
            hashes['rank-1-%d' % removed] = (1 << 63) | (removed << 4) | (8 + removed)

        values = list(hashes)
        expected = [61, 60, 59, 59] + [58] * 4 + [1] * 4 + [0] * 4

        high = HyperLogLog(0.13, hash_func=hashes.get)
        medium = HyperLogLog(0.184, hash_func=hashes.get)
        low = HyperLogLog(0.26, hash_func=hashes.get)

        high.add_bulk(values[:6])
        medium.add_bulk(values[6:])
        low.add_bulk(values)

        self.assertEqual(medium.p, 5)

        original_high = high.M.copy()
        original_medium = medium.M.copy()
        merged = high.downsample(4)
        merged.update(medium.downsample(4))

        np.testing.assert_array_equal(high.M, original_high)
        np.testing.assert_array_equal(medium.M, original_medium)
        np.testing.assert_array_equal(merged.M, low.M)
        np.testing.assert_array_equal(merged.M, expected)

    def test_downsample_does_not_increase_precision(self):
        hll = HyperLogLog(0.13)

        self.assertRaises(ValueError, hll.downsample, 7)
        self.assertRaises(ValueError, hll.downsample, 3)
        self.assertRaises(ValueError, hll.downsample, 4.0)

    def test_pickle(self):
        a = HyperLogLog(0.05)
        for x in range(100):
            a.add(str(x))
        b = pickle.loads(pickle.dumps(a))
        self.assertEqual(a, b)
        self.assertEqual(a.alpha, b.alpha)
        self.assertEqual(a.p, b.p)
