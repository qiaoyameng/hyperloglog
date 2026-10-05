#!/usr/bin/env python

import copy
from unittest import TestCase
from hyperloglog.hll import HyperLogLog, get_alpha, get_rho
from hyperloglog.const import biasData, tresholdData, rawEstimateData
import math
import numpy as np
import os
import pickle


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

    def test_pickle(self):
        a = HyperLogLog(0.05)
        for x in range(100):
            a.add(str(x))
        b = pickle.loads(pickle.dumps(a))
        self.assertEqual(a, b)
        self.assertEqual(a.alpha, b.alpha)
        self.assertEqual(a.p, b.p)


class SparseHyperLogLogTestCase(TestCase):
    def test_threshold_validation(self):
        # m = 512 for error_rate 0.05
        for bad in (0, -1, 1.5, '10', True, 513):
            self.assertRaises(ValueError, HyperLogLog, 0.05, bad)

        HyperLogLog(0.05)            # dense by default
        HyperLogLog(0.05, None)      # explicit dense
        HyperLogLog(0.05, 1)
        HyperLogLog(0.05, 512)

    def test_sparse_exact_count(self):
        s = HyperLogLog(0.05, sparse_threshold=8)
        self.assertTrue(s.is_sparse)

        # add fixed values one by one up to the threshold
        for i in range(8):
            s.add(str(i))
            self.assertTrue(s.is_sparse)
            self.assertEqual(len(s), i + 1)      # exact, not an estimate
            self.assertEqual(s.card(), float(i + 1))

        # duplicates are already in the set: they must not advance
        # the threshold nor trigger promotion
        for i in range(8):
            s.add(str(i))
        self.assertTrue(s.is_sparse)
        self.assertEqual(len(s), 8)

    def test_promotion(self):
        s = HyperLogLog(0.05, sparse_threshold=8)
        d = HyperLogLog(0.05)

        for i in range(8):
            s.add(str(i))
            d.add(str(i))

        self.assertTrue(s.is_sparse)
        self.assertIsNone(s.M)

        # the 9th distinct value crosses the threshold: promote once
        s.add(str(8))
        d.add(str(8))

        self.assertFalse(s.is_sparse)
        self.assertIsNone(s._sparse)
        # registers after promotion match a counter that was always dense
        self.assertTrue(np.array_equal(s.M, d.M))

        # promotion happens only once: further adds stay dense
        for i in range(9, 50):
            s.add(str(i))
            d.add(str(i))

        self.assertFalse(s.is_sparse)
        self.assertTrue(np.array_equal(s.M, d.M))
        self.assertEqual(s.card(), d.card())

    def test_add_bulk_sparse(self):
        s = HyperLogLog(0.05, sparse_threshold=8)

        s.add_bulk(str(i) for i in range(5))
        self.assertTrue(s.is_sparse)
        self.assertEqual(len(s), 5)

        # bulk add crossing the threshold promotes
        s.add_bulk(str(i) for i in range(5, 10))
        self.assertFalse(s.is_sparse)

        d = HyperLogLog(0.05)
        d.add_bulk(str(i) for i in range(10))
        self.assertEqual(s, d)

    def test_merge_sparse_sparse(self):
        a = HyperLogLog(0.05, sparse_threshold=8)
        b = HyperLogLog(0.05, sparse_threshold=8)

        for i in range(4):
            a.add(str(i))
        for i in range(4, 7):
            b.add(str(i))

        a.update(b)
        self.assertTrue(a.is_sparse)
        self.assertEqual(len(a), 7)

        # the merged input is unchanged
        self.assertTrue(b.is_sparse)
        self.assertEqual(len(b), 3)

        # union crossing the threshold promotes
        c = HyperLogLog(0.05, sparse_threshold=8)
        for i in range(7, 10):
            c.add(str(i))

        a.update(c)
        self.assertFalse(a.is_sparse)

        d = HyperLogLog(0.05)
        for i in range(10):
            d.add(str(i))
        self.assertEqual(a, d)

    def test_merge_sparse_dense_both_orders(self):
        sparse_vals = [str(i) for i in range(5)]
        dense_vals = [str(i) for i in range(3, 20)]

        s = HyperLogLog(0.05, sparse_threshold=8)
        for v in sparse_vals:
            s.add(v)

        d = HyperLogLog(0.05)
        for v in dense_vals:
            d.add(v)

        expect = HyperLogLog(0.05)
        for v in sparse_vals + dense_vals:
            expect.add(v)

        # sparse.update(dense): self promotes, then merges registers
        a = copy.deepcopy(s)
        a.update(d)
        self.assertFalse(a.is_sparse)
        self.assertEqual(a, expect)

        # dense.update(sparse): sparse hashes fold into the registers
        b = copy.deepcopy(d)
        b.update(s)
        self.assertEqual(b, expect)

        # both merge inputs are unchanged
        self.assertTrue(s.is_sparse)
        self.assertEqual(len(s), 5)
        d2 = HyperLogLog(0.05)
        for v in dense_vals:
            d2.add(v)
        self.assertEqual(d, d2)

    def test_merge_err(self):
        a = HyperLogLog(0.05, sparse_threshold=8)
        b = HyperLogLog(0.01, sparse_threshold=8)
        self.assertRaises(ValueError, a.update, b)

    def test_pickle_sparse_roundtrip(self):
        s = HyperLogLog(0.05, sparse_threshold=8)
        for i in range(5):
            s.add(str(i))

        s2 = pickle.loads(pickle.dumps(s))
        self.assertEqual(s, s2)
        self.assertTrue(s2.is_sparse)
        self.assertEqual(s2.sparse_threshold, 8)
        self.assertEqual(len(s2), 5)

    def test_pickle_promoted_roundtrip(self):
        s = HyperLogLog(0.05, sparse_threshold=8)
        for i in range(20):
            s.add(str(i))

        s2 = pickle.loads(pickle.dumps(s))
        self.assertEqual(s, s2)
        self.assertFalse(s2.is_sparse)
        self.assertEqual(s2.sparse_threshold, 8)

    def test_pickle_records_mode(self):
        s = HyperLogLog(0.05, sparse_threshold=8)
        self.assertEqual(s.__getstate__()['mode'], 'sparse')

        for i in range(20):
            s.add(str(i))
        self.assertEqual(s.__getstate__()['mode'], 'dense')

        d = HyperLogLog(0.05)
        self.assertEqual(d.__getstate__()['mode'], 'dense')

    def test_old_pickle_rejected(self):
        # state dict in the pre-sparse format carries no mode marker and
        # must not be silently interpreted as the new structure
        s = HyperLogLog(0.05)
        old_state = {'alpha': s.alpha, 'p': s.p, 'm': s.m, 'M': s.M}

        t = HyperLogLog.__new__(HyperLogLog)
        self.assertRaises(ValueError, t.__setstate__, old_state)

        bad_mode = dict(s.__getstate__(), mode='list')
        self.assertRaises(ValueError, HyperLogLog.__new__(HyperLogLog).__setstate__, bad_mode)

    def test_sparse_memory_footprint(self):
        # a sparse counter with k distinct items stores k hashes and no
        # register array, while a dense one always allocates m registers
        s = HyperLogLog(0.05, sparse_threshold=100)
        for i in range(10):
            s.add(str(i))

        self.assertEqual(len(s._sparse), 10)   # 10 stored hashes
        self.assertIsNone(s.M)                 # no register array at all

        d = HyperLogLog(0.05)
        self.assertEqual(d.M.size, 512)        # 512 registers allocated
        self.assertLess(len(s._sparse), d.M.size)
