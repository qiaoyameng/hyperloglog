#!/usr/bin/env python

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
    # error_rate 0.05 -> p = 9, m = 512 dense registers

    def make_sparse(self, threshold=16, count=5, offset=0):
        s = HyperLogLog(0.05, sparse=True, sparse_threshold=threshold)
        for i in range(offset, offset + count):
            s.add(str(i))
        return s

    def make_dense(self, count, offset=0):
        d = HyperLogLog(0.05)
        for i in range(offset, offset + count):
            d.add(str(i))
        return d

    def test_threshold_validation(self):
        self.assertRaises(ValueError, HyperLogLog, 0.05, sparse=True, sparse_threshold=0)
        self.assertRaises(ValueError, HyperLogLog, 0.05, sparse=True, sparse_threshold=-3)
        self.assertRaises(ValueError, HyperLogLog, 0.05, sparse=True, sparse_threshold=513)
        self.assertRaises(ValueError, HyperLogLog, 0.05, sparse=True, sparse_threshold=2.5)
        self.assertRaises(ValueError, HyperLogLog, 0.05, sparse_threshold=8)

        s = HyperLogLog(0.05, sparse=True)
        self.assertEqual(s.mode, 'sparse')
        self.assertEqual(s.sparse_threshold, 512 // 4)
        self.assertEqual(s.p, 9)
        self.assertEqual(s.m, 512)

    def test_exact_count_and_promotion(self):
        s = HyperLogLog(0.05, sparse=True, sparse_threshold=8)

        for i in range(8):
            s.add(str(i))
            self.assertEqual(s.mode, 'sparse')
            self.assertEqual(len(s), i + 1)
            self.assertEqual(len(s.M), i + 1)

        # duplicates never advance the threshold
        for i in range(8):
            s.add(str(i))
        self.assertEqual(s.mode, 'sparse')
        self.assertEqual(len(s.M), 8)
        self.assertEqual(len(s), 8)

        # the 9th distinct value promotes to the dense registers
        s.add(str(8))
        self.assertEqual(s.mode, 'dense')
        self.assertEqual(len(s.M), 512)

        d = self.make_dense(9)
        self.assertTrue(np.array_equal(s.M, d.M))
        self.assertEqual(s.card(), d.card())

        # the counter stays dense afterwards
        s.add(str(100))
        d.add(str(100))
        self.assertEqual(s.mode, 'dense')
        self.assertEqual(s, d)

    def test_promotion_happens_once(self):
        s = self.make_sparse(threshold=4, count=5)
        self.assertEqual(s.mode, 'dense')
        self.assertRaises(RuntimeError, s._promote)

    def test_add_bulk(self):
        s = HyperLogLog(0.05, sparse=True, sparse_threshold=8)
        s.add_bulk(str(i) for i in range(6))
        self.assertEqual(s.mode, 'sparse')
        self.assertEqual(len(s), 6)

        s.add_bulk(str(i) for i in range(6, 12))
        self.assertEqual(s.mode, 'dense')
        self.assertEqual(s, self.make_dense(12))

    def test_sparse_sparse_merge(self):
        a = self.make_sparse(count=5)
        b = self.make_sparse(count=6, offset=3)

        a.update(b)

        self.assertEqual(a.mode, 'sparse')
        self.assertEqual(len(a), 9)
        self.assertEqual(a.M, {HyperLogLog._hash(str(i)) for i in range(9)})

        # the merged counter is unchanged
        self.assertEqual(b.mode, 'sparse')
        self.assertEqual(len(b.M), 6)

    def test_sparse_merge_promotes_over_threshold(self):
        a = self.make_sparse(threshold=8, count=6)
        b = self.make_sparse(threshold=8, count=6, offset=6)

        a.update(b)

        self.assertEqual(a.mode, 'dense')
        self.assertEqual(a, self.make_dense(12))

    def test_mixed_merge_both_orders(self):
        # sparse.update(dense)
        a = self.make_sparse(count=5)
        b = self.make_dense(5, offset=5)
        b_before = b.M.copy()
        a.update(b)
        self.assertEqual(a.mode, 'dense')

        # dense.update(sparse)
        c = self.make_dense(5, offset=5)
        d = self.make_sparse(count=5)
        d_before = set(d.M)
        c.update(d)
        self.assertEqual(c.mode, 'dense')

        # both orders agree with each other and with a plain dense union
        self.assertEqual(a, c)
        self.assertEqual(a, self.make_dense(10))

        # the merged counters are unchanged
        self.assertTrue(np.array_equal(b.M, b_before))
        self.assertEqual(b, self.make_dense(5, offset=5))
        self.assertEqual(d.mode, 'sparse')
        self.assertEqual(d.M, d_before)

    def test_update_precision_mismatch(self):
        a = HyperLogLog(0.05, sparse=True)
        b = HyperLogLog(0.01)
        self.assertRaises(ValueError, a.update, b)

    def test_pickle_records_mode(self):
        a = self.make_sparse(threshold=8, count=5)

        state = a.__getstate__()
        self.assertEqual(state['mode'], 'sparse')
        self.assertEqual(state['sparse_threshold'], 8)

        b = pickle.loads(pickle.dumps(a))
        self.assertEqual(a, b)
        self.assertEqual(b.mode, 'sparse')
        self.assertEqual(b.sparse_threshold, 8)
        self.assertEqual(len(b), 5)

        # dense state survives the roundtrip after promotion
        a.add_bulk(str(i) for i in range(5, 20))
        self.assertEqual(a.mode, 'dense')
        c = pickle.loads(pickle.dumps(a))
        self.assertEqual(a, c)
        self.assertEqual(c.mode, 'dense')

    def test_legacy_pickle_rejected(self):
        # pickles written before the representation mode was recorded
        # must not be silently reinterpreted under the new layout
        legacy = {'alpha': get_alpha(9), 'p': 9, 'm': 512,
                  'M': np.zeros(512, np.int8)}
        s = HyperLogLog.__new__(HyperLogLog)
        self.assertRaises(ValueError, s.__setstate__, legacy)

        bad_mode = dict(legacy, mode='list', sparse_threshold=None)
        self.assertRaises(ValueError, s.__setstate__, bad_mode)

    def test_sparse_memory_footprint(self):
        # error_rate 0.01 -> p = 14, m = 16384 dense registers
        dense = HyperLogLog(0.01)
        self.assertEqual(dense.M.size, 16384)

        s = HyperLogLog(0.01, sparse=True)
        for i in range(100):
            s.add(str(i))

        # sparse keeps 100 distinct hashes instead of 16384 registers
        self.assertEqual(s.mode, 'sparse')
        self.assertEqual(len(s.M), 100)
        self.assertLess(len(s.M) * 8, dense.M.nbytes)
