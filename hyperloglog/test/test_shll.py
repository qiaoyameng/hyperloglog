#!/usr/bin/env python

import os
import math
import time
from unittest import TestCase
from hyperloglog.shll import SlidingHyperLogLog
import pickle


class SlidingHyperLogLogTestCase(TestCase):
    def test_init(self):
        s = SlidingHyperLogLog(0.05, 100)
        self.assertEqual(s.window, 100)
        self.assertEqual(s.p, 9)
        self.assertEqual(s.alpha, 0.7197831133217303)
        self.assertEqual(s.m, 512)
        self.assertEqual(len(s.LPFM), 512)

    def test_add(self):
        s = SlidingHyperLogLog(0.05, 100)

        for i in range(10):
            s.add(i, str(i))

        M = [(i, max(R for ts, R in lpfm)) for i, lpfm in enumerate(s.LPFM) if lpfm]
        self.assertEqual(M, [(40, 1), (121, 1), (197, 4), (200, 4), (247, 2), (260, 3), (377, 2), (444, 1), (500, 3)])

    def test_from_list(self):
        s1 = SlidingHyperLogLog(0.05, 100)

        for i in range(10):
            s1.add(i, str(i))

        s2 = SlidingHyperLogLog.from_list(s1.LPFM, 100)
        self.assertEqual(s1, s2)
        self.assertEqual(s1.card(9), s2.card(9))
        self.assertEqual(s1.card_wlist(9, [100, 3, 5]), [s2.card(9, 100), s2.card(9, 3), s2.card(9, 5)])

    def test_calc_cardinality(self):
        clist = [1, 5, 10, 30, 60, 200, 1000, 10000, 60000]
        n = 30
        rel_err = 0.05

        for card in clist:
            s = 0.0
            for c in range(n):
                a = SlidingHyperLogLog(rel_err, 100)

                for i in range(card):
                    ts = int(time.time())
                    a.add(ts, os.urandom(20))

                s += a.card(int(time.time()))

            z = (float(s) / n - card) / (rel_err * card / math.sqrt(n))
            self.assertLess(-3, z)
            self.assertGreater(3, z)

    def test_calc_cardinality_sliding1(self):
        a = SlidingHyperLogLog(0.05, 100)
        a.add(1, 'k1')
        self.assertEqual(int(a.card(1)), 1)
        self.assertEqual(int(a.card(101)), 1)
        self.assertEqual(int(a.card(102)), 0)
        a.add(2, 'k2')
        a.add(3, 'k3')
        self.assertEqual(int(a.card(3)), 3)
        self.assertEqual(int(a.card(101)), 3)
        self.assertEqual(int(a.card(102)), 2)
        self.assertEqual(int(a.card(103)), 1)
        self.assertEqual(int(a.card(104)), 0)

    def test_calc_cardinality_sliding2(self):
        clist = [1, 5, 10, 30, 60, 200, 1000, 10000, 60000]
        n = 30
        rel_err = 0.05

        for card in clist:
            s = 0.0
            for c in range(n):
                a = SlidingHyperLogLog(rel_err, 100)

                for i in range(card):
                    a.add(i / 2000.0, os.urandom(20))

                s += a.card(card / 2000.0)

            card_stored = min(card, 200000)
            z = (float(s) / n - card_stored) / (rel_err * card_stored / math.sqrt(n))
            self.assertLess(-3, z)
            self.assertGreater(3, z)

    def test_calc_cardinality_sliding3(self):
        clist = [30, 60, 200, 1000, 10000, 60000]
        rel_err = 0.05
        t1 = 0
        t2 = 0
        for card in clist:
            a = SlidingHyperLogLog(rel_err, card)

            for i in range(card):
                a.add(i, os.urandom(20))

            ts = time.time()
            l1 = [a.card(1.5 * card, w / 10.0) for w in range(1, card + 1, card // 10)]
            t1 = (time.time() - ts)
            ts = time.time()
            l2 = a.card_wlist(1.5 * card, [ w / 10.0 for w in range(1, card + 1, card // 10)])
            t2 = (time.time() - ts)
            # print card, t1, t2
            self.assertEqual(l1, l2)

    def test_update(self):
        a = SlidingHyperLogLog(0.05, 100)
        b = SlidingHyperLogLog(0.05, 100)
        c = SlidingHyperLogLog(0.05, 100)

        for i in range(10000):
            a.add(i, str('k1-%d' % i))
            c.add(i, str('k1-%d' % i))

        for i in range(10000):
            b.add(i, str('k2-%d' % i))
            c.add(i, str('k2-%d' % i))

        a.update(b)

        self.assertNotEqual(a, b)
        self.assertNotEqual(b, c)
        self.assertEqual(a, c)

    def test_update_err(self):
        a = SlidingHyperLogLog(0.05, 100)
        b = SlidingHyperLogLog(0.01, 100)

        self.assertRaises(ValueError, a.update, b)

    def test_pickle(self):
        a = SlidingHyperLogLog(0.05, 100)
        for i in range(10000):
            a.add(i, str('k1-%d' % i))

        b = pickle.loads(pickle.dumps(a))
        self.assertEqual(a.window, b.window)
        self.assertEqual(a.alpha, b.alpha)
        self.assertEqual(a.p, b.p)
        self.assertEqual(a, b)


class SlidingHyperLogLogTimeProtocolTestCase(TestCase):
    """
    Time protocol tests: fixed hashes ('k1', 'k2', ... hash to distinct
    registers, see test_calc_cardinality_sliding1) and fixed timestamps.
    """

    @staticmethod
    def points(s):
        return sorted(t for lpfm in s.LPFM for t, R in lpfm)

    def test_future_events_do_not_inflate_past(self):
        past = SlidingHyperLogLog(0.05, 100, allowed_lateness=10)
        mixed = SlidingHyperLogLog(0.05, 100, allowed_lateness=10)
        past.advance_watermark(10)   # horizon 0
        mixed.advance_watermark(10)  # horizon 0

        for i in range(10):
            past.add(i, 'past-%d' % i)
            mixed.add(i, 'past-%d' % i)

        # future events are recorded afterwards
        for i in range(10):
            mixed.add(1000 + i, 'future-%d' % i)

        # a historical query must not be raised by the future events
        self.assertEqual(mixed.card(9, 9), past.card(9, 9))
        self.assertEqual(int(mixed.card(9, 9)), 10)
        self.assertEqual(mixed.card_wlist(9, [9, 5]),
                         [past.card(9, 9), past.card(9, 5)])

        # querying at the future time sees only the future events
        self.assertEqual(int(mixed.card(1009, 9)), 10)

    def test_unanswerable_history_raises(self):
        s = SlidingHyperLogLog(0.05, 100, allowed_lateness=10)
        s.add(1000, 'k1')
        s.advance_watermark(1000)  # horizon 990

        # lower bound 989 is older than the horizon: no fake precise number
        self.assertRaises(ValueError, s.card, 1000, 11)
        self.assertRaises(ValueError, s.card_wlist, 1000, [10, 50])

        # lower bound exactly at the horizon is still answerable
        self.assertEqual(int(s.card(1000, 10)), 1)

    def test_window_lower_bound(self):
        s = SlidingHyperLogLog(0.05, 100, allowed_lateness=5)
        s.add(100, 'k1')
        s.advance_watermark(105)  # horizon 100

        # window lower bound == horizon: allowed, event at the bound counts
        self.assertEqual(int(s.card(105, 5)), 1)
        # window lower bound < horizon: rejected
        self.assertRaises(ValueError, s.card, 105, 6)

    def test_late_events_update_dominant_points(self):
        s = SlidingHyperLogLog(0.05, 100, allowed_lateness=10)
        s.advance_watermark(100)  # horizon 90
        s.add(100, 'k1')
        s.add(95, 'k2')  # late, but inside the allowed lateness band

        # too late: below the horizon
        self.assertRaises(ValueError, s.add, 89, 'k3')

        # the late event updated the dominant points ...
        self.assertEqual(self.points(s), [95, 100])
        # ... and is counted by queries covering the lateness band
        self.assertEqual(int(s.card(105, 15)), 2)  # window [90, 105]

        # once the watermark passes it, the late point is safely cleaned
        s.advance_watermark(106)  # horizon 96
        self.assertEqual(self.points(s), [100])
        self.assertEqual(int(s.card(105, 9)), 1)  # window [96, 105]

    def test_advance_watermark_cleans_and_expires(self):
        s = SlidingHyperLogLog(0.05, 100)
        s.add(1, 'k1')
        s.add(2, 'k2')
        s.add(3, 'k3')

        s.advance_watermark(2)  # horizon 2: ts=1 expires
        self.assertEqual(self.points(s), [2, 3])
        # queries above the horizon stay exact
        self.assertEqual(int(s.card(3, 1)), 2)  # window [2, 3]

        # watermark is monotonic
        self.assertRaises(ValueError, s.advance_watermark, 1)

        # advance past every event: all of them expire
        s.advance_watermark(104)
        self.assertEqual(self.points(s), [])
        # an answerable window over the expired range is empty, not an error
        self.assertEqual(s.card(105, 1), 0.0)
        self.assertEqual(s.card_wlist(110, [1, 5]), [0.0, 0.0])

    def test_update_merges_watermarks(self):
        a = SlidingHyperLogLog(0.05, 100, allowed_lateness=10)
        b = SlidingHyperLogLog(0.05, 100, allowed_lateness=10)
        a.advance_watermark(100)  # horizon 90
        b.advance_watermark(200)  # horizon 190
        a.add(95, 'k1')   # late but within a's lateness
        b.add(195, 'k2')

        a.update(b)

        # the merged summary inherits the highest watermark and cleans
        # points below the merged horizon
        self.assertEqual(a.watermark, 200)
        self.assertEqual(self.points(a), [195])
        self.assertRaises(ValueError, a.card, 200, 11)  # bound 189 < 190
        self.assertEqual(int(a.card(200, 10)), 1)

        # merging a summary without a watermark keeps the current one
        c = SlidingHyperLogLog(0.05, 100, allowed_lateness=10)
        c.add(500, 'k3')
        a.update(c)
        self.assertEqual(a.watermark, 200)
        self.assertEqual(int(a.card(500, 10)), 1)

        # allowed_lateness must match
        d = SlidingHyperLogLog(0.05, 100, allowed_lateness=5)
        self.assertRaises(ValueError, a.update, d)

    def test_pickle_time_protocol(self):
        a = SlidingHyperLogLog(0.05, 100, allowed_lateness=7)
        a.advance_watermark(50)
        a.add(49, 'k1')

        b = pickle.loads(pickle.dumps(a))
        self.assertEqual(b.watermark, 50)
        self.assertEqual(b.allowed_lateness, 7)
        self.assertEqual(a, b)

        c = SlidingHyperLogLog.from_list(a.LPFM, 100, watermark=50,
                                         allowed_lateness=7)
        self.assertEqual(a, c)
