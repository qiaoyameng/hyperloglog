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
        # c ingests the k2 stream after the k1 stream, i.e. with
        # timestamps that lag the watermark by up to 9999
        a = SlidingHyperLogLog(0.05, 100, max_lateness=10000)
        b = SlidingHyperLogLog(0.05, 100, max_lateness=10000)
        c = SlidingHyperLogLog(0.05, 100, max_lateness=10000)

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
    Fixed-hash, fixed-timestamp tests for the watermark / max_lateness
    time protocol of the sliding summary.
    """

    def test_future_events_do_not_inflate_past_queries(self):
        s = SlidingHyperLogLog(0.05, 100, max_lateness=100)

        # the future event is written first
        s.add(150, 'future')

        # late events inside the allowed lateness are still accepted
        s.add(60, 'k1')
        s.add(70, 'k2')

        # querying the past: the window [50, 100] must not count 'future'
        self.assertEqual(int(s.card(100, 50)), 2)

        # and must give exactly the estimate of a summary that never
        # saw the future event
        t = SlidingHyperLogLog(0.05, 100, max_lateness=100)
        t.add(60, 'k1')
        t.add(70, 'k2')
        self.assertEqual(s.card(100, 50), t.card(100, 50))
        self.assertEqual(s.card_wlist(100, [50, 40]), t.card_wlist(100, [50, 40]))

        # the future event is still visible from its own window
        self.assertEqual(int(s.card(150, 100)), 3)

    def test_too_late_event_rejected(self):
        s = SlidingHyperLogLog(0.05, 100, max_lateness=10)
        s.add(100, 'k1')

        # late but inside the allowed lateness: accepted
        s.add(95, 'k2')
        self.assertEqual(int(s.card(100)), 2)

        # older than watermark - max_lateness: rejected
        self.assertRaises(ValueError, s.add, 89, 'k3')

        # the rejected event did not corrupt the summary
        self.assertEqual(int(s.card(100)), 2)

    def test_late_event_updates_dominant_points(self):
        s = SlidingHyperLogLog(0.05, 100, max_lateness=20)
        s.add(100, 'k1')
        s.add(85, 'k2')

        ts_all = [t for reg in s.LPFM for t, R in reg]
        self.assertIn(100, ts_all)
        self.assertIn(85, ts_all)
        self.assertEqual(s.watermark, 100)

    def test_unanswerable_history_raises(self):
        s = SlidingHyperLogLog(0.05, 100, max_lateness=10)

        for i in range(10):
            s.add(1000 + i, 'k%d' % i)

        self.assertEqual(s.watermark, 1009)

        # window [900, 1000] reaches before the retention horizon 909
        self.assertRaises(ValueError, s.card, 1000, 100)

        # timestamp 998 is before the queryable horizon 999
        self.assertRaises(ValueError, s.card, 998, 100)

        self.assertRaises(ValueError, s.card_wlist, 1000, [100, 50])

        # the current window is answerable
        self.assertAlmostEqual(s.card(1009, 100), 10, delta=1.5)

        # a query exactly on the queryable horizon is answerable and is
        # not inflated by the events that lie in its future
        self.assertEqual(int(s.card(999, 90)), 0)

    def test_window_lower_boundary(self):
        s = SlidingHyperLogLog(0.05, 100)
        s.add(-1, 'k-1')
        s.add(0, 'k0')
        s.add(100, 'k1')

        # the lower boundary is inclusive: the event at T - W counts,
        # the event just before it does not
        self.assertEqual(int(s.card(100, 100)), 2)

        # window wider than W
        self.assertRaises(ValueError, s.card, 100, 101)

        # timestamp before the queryable horizon (max_lateness=0)
        self.assertRaises(ValueError, s.card, 99, 100)

    def test_merge_different_watermarks(self):
        a = SlidingHyperLogLog(0.05, 100, max_lateness=10)
        b = SlidingHyperLogLog(0.05, 100, max_lateness=10)

        for i in range(5):
            a.add(100 + i, 'a%d' % i)

        for i in range(5):
            b.add(500 + i, 'b%d' % i)

        a.update(b)

        # the merged summary takes the larger watermark
        self.assertEqual(a.watermark, 504)

        # the recent window is answered with b's events only
        d = SlidingHyperLogLog(0.05, 100, max_lateness=10)
        for i in range(5):
            d.add(500 + i, 'b%d' % i)

        self.assertEqual(int(a.card(504, 100)), 5)
        self.assertEqual(a.card(504, 100), d.card(504, 100))

        # a's old events are beyond the retention horizon of the merged
        # summary: querying them raises instead of guessing
        self.assertRaises(ValueError, a.card, 200, 100)

        # merging counters with different lateness policies is rejected
        e = SlidingHyperLogLog(0.05, 100, max_lateness=5)
        self.assertRaises(ValueError, a.update, e)

    def test_empty_estimate_after_all_events_expire(self):
        s = SlidingHyperLogLog(0.05, 100, max_lateness=10)

        for i in range(3):
            s.add(1 + i, 'k%d' % i)

        s.advance_watermark(1000)

        self.assertEqual(s.watermark, 1000)
        self.assertTrue(all(not reg for reg in s.LPFM))
        self.assertEqual(int(s.card(1000)), 0)
        self.assertEqual(int(s.card(1000, 50)), 0)
        self.assertEqual(s.card_wlist(1000, [100, 50]), [0, 0])

        # the advanced watermark also finalizes ingestion
        self.assertRaises(ValueError, s.add, 500, 'late')

        # the watermark never moves backward
        s.advance_watermark(10)
        self.assertEqual(s.watermark, 1000)

    def test_pickle_time_protocol(self):
        s = SlidingHyperLogLog(0.05, 100, max_lateness=10)

        for i in range(10):
            s.add(i, 'k%d' % i)

        b = pickle.loads(pickle.dumps(s))
        self.assertEqual(s, b)
        self.assertEqual(b.watermark, 9)
        self.assertEqual(b.max_lateness, 10)

        c = SlidingHyperLogLog.from_list(s.LPFM, 100, max_lateness=10)
        self.assertEqual(c.watermark, 9)
        self.assertEqual(s, c)
