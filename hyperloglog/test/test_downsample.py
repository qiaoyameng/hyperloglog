#!/usr/bin/env python

import numpy as np
from unittest import TestCase

from hyperloglog.hll import HyperLogLog, HLL_COUNTER_TYPE


def hll_of_p(p):
    # build a counter directly from its precision p (number of index bits)
    return HyperLogLog._from_p(p)


def register_values(hll):
    return [int(v) for v in hll.M]


# A high precision counter fed only these raw 64-bit hashes (p = 8):
#
#   x = j1 * 256 + w1, where j1 are the source-bucket bits and w1 the
#   remaining 56 hash bits. The values are picked so that every case of the
#   p = 8 -> p = 6 downsampling is exercised:
#
#     * buckets 0/64 share coarse bucket 0 and straddle the w1 == 0 case;
#     * buckets 1/65 share coarse bucket 1: bucket 1 has w1 == 1 with the
#       lost index bits all zero (x = 0x101), bucket 65 has w1 == 1 with a
#       non-zero lost index bit (x = 0x141), and x = 0x40 is the pure
#       maximum-rank case for j1 = 64;
#     * bucket 4 keeps a plain rank (no maximum rank involved).
CRAFTED_HASHES_P8 = [
    0x0000000000000000,  # j1 = 0,  w1 = 0 -> R = 57 (maximum rank)
    0x0000000000000101,  # j1 = 1,  w1 = 1 -> R = 56
    0x0000000000000404,  # j1 = 4,  w1 = 4 -> R = 54
    0x0000000000000040,  # j1 = 64, w1 = 0 -> R = 57 (maximum rank)
    0x0000000000000141,  # j1 = 65, w1 = 1 -> R = 56
]


class DownsampleTestCase(TestCase):
    def _build_p8(self):
        hll = hll_of_p(8)
        for x in CRAFTED_HASHES_P8:
            hll.add_hash(x)
        return hll

    def test_crafted_source_registers(self):
        hll = self._build_p8()
        expected = {0: 57, 1: 56, 4: 54, 64: 57, 65: 56}
        actual = {j: int(v) for j, v in enumerate(hll.M) if v}
        self.assertEqual(actual, expected)

    def test_downsample_register_exact(self):
        # Exact register-level comparison against a counter built directly
        # at the low precision; cardinality estimates are never used here
        high = self._build_p8()
        high_before = register_values(high)

        low = high.downsample(6)

        self.assertEqual(low.p, 6)
        self.assertEqual(low.m, 64)

        reference = hll_of_p(6)
        for x in CRAFTED_HASHES_P8:
            reference.add_hash(x)

        self.assertEqual(register_values(low), register_values(reference))

        # explicit expectations derived from the bit layout:
        #   j'=0 <- strided buckets 0 (max-rank, rho 59) and
        #           64 (max-rank, rho 58)                        -> 59
        #   j'=1 <- strided buckets 1 (R=56) and 65 (R=56)      -> 56
        #   j'=4 <- bucket 4 (R=54)                              -> 54
        expected = {0: 59, 1: 56, 4: 54}
        actual = {j: int(v) for j, v in enumerate(low.M) if v}
        self.assertEqual(actual, expected)

        # the source summary must remain untouched
        self.assertEqual(register_values(high), high_before)
        self.assertEqual(high.p, 8)
        self.assertIsNot(low.M, high.M)

    def test_consecutive_buckets_max_is_wrong(self):
        # grouping consecutive source buckets (0,1,2,3) instead of strided
        # ones (0,64,...) must NOT reproduce the correct low precision
        # layout: this guards against the naive max-of-neighbors shortcut
        high = self._build_p8()
        correct = register_values(high.downsample(6))

        naive_src = high.M.astype(np.int64).reshape(64, 4)
        naive = [int(v) for v in np.max(naive_src, axis=1)]
        self.assertNotEqual(naive, correct)

    def test_shifted_out_bits_zero_and_nonzero(self):
        # x = 0x101 has w1 = 1 and the 2 lost index bits (bits 6..7) all
        # zero; x = 0x4141 has w1 = 1 with bit 6 = 1, i.e. a non-zero bit
        # moves out of the bucket into w. Both must match a summary built
        # directly at p = 6, register by register
        high = hll_of_p(8)
        high.add_hash(0x0000000000000101)  # j1 = 1, bits 6..7 are zero
        high.add_hash(0x0000000000000141)  # j1 = 65, bit 6 is 1

        low = high.downsample(6)

        reference = hll_of_p(6)
        reference.add_hash(0x0000000000000101)
        reference.add_hash(0x0000000000000141)

        self.assertEqual(register_values(low), register_values(reference))
        self.assertEqual(int(low.M[1]), 56)

    def test_maximum_rank_recomputed_from_bucket(self):
        # w == 0 at p = 8 stores the maximum rank 57. The hash then equals
        # the source bucket index, so at p = 6 rho is recomputed exactly
        # from the stratum row c rather than clamped blindly
        high = hll_of_p(8)
        high.add_hash(0)

        low = high.downsample(6)
        self.assertEqual(int(low.M[0]), 59)  # j1 = 0 -> c = 0, w = 0

        high64 = hll_of_p(8)
        high64.add_hash(0x40)  # w == 0 at p = 8, j1 = 64 -> c = 1
        low64 = high64.downsample(6)
        self.assertEqual(int(low64.M[0]), 58)  # w = 1 at p = 6 -> rho 58

        # and the information loss prevents raising precision back
        with self.assertRaises(ValueError):
            low.downsample(8)

    def test_downsample_same_precision_is_independent_copy(self):
        high = self._build_p8()
        copy_ = high.downsample(8)

        self.assertEqual(copy_, high)
        self.assertIsNot(copy_, high)
        self.assertFalse(np.shares_memory(copy_.M, high.M))

        copy_.add_hash(0xFFFFFFFFFFFFFF)
        self.assertNotEqual(copy_, high)

    def test_downsample_invalid_precision(self):
        high = self._build_p8()

        with self.assertRaises(ValueError):
            high.downsample(3)
        with self.assertRaises(ValueError):
            high.downsample(17)
        with self.assertRaises(ValueError):
            high.downsample(10)
        with self.assertRaises(TypeError):
            high.downsample(6.0)

    def test_add_hash_validation(self):
        hll = hll_of_p(8)

        with self.assertRaises(ValueError):
            hll.add_hash(-1)
        with self.assertRaises(ValueError):
            hll.add_hash(1 << 64)

        with self.assertRaises(ValueError):
            hll.add_hash_bulk([0, 1 << 64])
        with self.assertRaises(ValueError):
            hll.add_hash_bulk([0, -1])
        with self.assertRaises(TypeError):
            hll.add_hash_bulk([0.5])

    def test_add_hash_bulk_matches_scalar(self):
        scalars = hll_of_p(8)
        for x in CRAFTED_HASHES_P8:
            scalars.add_hash(x)

        bulk = hll_of_p(8)
        bulk.add_hash_bulk(np.array(CRAFTED_HASHES_P8, dtype=np.uint64))

        self.assertEqual(register_values(bulk), register_values(scalars))

        empty = hll_of_p(8)
        empty.add_hash_bulk([])
        self.assertEqual(register_values(empty), [0] * 256)

        scalarish = hll_of_p(8)
        scalarish.add_hash_bulk(np.array(0x0000000000000101, dtype=np.uint64))
        single = hll_of_p(8)
        single.add_hash(0x0000000000000101)
        self.assertEqual(register_values(scalarish), register_values(single))

    def test_random_hashes_match_direct_summary(self):
        # exhaustive register conversion check over random bit patterns,
        # including hashes restricted to 16 bits so the w == 0 maximum-rank
        # case at p = 8 occurs frequently
        rng = np.random.RandomState(1234)

        for hi_p, lo_p in ((8, 6), (10, 4), (16, 8), (12, 12), (16, 4)):
            def rand64(count):
                hi = rng.randint(0, 1 << 32, size=count,
                                 dtype=np.int64).astype(np.uint64)
                lo = rng.randint(0, 1 << 32, size=count,
                                 dtype=np.int64).astype(np.uint64)
                return (hi << np.uint64(32)) | lo

            hashes = np.concatenate([
                rand64(3000),
                # small values make w == 0 / tiny-w maximum-rank cases
                # happen at the high precision
                rng.randint(0, 1 << hi_p, size=3000,
                            dtype=np.int64).astype(np.uint64),
                rng.randint(0, 1 << (hi_p + 3), size=3000,
                            dtype=np.int64).astype(np.uint64),
            ])

            high = hll_of_p(hi_p)
            high.add_hash_bulk(hashes)

            reference = hll_of_p(lo_p)
            reference.add_hash_bulk(hashes)

            low = high.downsample(lo_p)
            self.assertEqual(
                register_values(low), register_values(reference),
                msg='register mismatch for p=%d -> p=%d' % (hi_p, lo_p))

    def test_update_still_requires_equal_precision(self):
        a = hll_of_p(8)
        b = hll_of_p(6)
        with self.assertRaises(ValueError):
            a.update(b)

        with self.assertRaises(TypeError):
            a.update(object())

    def test_merge_mixed_precision(self):
        rng = np.random.RandomState(42)

        a = hll_of_p(10)
        b = hll_of_p(8)
        c = hll_of_p(6)

        xa = rng.randint(0, 1 << 16, size=2000, dtype=np.int64)
        xb = rng.randint(0, 1 << 16, size=2000, dtype=np.int64)
        xc = rng.randint(0, 1 << 16, size=2000, dtype=np.int64)

        a.add_hash_bulk(xa)
        b.add_hash_bulk(xb)
        c.add_hash_bulk(xc)

        all_hashes = np.concatenate([xa, xb, xc])

        merged = HyperLogLog.merge(a, b, c)
        self.assertEqual(merged.p, 6)

        reference = hll_of_p(6)
        reference.add_hash_bulk(all_hashes)
        self.assertEqual(register_values(merged), register_values(reference))

        # sources are not modified
        self.assertEqual(a.p, 10)
        self.assertEqual(b.p, 8)
        self.assertEqual(c.p, 6)

    def test_merge_at_explicit_common_precision(self):
        a = hll_of_p(10)
        b = hll_of_p(8)
        a.add_hash(0x123456789ABCDEF0)
        b.add_hash(0xFEDCBA9876543210)

        merged = HyperLogLog.merge(a, b, p=6)
        self.assertEqual(merged.p, 6)

        reference = hll_of_p(6)
        reference.add_hash(0x123456789ABCDEF0)
        reference.add_hash(0xFEDCBA9876543210)
        self.assertEqual(register_values(merged), register_values(reference))

        with self.assertRaises(ValueError):
            HyperLogLog.merge(a, b, p=9)

    def test_merge_single_counter(self):
        a = hll_of_p(8)
        a.add_hash(42)
        merged = HyperLogLog.merge(a)
        self.assertEqual(merged, a)
        self.assertIsNot(merged, a)

    def test_merge_validation(self):
        with self.assertRaises(ValueError):
            HyperLogLog.merge()

        a = hll_of_p(8)
        with self.assertRaises(TypeError):
            HyperLogLog.merge(a, object())
