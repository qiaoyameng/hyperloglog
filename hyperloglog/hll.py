"""
This module implements probabilistic data structure which is able to calculate the cardinality of large multisets in a single pass using little auxiliary memory
"""

import math
import numpy as np
from hashlib import sha1
from msgpack import packb

from .const import rawEstimateData as _rawEstimateData, \
                   biasData as _biasData, \
                   tresholdData as _tresholdData

rawEstimateData = tuple(np.array(row) for row in _rawEstimateData)
biasData = tuple(np.array(row) for row in _biasData)
tresholdData = tuple(np.array(row) for row in _tresholdData)


def get_treshold(p):
    return tresholdData[p - 4]


def estimate_bias(E, p):
    bias_vector = biasData[p - 4]
    nearest_neighbors = get_nearest_neighbors(E, rawEstimateData[p - 4])
    return np.sum(bias_vector[nearest_neighbors]) / len(nearest_neighbors)


def get_nearest_neighbors(E, estimate_vector):
    return np.argsort((E - estimate_vector) ** 2)[:6]


def get_alpha(p):
    if not (4 <= p <= 16):
        raise ValueError("p=%d should be in range [4 : 16]" % p)

    if p == 4:
        return 0.673

    if p == 5:
        return 0.697

    if p == 6:
        return 0.709

    return 0.7213 / (1.0 + 1.079 / (1 << p))


def get_rho(w, max_width):
    rho = max_width - w.bit_length() + 1

    if rho <= 0:
        raise ValueError('w overflow')

    return rho


# Check for NumPy 2.x
if hasattr(np,'bitwise_count'):
    def bit_length_vec(arr):
        bits = arr >> 1
        bits |= arr
        bits |= bits >> 2
        bits |= bits >> 4
        bits |= bits >> 8
        bits |= bits >> 16
        bits |= bits >> 32
        return np.bitwise_count(bits)

    # NumPy 2.x doesn't have performance drawback for small integers
    HLL_COUNTER_TYPE = np.int8

else:
    def bit_length_vec(arr):
        _, high_exp = np.frexp(arr >> 32)
        _, low_exp = np.frexp(arr & 0xFFFFFFFF)
        return np.where(high_exp, high_exp + 32, low_exp)

    # int8/16/32 are smaller but much slower than int64 for NumPy 1.x
    HLL_COUNTER_TYPE = np.int64


def get_rho_vec(w, max_width):
    # compare before subtracting: bit_length_vec may return an unsigned
    # dtype (NumPy 2.x), where a negative rho would silently wrap around
    bits = bit_length_vec(w)

    if np.count_nonzero(bits > max_width):
        raise ValueError('w overflow')

    return max_width - bits + 1


def get_estimate(M, m, p, alpha):
    # small-range linear counting below the empirical threshold, otherwise
    # the HLL raw estimate (bias-corrected while E <= 5m)
    V = np.count_nonzero(M == 0)

    if V > 0:
        H = m * math.log(m / V)
        if H <= get_treshold(p):
            return H

    E = alpha * (m ** 2) / np.power(2.0, -M, dtype=float).sum()
    return (E - estimate_bias(E, p)) if E <= 5 * m else E


class HyperLogLog(object):
    """
    HyperLogLog cardinality counter

    By default the counter allocates the full dense register array of
    m = 2 ** p counters up front.  With sparse=True it starts in the
    sparse representation instead, keeping the distinct 64-bit hashes
    of the added values in a set, and is promoted to the dense
    representation once more than sparse_threshold distinct values have
    been seen.  Promotion happens exactly once and is never reverted.
    """

    __slots__ = ('alpha', 'p', 'm', 'M', 'mode', 'sparse_threshold')

    MODE_SPARSE = 'sparse'
    MODE_DENSE = 'dense'

    def __init__(self, error_rate, sparse=False, sparse_threshold=None):
        """
        Implementes a HyperLogLog

        error_rate = abs_err / cardinality
        sparse = start in the sparse representation
        sparse_threshold = promote to dense after this many distinct
            values (default m // 4); fixed at creation together with
            the precision
        """

        if not (0 < error_rate < 1):
            raise ValueError("Error_Rate must be between 0 and 1.")

        # error_rate = 1.04 / sqrt(m)
        # m = 2 ** p
        # M(1)... M(m) = 0

        p = math.ceil(math.log((1.04 / error_rate) ** 2, 2))

        self.alpha = get_alpha(p)
        self.p = p
        self.m = 1 << p

        if sparse:
            if sparse_threshold is None:
                sparse_threshold = self.m // 4

            if not isinstance(sparse_threshold, int) or \
                    isinstance(sparse_threshold, bool) or \
                    not (0 < sparse_threshold <= self.m):
                raise ValueError("sparse_threshold must be an integer in range [1 : m]")

            self.mode = self.MODE_SPARSE
            self.M = set()
            self.sparse_threshold = sparse_threshold
        else:
            if sparse_threshold is not None:
                raise ValueError("sparse_threshold requires sparse=True")

            self.mode = self.MODE_DENSE
            self.M = np.zeros(self.m, HLL_COUNTER_TYPE)
            self.sparse_threshold = None

    def __getstate__(self):
        return dict([x, getattr(self, x)] for x in self.__slots__)

    def __setstate__(self, d):
        # the representation mode is recorded explicitly; states pickled
        # by older versions (no 'mode' key) are rejected rather than
        # silently reinterpreted under the new layout
        if 'mode' not in d:
            raise ValueError('Cannot load a pickle without a representation '
                             'mode (legacy format); re-create the counter')

        if d['mode'] not in (self.MODE_SPARSE, self.MODE_DENSE):
            raise ValueError('Unknown representation mode: %r' % (d['mode'],))

        for key in d:
            setattr(self, key, d[key])

        if self.mode == self.MODE_DENSE:
            # normalize M from pickles created by pre-NumPy versions (plain
            # list) or under the other NumPy major version (different dtype)
            self.M = np.asarray(self.M, HLL_COUNTER_TYPE)
        else:
            self.M = set(self.M)

    @staticmethod
    def _hash(value):
        # h: D -> {0,1} ** 64
        # x = h(v)
        return int.from_bytes(sha1(packb(value)).digest()[:8], byteorder='big')

    def _add_hash(self, x):
        # j = <x_0x_1..x_{p-1}>
        # w = <x_{p}x_{p+1}..>
        # M[j] = max(M[j], rho(w))
        j = x & (self.m - 1)
        w = x >> self.p
        rho = get_rho(w, 64 - self.p)

        if rho > self.M[j]:
            self.M[j] = rho

    def _sparse_registers(self):
        # dense register image of the hashes held by the sparse set
        M = np.zeros(self.m, HLL_COUNTER_TYPE)

        for x in self.M:
            j = x & (self.m - 1)
            w = x >> self.p
            rho = get_rho(w, 64 - self.p)

            if rho > M[j]:
                M[j] = rho

        return M

    def _promote(self):
        # one-way switch from the sparse hash set to dense registers
        if self.mode != self.MODE_SPARSE:
            raise RuntimeError('counter is not in the sparse representation')

        self.M = self._sparse_registers()
        self.mode = self.MODE_DENSE

    def add(self, value):
        """
        Adds the item to the HyperLogLog
        """
        x = self._hash(value)

        if self.mode == self.MODE_SPARSE:
            # the set deduplicates, so re-adding a value never moves
            # the counter closer to the promotion threshold
            self.M.add(x)

            if len(self.M) > self.sparse_threshold:
                self._promote()

            return

        self._add_hash(x)

    def add_bulk(self, values):
        """
        Adds the item to the HyperLogLog
        """
        # h: D -> {0,1} ** 64
        # x = h(v)
        # j = <x_0x_1..x_{p-1}>
        # w = <x_{p}x_{p+1}..>
        # M[j] = max(M[j], rho(w))

        if isinstance(values, (bytes, str)) or not hasattr(values, '__iter__'):
            raise TypeError('values must be a non-string iterable')

        if self.mode == self.MODE_SPARSE:
            for value in values:
                self.add(value)

            return

        x = np.fromiter((int.from_bytes(sha1(packb(value)).digest()[:8], byteorder='big') for value in values), np.uint64)
        j = x & (self.m - 1)
        w = x >> self.p
        rho = get_rho_vec(w, 64 - self.p)

        unique_j, inverse_indices = np.unique(j, return_inverse=True)
        sort_inv = np.argsort(inverse_indices)
        group_start_indices = np.searchsorted(inverse_indices[sort_inv], np.arange(len(unique_j)))
        grouped_rho = np.maximum.reduceat(rho[sort_inv], group_start_indices)

        self.M[unique_j] = np.maximum(self.M[unique_j], grouped_rho)

    def update(self, *others):
        """
        Merge other counters

        Sparse and dense counters of equal precision can be merged in
        either direction; the merged counters are left unchanged.  A
        sparse self is promoted first when any other is dense or the
        combined distinct hashes exceed its threshold.
        """
        for item in others:
            if self.m != item.m:
                raise ValueError('Counters precisions should be equal')

        if self.mode == self.MODE_SPARSE:
            dense_seen = False

            for item in others:
                if item.mode == self.MODE_SPARSE:
                    self.M |= item.M
                else:
                    dense_seen = True

            if dense_seen or len(self.M) > self.sparse_threshold:
                self._promote()
            else:
                return

        ml = [self.M]

        for item in others:
            if item.mode == self.MODE_SPARSE:
                ml.append(item._sparse_registers())
            else:
                ml.append(item.M)

        self.M = np.maximum.reduce(ml)

    def __eq__(self, other):
        if not isinstance(other, HyperLogLog):
            return NotImplemented

        if self.m != other.m or self.mode != other.mode:
            return False

        if self.mode == self.MODE_SPARSE:
            return self.M == other.M

        return np.array_equal(self.M, other.M)

    def __len__(self):
        return round(self.card())

    def card(self):
        """
        Returns the estimate of the cardinality

        The sparse representation holds every distinct hash, so its
        cardinality is exact.
        """
        if self.mode == self.MODE_SPARSE:
            return len(self.M)

        return get_estimate(self.M, self.m, self.p, self.alpha)
