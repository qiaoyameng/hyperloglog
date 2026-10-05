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
    """

    __slots__ = ('alpha', 'p', 'm', 'M', 'sparse_threshold', '_sparse')

    def __init__(self, error_rate, sparse_threshold=None):
        """
        Implementes a HyperLogLog

        error_rate = abs_err / cardinality

        sparse_threshold: if set, keep an exact set of distinct hashes
        (sparse representation) until more than sparse_threshold distinct
        items are seen, then promote once to the dense register array.
        """

        if not (0 < error_rate < 1):
            raise ValueError("Error_Rate must be between 0 and 1.")

        # error_rate = 1.04 / sqrt(m)
        # m = 2 ** p
        # M(1)... M(m) = 0

        p = math.ceil(math.log((1.04 / error_rate) ** 2, 2))

        if sparse_threshold is not None:
            if isinstance(sparse_threshold, bool) or \
                    not isinstance(sparse_threshold, int):
                raise ValueError("sparse_threshold must be an integer")
            if not 1 <= sparse_threshold <= (1 << p):
                raise ValueError(
                    "sparse_threshold must be in range [1 : %d]" % (1 << p))

        self.alpha = get_alpha(p)
        self.p = p
        self.m = 1 << p
        self.sparse_threshold = sparse_threshold
        if sparse_threshold is None:
            self.M = np.zeros(self.m, HLL_COUNTER_TYPE)
            self._sparse = None
        else:
            self.M = None
            self._sparse = set()

    @property
    def is_sparse(self):
        """
        True while the counter uses the sparse (exact set) representation
        """
        return self._sparse is not None

    def __getstate__(self):
        state = dict([x, getattr(self, x)] for x in self.__slots__)
        # record the representation mode explicitly so that pickles in the
        # old format (no mode marker) are never silently reinterpreted
        state['mode'] = 'sparse' if self.is_sparse else 'dense'
        return state

    def __setstate__(self, d):
        d = dict(d)
        mode = d.pop('mode', None)

        if mode not in ('sparse', 'dense'):
            raise ValueError(
                'Unknown HyperLogLog serialization format: '
                'missing or invalid representation mode')

        for key in d:
            setattr(self, key, d[key])

        if mode == 'sparse':
            if not isinstance(self._sparse, set) or self.M is not None:
                raise ValueError('Inconsistent sparse HyperLogLog state')
        else:
            if self._sparse is not None or self.M is None:
                raise ValueError('Inconsistent dense HyperLogLog state')
            # normalize M from pickles created under the other NumPy major
            # version (different counter dtype)
            self.M = np.asarray(self.M, HLL_COUNTER_TYPE)

    def _add_hash(self, x):
        # j = <x_0x_1..x_{p-1}>
        # w = <x_{p}x_{p+1}..>
        # M[j] = max(M[j], rho(w))
        j = x & (self.m - 1)
        w = x >> self.p
        rho = get_rho(w, 64 - self.p)

        if rho > self.M[j]:
            self.M[j] = rho

    def _promote(self):
        """
        One-way switch from the sparse hash set to dense registers
        """
        x = np.fromiter(self._sparse, np.uint64, len(self._sparse))
        j = x & (self.m - 1)
        w = x >> self.p
        rho = get_rho_vec(w, 64 - self.p)

        M = np.zeros(self.m, HLL_COUNTER_TYPE)
        np.maximum.at(M, j, rho)

        self.M = M
        self._sparse = None

    def _add_sparse_hash(self, x):
        self._sparse.add(x)

        if len(self._sparse) > self.sparse_threshold:
            self._promote()

    def add(self, value):
        """
        Adds the item to the HyperLogLog
        """
        # h: D -> {0,1} ** 64
        # x = h(v)
        # j = <x_0x_1..x_{p-1}>
        # w = <x_{p}x_{p+1}..>
        # M[j] = max(M[j], rho(w))

        x = int.from_bytes(sha1(packb(value)).digest()[:8], byteorder='big')

        if self.is_sparse:
            self._add_sparse_hash(x)
        else:
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

        x = np.fromiter((int.from_bytes(sha1(packb(value)).digest()[:8], byteorder='big') for value in values), np.uint64)

        if self.is_sparse:
            self._sparse.update(x.tolist())

            if len(self._sparse) > self.sparse_threshold:
                self._promote()
            return

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
        """
        for item in others:
            if self.m != item.m:
                raise ValueError('Counters precisions should be equal')

        for item in others:
            if self.is_sparse and item.is_sparse:
                self._sparse |= item._sparse
                continue

            if self.is_sparse:
                self._promote()

            if item.is_sparse:
                for x in item._sparse:
                    self._add_hash(x)
            else:
                self.M = np.maximum(self.M, item.M)

        if self.is_sparse and len(self._sparse) > self.sparse_threshold:
            self._promote()

    def __eq__(self, other):
        if not isinstance(other, HyperLogLog):
            return NotImplemented

        if self.m != other.m or self.is_sparse != other.is_sparse:
            return False

        if self.is_sparse:
            return self._sparse == other._sparse

        return np.array_equal(self.M, other.M)

    def __len__(self):
        return round(self.card())

    def card(self):
        """
        Returns the estimate of the cardinality
        """
        if self.is_sparse:
            # sparse mode stores every distinct hash, so the count is exact
            return float(len(self._sparse))

        return get_estimate(self.M, self.m, self.p, self.alpha)
