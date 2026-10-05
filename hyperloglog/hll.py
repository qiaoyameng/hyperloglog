"""
This module implements probabilistic data structure which is able to calculate the cardinality of large multisets in a single pass using little auxiliary memory
"""

import math
import operator
import numpy as np
from hashlib import sha1
from msgpack import packb

from .const import rawEstimateData as _rawEstimateData, \
                   biasData as _biasData, \
                   tresholdData as _tresholdData

rawEstimateData = tuple(np.array(row) for row in _rawEstimateData)
biasData = tuple(np.array(row) for row in _biasData)
tresholdData = tuple(np.array(row) for row in _tresholdData)


def hash_value(value):
    return int.from_bytes(sha1(packb(value)).digest()[:8], byteorder='big')


def as_hash_value(x):
    try:
        x = operator.index(x)
    except TypeError as error:
        raise ValueError('hash must be an unsigned 64-bit integer') from error

    if not (0 <= x < 1 << 64):
        raise ValueError('hash must be an unsigned 64-bit integer')

    return x


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

    __slots__ = ('alpha', 'p', 'm', 'M', 'hash_func')

    def __init__(self, error_rate, hash_func=None):
        """
        Implementes a HyperLogLog

        error_rate = abs_err / cardinality
        hash_func optionally maps a value to an unsigned 64-bit integer
        """

        if not (0 < error_rate < 1):
            raise ValueError("Error_Rate must be between 0 and 1.")

        # error_rate = 1.04 / sqrt(m)
        # m = 2 ** p
        # M(1)... M(m) = 0

        p = math.ceil(math.log((1.04 / error_rate) ** 2, 2))

        self.hash_func = hash_func if hash_func is not None else hash_value
        self.alpha = get_alpha(p)
        self.p = p
        self.m = 1 << p
        self.M = np.zeros(self.m, HLL_COUNTER_TYPE)

    def __getstate__(self):
        return dict([x, getattr(self, x)] for x in self.__slots__)

    def __setstate__(self, d):
        for key in d:
            setattr(self, key, d[key])

        if not hasattr(self, 'hash_func'):
            self.hash_func = hash_value

        # normalize M from pickles created by pre-NumPy versions (plain list)
        # or under the other NumPy major version (different counter dtype)
        self.M = np.asarray(self.M, HLL_COUNTER_TYPE)

    def add(self, value):
        """
        Adds the item to the HyperLogLog
        """
        # h: D -> {0,1} ** 64
        # x = h(v)
        # j = <x_0x_1..x_{p-1}>
        # w = <x_{p}x_{p+1}..>
        # M[j] = max(M[j], rho(w))

        x = as_hash_value(self.hash_func(value))
        j = x & (self.m - 1)
        w = x >> self.p
        rho = get_rho(w, 64 - self.p)

        if rho > self.M[j]:
            self.M[j] = rho

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

        x = np.fromiter((as_hash_value(self.hash_func(value)) for value in values), np.uint64)
        j = x & (self.m - 1)
        w = x >> self.p
        rho = get_rho_vec(w, 64 - self.p)

        unique_j, inverse_indices = np.unique(j, return_inverse=True)
        sort_inv = np.argsort(inverse_indices)
        group_start_indices = np.searchsorted(inverse_indices[sort_inv], np.arange(len(unique_j)))
        grouped_rho = np.maximum.reduceat(rho[sort_inv], group_start_indices)

        self.M[unique_j] = np.maximum(self.M[unique_j], grouped_rho)

    def downsample(self, p):
        """
        Return a copy reduced to precision p without changing this summary.
        """
        if not isinstance(p, int) or not (4 <= p <= self.p):
            raise ValueError('p should be an integer in range [4 : %d]' % self.p)

        result = HyperLogLog.__new__(HyperLogLog)
        result.alpha = get_alpha(p)
        result.p = p
        result.m = 1 << p
        result.M = np.zeros(result.m, HLL_COUNTER_TYPE)
        result.hash_func = self.hash_func

        source_max_rank = 64 - self.p + 1
        target_max_rank = 64 - p + 1

        for source_j, rank in enumerate(self.M):
            if rank == 0:
                continue

            rank = int(rank)
            target_j = source_j & (result.m - 1)

            if rank == source_max_rank:
                removed_bucket_bits = source_j >> p
                if removed_bucket_bits == 0:
                    rank = target_max_rank
                else:
                    rank = target_max_rank - removed_bucket_bits.bit_length()

            if rank > result.M[target_j]:
                result.M[target_j] = rank

        return result

    def update(self, *others):
        """
        Merge other counters
        """
        ml = [self.M]

        for item in others:
            if self.m != item.m:
                raise ValueError('Counters precisions should be equal')
            ml.append(item.M)

        self.M = np.maximum.reduce(ml)

    def __eq__(self, other):
        if not isinstance(other, HyperLogLog):
            return NotImplemented

        return self.m == other.m and np.array_equal(self.M, other.M)

    def __len__(self):
        return round(self.card())

    def card(self):
        """
        Returns the estimate of the cardinality
        """
        return get_estimate(self.M, self.m, self.p, self.alpha)
