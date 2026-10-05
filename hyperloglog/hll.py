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

    __slots__ = ('alpha', 'p', 'm', 'M')

    def __init__(self, error_rate):
        """
        Implementes a HyperLogLog

        error_rate = abs_err / cardinality
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
        self.M = np.zeros(self.m, HLL_COUNTER_TYPE)

    @classmethod
    def _from_p(cls, p, M=None):
        """
        Build a counter from an explicit precision p (instead of an error
        rate), optionally seeding it with an existing register vector
        """
        if not (4 <= p <= 16):
            raise ValueError("p=%d should be in range [4 : 16]" % p)

        hll = cls.__new__(cls)
        hll.alpha = get_alpha(p)
        hll.p = p
        hll.m = 1 << p

        if M is None:
            hll.M = np.zeros(hll.m, HLL_COUNTER_TYPE)
        else:
            hll.M = np.asarray(M, HLL_COUNTER_TYPE).reshape(hll.m).copy()

        return hll

    def __getstate__(self):
        return dict([x, getattr(self, x)] for x in self.__slots__)

    def __setstate__(self, d):
        for key in d:
            setattr(self, key, d[key])

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

        x = int.from_bytes(sha1(packb(value)).digest()[:8], byteorder='big')
        self.add_hash(x)

    def add_hash(self, x):
        """
        Adds an already computed 64-bit hash value to the HyperLogLog.

        This allows callers (and tests) to control the exact bit pattern
        that is fed into the register update.
        """
        if not (0 <= x < (1 << 64)):
            raise ValueError('hash value must be an unsigned 64-bit integer')

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

        x = np.fromiter((int.from_bytes(sha1(packb(value)).digest()[:8], byteorder='big') for value in values), np.uint64)
        self.add_hash_bulk(x)

    def add_hash_bulk(self, x):
        """
        Bulk-adds an iterable of already computed 64-bit hash values
        """
        x = np.asarray(x)

        if x.ndim == 0:
            x = x.reshape(1)

        if x.size == 0:
            return

        # validate against the original dtype so that out-of-range values
        # are not masked by a cast to uint64
        if np.issubdtype(x.dtype, np.integer):
            if np.any(x < 0) or np.any(x >= (1 << 64)):
                raise ValueError('hash values must be unsigned 64-bit integers')
            x = x.astype(np.uint64)
        else:
            # object arrays / lists holding Python ints: check each value
            items = x.reshape(-1).tolist()
            for value in items:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError('hash values must be integers')
                if not (0 <= value < (1 << 64)):
                    raise ValueError('hash values must be unsigned 64-bit integers')
            x = np.asarray(items, dtype=np.uint64).reshape(x.shape)

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
        Merge other counters of the same precision.

        Counters built for different error rates use different register
        layouts and cannot be merged directly. Use :meth:`downsample` to
        lower the precision explicitly and :meth:`merge` to combine
        counters at a common (lower) precision.
        """
        ml = [self.M]

        for item in others:
            if not isinstance(item, HyperLogLog):
                raise TypeError('can only merge HyperLogLog counters')
            if self.m != item.m:
                raise ValueError('Counters precisions should be equal')
            ml.append(item.M)

        self.M = np.maximum.reduce(ml)

    def copy(self):
        """
        Returns an independent copy of the counter
        """
        return self._from_p(self.p, self.M)

    def downsample(self, p):
        """
        Returns a copy of this counter reduced to precision p (p <= self.p).

        Precision can only be lowered, never raised: the information needed
        to split one coarse register into finer ones does not exist, so a
        low precision counter can never fake a higher one.

        The register values are recomputed from the 64-bit hash layout used
        by add(): the low p bits of the hash x are the index j and the
        remaining bits form w = x >> p, with rho(w) = 65 - p
        - bit_length(w) (bit_length(0) == 0).

        Lowering the precision by d = self.p - p bits makes every coarse
        bucket j receive strided source buckets j + c * 2**p for
        c in range(2**d) (NOT consecutive buckets, which would mix up the
        index bits). For a source register R the rank at the lower
        precision is:

          * R, if R < 65 - self.p (w had at least one 1 bit; shifting in
            the d lost index bits adds exactly d to its bit_length, so the
            rank at the lower precision is unchanged);
          * (65 - p) - bit_length(c), if R == 65 - self.p (w was all
            zeros). The maximum rank means the hashed value equalled the
            source bucket index j + c * 2**p, which is known while
            iterating the registers, so at the lower precision w equals c
            and rho can be recomputed exactly instead of being guessed or
            blindly clamped to the new maximum rank.

        This counter is left unchanged.
        """
        if not isinstance(p, int):
            raise TypeError('p must be an integer')

        if not (4 <= p <= 16):
            raise ValueError("p=%d should be in range [4 : 16]" % p)

        if p > self.p:
            raise ValueError('cannot downsample to higher precision')

        if p == self.p:
            return self.copy()

        d = self.p - p
        dst_m = 1 << p
        max_rank_src = 65 - self.p
        max_rank_dst = 65 - p

        # reshape as (2**d, 2**p): source buckets sharing coarse bucket j
        # are strided by dst_m, i.e. row c covers bucket j + c * dst_m
        grouped = self.M.astype(np.int64).reshape(2 ** d, dst_m)

        # R = max_rank_src means w was all zero at the source precision, so
        # the hashed value equalled the source bucket index; at the lower
        # precision w is therefore the stratum row index c, and rho is
        # (65 - p) - bit_length(c). bit_length(0) == 0 yields the new
        # maximum rank for row c = 0.
        row_rho = max_rank_dst - bit_length_vec(
            np.arange(2 ** d, dtype=np.int64))
        transformed = np.where(
            grouped == max_rank_src, row_rho[:, np.newaxis], grouped)

        M_dst = np.max(transformed, axis=0)

        return self._from_p(p, M_dst)

    @classmethod
    def merge(cls, *counters, p=None):
        """
        Merge counters configured with different error rates.

        All counters are downsampled to a common precision p, which
        defaults to the lowest precision among the inputs. p cannot exceed
        the precision of any input counter. The source counters are never
        modified.
        """
        if not counters:
            raise ValueError('at least one counter is required')

        for item in counters:
            if not isinstance(item, cls):
                raise TypeError('can only merge HyperLogLog counters')

        min_p = min(item.p for item in counters)

        if p is None:
            p = min_p

        if not isinstance(p, int):
            raise TypeError('p must be an integer')

        if p > min_p:
            raise ValueError(
                'common precision p=%d exceeds the lowest input precision %d'
                % (p, min_p))

        merged = counters[0].downsample(p)
        merged.update(*(item.downsample(p) for item in counters[1:]))
        return merged

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
