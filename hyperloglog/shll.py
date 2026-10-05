"""
Sliding HyperLogLog
"""

import math
import heapq
import numpy as np

from hashlib import sha1
from msgpack import packb
from .hll import get_alpha, get_rho, get_estimate


class SlidingHyperLogLog(object):
    """
    Sliding HyperLogLog: Estimating cardinality in a data stream (Telecom ParisTech)

    Time protocol
    -------------
    The summary keeps an explicit *watermark*: the largest event
    timestamp seen so far (it can also be moved forward manually with
    ``advance_watermark()``).  Together with ``max_lateness`` it defines
    which events are still accepted and which time windows the LPFM
    registers can still answer for:

    * Ingestion: ``add(ts, value)`` accepts an event only while
      ``ts >= watermark - max_lateness``.  Events older than that are
      rejected with ``ValueError`` -- the windows they belong to are
      already finalized and the summary may have cleaned them up.
      Events that are late but still inside the allowed lateness are
      merged into the dominant points of their register.

    * Queries: ``card(T, w)`` is answered only when the window
      ``[T - w, T]`` is fully covered by the retained summary:

        - ``T >= watermark - max_lateness``: the queried period is
          complete, no still-acceptable late event can change it, and
        - ``T - w >= watermark - window``: the window does not reach
          before the retention horizon, older events may have expired.

      Queries violating either condition raise ``ValueError`` instead of
      returning a seemingly precise but wrong number.  Events with
      ``ts > T`` never contribute to ``card(T, w)``: historical queries
      are not inflated by records that arrived "from the future".

    * Cleanup: a retained point is dropped once it expires
      (``ts < watermark - window``) or once it is dominated by a newer
      point at or before the queryable horizon
      (``watermark - max_lateness``), because every still-allowed query
      that could see the dominated point is then guaranteed to see its
      dominator as well.  This cleanup happens whenever the watermark
      advances: in ``add()``, ``update()`` and ``advance_watermark()``.
    """

    __slots__ = ('window', 'max_lateness', 'alpha', 'p', 'm', 'LPFM', 'watermark')

    def __init__(self, error_rate, window, max_lateness=0, lpfm=None, watermark=None):
        """
        Implementes a Sliding HyperLogLog

        error_rate = abs_err / cardinality
        max_lateness = how far behind the watermark an event timestamp may
                       lag and still be accepted / queried
        """

        self.window = window

        if not (window > 0):
            raise ValueError('window must be > 0')

        if not (max_lateness >= 0):
            raise ValueError('max_lateness must be >= 0')

        self.max_lateness = max_lateness

        if lpfm is not None:
            m = len(lpfm)

            if m == 0 or (m & (m - 1)) != 0:
                raise ValueError('List length is not power of 2')

            p = m.bit_length() - 1
            self.LPFM = [tuple(sorted(reg, reverse=True)) if reg else tuple() for reg in lpfm]

        else:
            if not (0 < error_rate < 1):
                raise ValueError("Error_Rate must be between 0 and 1.")

            # error_rate = 1.04 / sqrt(m)
            # m = 2 ** p

            p = math.ceil(math.log((1.04 / error_rate) ** 2, 2))
            m = 1 << p
            self.LPFM = [tuple() for i in range(m)]

        if watermark is None:
            # registers are sorted by descending timestamp
            watermark = max((reg[0][0] for reg in self.LPFM if reg), default=None)

        self.watermark = watermark
        self.alpha = get_alpha(p)
        self.p = p
        self.m = m

    def __getstate__(self):
        return dict([x, getattr(self, x)] for x in self.__slots__)

    def __setstate__(self, d):
        for key in d:
            setattr(self, key, d[key])

        # normalize LPFM from pickles created by older versions, which used
        # None for empty registers and ascending timestamp order
        self.LPFM = [tuple(sorted(reg, reverse=True)) if reg else tuple() for reg in self.LPFM]

        # defaults for pickles created before the time protocol existed
        if 'max_lateness' not in d:
            self.max_lateness = 0

        if 'watermark' not in d:
            self.watermark = max((reg[0][0] for reg in self.LPFM if reg), default=None)

    @classmethod
    def from_list(cls, lpfm, window, max_lateness=0, watermark=None):
        return cls(None, window, max_lateness=max_lateness, lpfm=lpfm, watermark=watermark)

    def _merge_points(self, *lists, expiry=None):
        """
        Merge sorted (descending timestamp) dominant-point lists into one,
        keeping only the points every still-answerable query may need:

        * points older than the retention horizon are expired;
        * a point dominated by a newer point with R at least as large is
          dropped only if that dominator is not newer than the queryable
          horizon (watermark - max_lateness), so any allowed historical
          query that contains the dominated point also contains a
          dominator with an R at least as large.
        """
        q_horizon = self.watermark - self.max_lateness if self.watermark is not None else None

        Rmax = None
        t_dom = None
        tmax = None
        tmp = []

        for t, R in heapq.merge(*lists, reverse=True):
            if tmax is None:
                tmax = t - self.window

            cut = tmax if expiry is None else expiry

            if t < cut:
                break

            if Rmax is None or R > Rmax:
                tmp.append((t, R))
                Rmax = R
                t_dom = t
            elif q_horizon is None or t_dom > q_horizon:
                # dominated, but the dominator is too close to the
                # watermark: historical queries may still need this point
                tmp.append((t, R))

        return tuple(tmp)

    def add(self, timestamp, value):
        """
        Adds the item to the HyperLogLog
        """
        if self.watermark is not None and timestamp < self.watermark - self.max_lateness:
            raise ValueError('event timestamp %r is too late: it is below '
                             'watermark %r minus max_lateness %r' %
                             (timestamp, self.watermark, self.max_lateness))

        if self.watermark is None or timestamp > self.watermark:
            self.watermark = timestamp

        # h: D -> {0,1} ** 64
        # x = h(v)
        # j = <x_0x_1..x_{p-1})>
        # w = <x_{p}x_{p+1}..>
        # <t_i, rho(w)>

        x = int.from_bytes(sha1(packb(value)).digest()[:8], byteorder='big')
        j = x & (self.m - 1)
        w = x >> self.p
        R = get_rho(w, 64 - self.p)

        self.LPFM[j] = self._merge_points(self.LPFM[j], ((timestamp, R),))

    def advance_watermark(self, timestamp):
        """
        Moves the watermark forward (never backward) and safely cleans up
        every register: points that expired relative to the new watermark
        or that are dominated at or before the new queryable horizon are
        removed.  After the watermark passes an event's timestamp plus
        max_lateness, that event can no longer be added.
        """
        if self.watermark is None or timestamp > self.watermark:
            self.watermark = timestamp

        expiry = self.watermark - self.window
        self.LPFM = [self._merge_points(reg, expiry=expiry) for reg in self.LPFM]

    def update(self, *others):
        """
        Merge other counters
        """

        for item in others:
            if self.m != item.m:
                raise ValueError('Counters precisions should be equal')

            if self.window != item.window:
                raise ValueError('Counters windows should be equal')

            if self.max_lateness != item.max_lateness:
                raise ValueError('Counters max_lateness should be equal')

        watermarks = [w for w in [self.watermark] + [item.watermark for item in others]
                      if w is not None]

        if watermarks:
            self.watermark = max(watermarks)

        for j, lpfms_j in enumerate(zip(self.LPFM, *list(item.LPFM for item in others))):
            self.LPFM[j] = self._merge_points(*lpfms_j)

    def __eq__(self, other):
        if not isinstance(other, SlidingHyperLogLog):
            return NotImplemented

        return self.m == other.m and self.window == other.window and \
            self.max_lateness == other.max_lateness and \
            self.watermark == other.watermark and self.LPFM == other.LPFM

    def __len__(self):
        raise NotImplementedError('use card(timestamp) to estimate cardinality')

    def _check_query(self, timestamp, window):
        if self.watermark is None:
            return

        q_horizon = self.watermark - self.max_lateness

        if timestamp < q_horizon:
            raise ValueError('query timestamp %r is before the queryable horizon %r '
                             '(watermark %r - max_lateness %r): the queried period '
                             'may still receive late events' %
                             (timestamp, q_horizon, self.watermark, self.max_lateness))

        horizon = self.watermark - self.window

        if timestamp - window < horizon:
            raise ValueError('query window [%r, %r] reaches before the retention '
                             'horizon %r (watermark %r - window %r): events that old '
                             'may have expired from the summary' %
                             (timestamp - window, timestamp, horizon,
                              self.watermark, self.window))

    def card(self, timestamp, window=None):
        """
        Returns the estimate of the cardinality at 'timestamp' using 'window'
        """
        if window is None:
            window = self.window

        if not 0 < window <= self.window:
            raise ValueError('0 < window <= W')

        self._check_query(timestamp, window)

        _t = timestamp - window
        M = np.fromiter((np.max(np.fromiter((R for ts, R in lpfm if _t <= ts <= timestamp), int), initial=0) if lpfm else 0 for lpfm in self.LPFM), int)

        return get_estimate(M, self.m, self.p, self.alpha)

    def card_wlist(self, timestamp, window_list):
        """
        Returns the estimate of the cardinality at 'timestamp' using list of windows
        """
        for window in window_list:
            if not 0 < window <= self.window:
                raise ValueError('0 < window <= W')

            self._check_query(timestamp, window)

        tsl = sorted((timestamp - window, idx) for idx, window in enumerate(window_list))
        M_list = [[] for _ in window_list]

        for lpfm in self.LPFM:
            R_max = 0
            _p = len(tsl) - 1

            for ts, R in lpfm:
                if ts > timestamp:
                    # records from the future must not inflate the past
                    continue
                while _p >= 0:
                    _ts, _idx = tsl[_p]
                    if ts >= _ts: break
                    M_list[_idx].append(R_max)
                    _p -= 1
                if _p < 0: break
                if R > R_max:
                    R_max = R

            for i in range(0, _p + 1):
                M_list[tsl[i][1]].append(R_max)

        res = []
        for M in M_list:
            res.append(get_estimate(np.array(M, int), self.m, self.p, self.alpha))
        return res
