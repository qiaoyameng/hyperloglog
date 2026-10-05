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

    Every event carries an event-time timestamp and ``card(timestamp, window)``
    estimates the number of distinct values with event times in the half-open
    interval ``[timestamp - window, timestamp]``.  Because the summary may
    already hold events *newer* than a historical query, ``card`` only counts
    events with ``ts <= timestamp``; future events never inflate a historical
    estimate.

    The summary tracks two knobs that define which history is retained and
    which windows remain answerable:

    * ``watermark`` -- a monotonically advancing event-time lower bound,
      moved explicitly via ``advance_watermark()``.  ``None`` (the default)
      means no history has been discarded yet.
    * ``allowed_lateness`` -- how far behind the watermark late events may
      fall and still be accepted.

    Together they define the retention horizon ``H = watermark -
    allowed_lateness``:

    * ``add(timestamp, value)`` raises ``ValueError`` for events older than
      ``H``.  Late events inside ``[H, watermark)`` are still folded into the
      dominant points.
    * ``card``/``card_wlist`` answer exactly any window whose lower bound
      ``timestamp - window >= H`` and raise ``ValueError`` for older windows
      instead of returning a seemingly precise but wrong number.  Windows
      with ``timestamp - window >= watermark`` are final (no late event can
      change them); windows inside the lateness band ``[H, watermark)`` are
      answerable but may still be updated by late arrivals.
    * ``advance_watermark(t)`` safely drops every dominant point older than
      the new horizon ``t - allowed_lateness``; no future query can reference
      them again.

    While a watermark is set, all points newer than the horizon are kept
    verbatim (no LPFM dominance compression), because a future event may
    dominate -- and would evict -- a past point that a historical query still
    needs.  The price of exact historical queries is therefore bounded by the
    watermark, not by the sliding window: advance the watermark to control
    memory.  Without a watermark the legacy LPFM compression is used and only
    windows ending at or after the newest event are guaranteed.
    """

    __slots__ = ('window', 'alpha', 'p', 'm', 'LPFM', 'watermark', 'allowed_lateness')

    def __init__(self, error_rate, window, lpfm=None, allowed_lateness=0):
        """
        Implementes a Sliding HyperLogLog

        error_rate = abs_err / cardinality
        """

        self.window = window
        self.watermark = None

        if not (window > 0):
            raise ValueError('window must be > 0')

        if not (allowed_lateness >= 0):
            raise ValueError('allowed_lateness must be >= 0')

        self.allowed_lateness = allowed_lateness

        if lpfm is not None:
            m = len(lpfm)

            if m == 0 or (m & (m - 1)) != 0:
                raise ValueError('List length is not power of 2')

            p = m.bit_length() - 1
            self.LPFM = list(lpfm)

        else:
            if not (0 < error_rate < 1):
                raise ValueError("Error_Rate must be between 0 and 1.")

            # error_rate = 1.04 / sqrt(m)
            # m = 2 ** p

            p = math.ceil(math.log((1.04 / error_rate) ** 2, 2))
            m = 1 << p
            self.LPFM = [tuple() for i in range(m)]

        self.alpha = get_alpha(p)
        self.p = p
        self.m = m

    def __getstate__(self):
        return dict([x, getattr(self, x)] for x in self.__slots__)

    def __setstate__(self, d):
        for key in d:
            setattr(self, key, d[key])

        # defaults for pickles created before the time protocol existed
        if 'watermark' not in d:
            self.watermark = None

        if 'allowed_lateness' not in d:
            self.allowed_lateness = 0

        # normalize LPFM from pickles created by older versions, which used
        # None for empty registers and ascending timestamp order
        self.LPFM = [tuple(sorted(reg, reverse=True)) if reg else tuple() for reg in self.LPFM]

    @classmethod
    def from_list(cls, lpfm, window, watermark=None, allowed_lateness=0):
        obj = cls(None, window, lpfm, allowed_lateness=allowed_lateness)
        obj.watermark = watermark
        return obj

    def _horizon(self):
        """
        Oldest event time still retained / answerable, or None when unbounded.
        """
        if self.watermark is None:
            return None

        return self.watermark - self.allowed_lateness

    def advance_watermark(self, timestamp):
        """
        Moves the watermark forward and safely cleans up expired points.

        Every dominant point older than ``timestamp - allowed_lateness`` is
        dropped: all future queries have a window lower bound >= the horizon,
        so they can never reference those points again.
        """
        if self.watermark is not None and timestamp < self.watermark:
            raise ValueError('watermark cannot move backwards')

        self.watermark = timestamp
        horizon = self._horizon()
        self.LPFM = [tuple((t, R) for t, R in lpfm if t >= horizon) for lpfm in self.LPFM]

    def add(self, timestamp, value):
        """
        Adds the item to the HyperLogLog
        """
        horizon = self._horizon()

        if horizon is not None and timestamp < horizon:
            raise ValueError('event timestamp %r is too old: below the retention '
                             'horizon %r (watermark %r - allowed_lateness %r)' %
                             (timestamp, horizon, self.watermark, self.allowed_lateness))

        # h: D -> {0,1} ** 64
        # x = h(v)
        # j = <x_0x_1..x_{p-1})>
        # w = <x_{p}x_{p+1}..>
        # <t_i, rho(w)>

        x = int.from_bytes(sha1(packb(value)).digest()[:8], byteorder='big')
        j = x & (self.m - 1)
        w = x >> self.p
        R = get_rho(w, 64 - self.p)

        if horizon is not None:
            # Watermark protocol: keep every point newer than the horizon so
            # that historical queries stay exact; only expired points (which
            # no future query can reference) are dropped.
            tmp = []

            for t, R in heapq.merge(self.LPFM[j], ((timestamp, R),), reverse=True):
                if t < horizon:
                    break

                if tmp and tmp[-1][0] == t:
                    if R > tmp[-1][1]:
                        tmp[-1] = (t, R)
                    continue

                tmp.append((t, R))

            self.LPFM[j] = tuple(tmp)
            return

        Rmax = None
        tmp = []
        tmax = None

        for t, R in heapq.merge(self.LPFM[j], ((timestamp, R),), reverse=True):
            if tmax is None:
                tmax = t - self.window

            if t < tmax:
                break

            if Rmax is None or R > Rmax:
                tmp.append((t, R))
                Rmax = R

        self.LPFM[j] = tuple(tmp)

    def update(self, *others):
        """
        Merge other counters
        """

        for item in others:
            if self.m != item.m:
                raise ValueError('Counters precisions should be equal')

            if self.window != item.window:
                raise ValueError('Counters windows should be equal')

            if self.allowed_lateness != item.allowed_lateness:
                raise ValueError('Counters allowed_lateness should be equal')

        watermarks = [w for w in [self.watermark] + [item.watermark for item in others] if w is not None]
        watermark = max(watermarks) if watermarks else None

        if watermark is not None:
            # The merged summary can only guarantee what every input
            # guaranteed, so it inherits the highest watermark and keeps all
            # points newer than its horizon.
            horizon = watermark - self.allowed_lateness

            for j, lpfms_j in enumerate(zip(self.LPFM, *list(item.LPFM for item in others))):
                tmp = []

                for t, R in heapq.merge(*lpfms_j, reverse=True):
                    if t < horizon:
                        break

                    if tmp and tmp[-1][0] == t:
                        if R > tmp[-1][1]:
                            tmp[-1] = (t, R)
                        continue

                    tmp.append((t, R))

                self.LPFM[j] = tuple(tmp)

            self.watermark = watermark
            return

        for j, lpfms_j in enumerate(zip(self.LPFM, *list(item.LPFM for item in others))):
            Rmax = None
            tmp = []
            tmax = None

            for t, R in heapq.merge(*lpfms_j, reverse=True):
                if tmax is None:
                    tmax = t

                if t < (tmax - self.window):
                    break

                if Rmax is None or R > Rmax:
                    tmp.append((t, R))
                    Rmax = R

            self.LPFM[j] = tuple(tmp)

    def __eq__(self, other):
        if not isinstance(other, SlidingHyperLogLog):
            return NotImplemented

        return self.m == other.m and self.window == other.window and \
            self.watermark == other.watermark and \
            self.allowed_lateness == other.allowed_lateness and \
            self.LPFM == other.LPFM

    def __len__(self):
        raise NotImplementedError('use card(timestamp) to estimate cardinality')

    def card(self, timestamp, window=None):
        """
        Returns the estimate of the cardinality at 'timestamp' using 'window'

        Only events with ``ts <= timestamp`` are counted, so events recorded
        after ``timestamp`` never inflate a historical query.  Raises
        ``ValueError`` if the window lower bound is older than the retention
        horizon (watermark - allowed_lateness).
        """
        if window is None:
            window = self.window

        if not 0 < window <= self.window:
            raise ValueError('0 < window <= W')

        self._check_answerable(timestamp - window)

        _t = timestamp - window
        M = np.fromiter((np.max(np.fromiter((R for ts, R in lpfm if _t <= ts <= timestamp), int), initial=0) if lpfm else 0 for lpfm in self.LPFM), int)

        return get_estimate(M, self.m, self.p, self.alpha)

    def _check_answerable(self, lower_bound):
        horizon = self._horizon()

        if horizon is not None and lower_bound < horizon:
            raise ValueError('window lower bound %r is older than the retention '
                             'horizon %r (watermark %r - allowed_lateness %r): '
                             'the required events may already be discarded' %
                             (lower_bound, horizon, self.watermark, self.allowed_lateness))

    def card_wlist(self, timestamp, window_list):
        """
        Returns the estimate of the cardinality at 'timestamp' using list of windows
        """
        for window in window_list:
            if not 0 < window <= self.window:
                raise ValueError('0 < window <= W')

            self._check_answerable(timestamp - window)

        tsl = sorted((timestamp - window, idx) for idx, window in enumerate(window_list))
        M_list = [[] for _ in window_list]

        for lpfm in self.LPFM:
            R_max = 0
            _p = len(tsl) - 1

            for ts, R in lpfm:
                if ts > timestamp:
                    continue

                while _p >= 0:
                    _ts, _idx = tsl[_p]
                    if ts >= _ts: break
                    M_list[_idx].append(R_max)
                    _p -= 1
                if _p < 0: break
                R_max = R

            for i in range(0, _p + 1):
                M_list[tsl[i][1]].append(R_max)

        res = []
        for M in M_list:
            res.append(get_estimate(np.array(M, int), self.m, self.p, self.alpha))
        return res
