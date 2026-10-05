Python implementation of the Hyper LogLog and Sliding Hyper LogLog cardinality counter algorithms. 
==================================================================================================

Installation:
=============

Use ``pip install hyperloglog`` to install from PyPI.

Usage:
======

.. code-block:: python

    import hyperloglog
    hll = hyperloglog.HyperLogLog(0.01)  # accept 1% counting error
    hll.add("hello")
    print(len(hll))  # 1
    hll.add("hello")
    print(len(hll))  # 1 as items aren't added more than once
    hll.add("hello again")
    print(len(hll))  # 2

If we add a further 1000 random strings (giving a total of 1002 strings) we'll have a count roughly within 1% of the true value, in this case it counts 1007 (within +/- 10.2 of the true value)

.. code-block:: python

    # add 1000 random 30 char strings to hll
    import random
    import string
    [hll.add("".join([string.ascii_letters[random.randint(0, len(string.ascii_letters)-1)] for n in range(30)])) for m in range(1000)]  
    print(len(hll))  # 1007


Changes:
========
- Added Sliding window HLL version
- Added bias correction from HLL++

Sliding HyperLogLog time protocol:
==================================

``SlidingHyperLogLog.card(timestamp, window)`` estimates the cardinality of
events in ``[timestamp - window, timestamp]`` and never counts events newer
than ``timestamp``, so historical queries are not inflated by records that
arrived afterwards.  Two knobs define which history is retained:

.. code-block:: python

    from hyperloglog.shll import SlidingHyperLogLog
    shll = SlidingHyperLogLog(0.05, 100, allowed_lateness=10)
    shll.advance_watermark(1000)   # retention horizon H = 1000 - 10

* Events older than the horizon ``H = watermark - allowed_lateness`` are
  rejected by ``add``; late events inside ``[H, watermark)`` still update the
  dominant points.
* ``card``/``card_wlist`` answer exactly any window with lower bound
  ``timestamp - window >= H`` and raise ``ValueError`` for older windows
  instead of returning a seemingly precise but wrong number.
* ``advance_watermark(t)`` moves the watermark forward and safely drops
  points older than ``t - allowed_lateness``; once every event has expired
  the estimate is 0.
* Merging summaries with different watermarks yields the highest watermark;
  ``allowed_lateness`` must be equal.

References:
===========

1. http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf
2. http://hal.archives-ouvertes.fr/docs/00/46/53/13/PDF/sliding_HyperLogLog.pdf
3. http://research.google.com/pubs/pub40671.html

.. image:: https://deepwiki.com/badge.svg
   :target: https://deepwiki.com/svpcom/hyperloglog
