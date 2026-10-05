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


Sparse mode:
============

Counters that only ever see a handful of distinct values (e.g. one counter per
small tenant) can avoid allocating the full ``m = 2 ** p`` register array by
starting in the sparse representation.  The counter then keeps the distinct
64-bit hashes of the added values in a set — the cardinality is exact — and is
promoted to the dense registers once more than ``sparse_threshold`` distinct
values have been seen.  Promotion happens exactly once; re-adding known values
never moves the counter closer to the threshold.  Sparse and dense counters of
equal precision can be merged in either direction.

.. code-block:: python

    hll = hyperloglog.HyperLogLog(0.01, sparse=True, sparse_threshold=1000)
    hll.add("hello")
    print(len(hll))       # 1 (exact while sparse)
    print(hll.mode)       # 'sparse', becomes 'dense' after promotion

With the default 1% error rate the dense representation always holds 16384
registers, while a sparse counter with 100 distinct values stores only those
100 hashes.

Changes:
========
- Added Sliding window HLL version
- Added bias correction from HLL++
- Added optional sparse representation with one-time promotion to dense

References:
===========

1. http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf
2. http://hal.archives-ouvertes.fr/docs/00/46/53/13/PDF/sliding_HyperLogLog.pdf
3. http://research.google.com/pubs/pub40671.html

.. image:: https://deepwiki.com/badge.svg
   :target: https://deepwiki.com/svpcom/hyperloglog
