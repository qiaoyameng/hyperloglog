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

Joint intersection estimation:
==============================

Instead of the naive inclusion-exclusion estimate ``|A| + |B| - |A u B|``
(which often goes negative), two compatible sketches (same precision and
hash scheme) can be compared through their paired register distributions
using the joint maximum-likelihood estimator from [4].  It estimates the
three Venn components directly and reports the solver convergence state,
raising ``JointEstimationError`` instead of silently returning zero when
the numerical solve fails (e.g. saturated registers).

.. code-block:: python

    from hyperloglog import HyperLogLog
    a = HyperLogLog(0.01)
    b = HyperLogLog(0.01)
    # ... add items to a and b ...
    est = a.joint(b)        # or hyperloglog.joint_estimate(a, b)
    print(est.a_only)       # |A \ B|
    print(est.b_only)       # |B \ A|
    print(est.both)         # |A n B|  (aka est.intersection)
    print(est.union)        # |A u B|
    print(est.converged, est.iterations, est.residual)


Changes:
========
- Added Sliding window HLL version
- Added bias correction from HLL++
- Added joint intersection estimation for sketch pairs (Ertl's joint MLE)

References:
===========

1. http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf
2. http://hal.archives-ouvertes.fr/docs/00/46/53/13/PDF/sliding_HyperLogLog.pdf
3. http://research.google.com/pubs/pub40671.html
4. https://arxiv.org/abs/1702.01284

.. image:: https://deepwiki.com/badge.svg
   :target: https://deepwiki.com/svpcom/hyperloglog
