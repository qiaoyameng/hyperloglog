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

Joint estimation of two sketches:
=================================

``estimate_joint(a, b)`` estimates |A ∩ B|, |A \\ B| and |B \\ A| from the
joint distribution of the paired registers (joint MLE, see
https://arxiv.org/abs/1706.07290), instead of the naive
``card(A) + card(B) - card(A | B)`` inclusion-exclusion which may go
negative. Both sketches must be HyperLogLog counters with equal precision.

.. code-block:: python

    from hyperloglog import HyperLogLog, estimate_joint
    r = estimate_joint(a, b)
    print(r.intersection, r.a_only, r.b_only, r.union)
    print(r.converged, r.status)  # check solver convergence before trusting values

The components are always non-negative and satisfy the basic set bounds.
If the numerical solver does not converge, ``converged`` is False (and
``strict=True`` raises ``JointEstimationError``) instead of silently
returning a bogus value.

Changes:
========
- Added Sliding window HLL version
- Added bias correction from HLL++

References:
===========

1. http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf
2. http://hal.archives-ouvertes.fr/docs/00/46/53/13/PDF/sliding_HyperLogLog.pdf
3. http://research.google.com/pubs/pub40671.html

.. image:: https://deepwiki.com/badge.svg
   :target: https://deepwiki.com/svpcom/hyperloglog
