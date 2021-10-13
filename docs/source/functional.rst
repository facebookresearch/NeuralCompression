neuralcompression.functional
============================

.. currentmodule:: neuralcompression.functional

.. autofunction:: dense_image_warp
.. autofunction:: hsv2rgb
.. autofunction:: optical_flow_to_color


Entropy coding
--------------

Range coding
^^^^^^^^^^^^

.. autofunction:: unbounded_index_range_encode
.. autofunction:: unbounded_index_range_decode

Metrics
-------

Complexity
^^^^^^^^^^

.. autofunction:: count_flops

Information
^^^^^^^^^^^

.. autofunction:: information_content

Quality
^^^^^^^

.. autofunction:: learned_perceptual_image_patch_similarity,
.. autofunction:: multiscale_structural_similarity

Probability
-----------

Cumulative distribution function (CDF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: log_cdf
.. autofunction:: ndtr
.. autofunction:: log_ndtr

Shape
^^^^^

.. autofunction:: estimate_tails
.. autofunction:: lower_tail
.. autofunction:: upper_tail

Survival function
^^^^^^^^^^^^^^^^^

.. autofunction:: sf
.. autofunction:: log_sf

Rounding
--------

.. autofunction:: soft_round
.. autofunction:: soft_round_conditional_mean
.. autofunction:: soft_round_inverse