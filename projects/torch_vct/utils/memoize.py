# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Memoize decorator for PyTorch computations. 

This module is based on `tf_memoize.py` module from VCT 
(https://github.com/google-research/google-research/tree/master/vct/src). 

## Example usage:

>>> import memoize

>>> @memoize.memoize
>>> def my_function(x, y, z):
>>>     return x + y + z


>>> cache = memoize.create_cache() # create cache
>>> my_function_bound = memoize.bind(my_function, cache) # bind it

>>> x0, y0, z0 = 2, 3, 5
>>> print(my_function_bound(x0, y0, z0))
>>> print(my_function_bound(x0, y0, z0)) # cache will be hit on second call
>>> assert my_function.get_total_cache_hits(cache) == 1

### No caching: 

>>> print(my_function(x0, y0, z0)) # call function wihtout binding...
>>> print(my_function(x0, y0, z0))  # no cache hit

>>> my_function_nocache = memoize.bind(my_function, cache=None) # ... or bind to None
>>> my_function_nocache(x0, y0, z0)
>>> my_function_nocache(x0, y0, z0)  # no cache hit, so same as before

>>> assert my_function.get_total_cache_hits(cache) == 1
"""
import collections
import functools
from typing import Callable, Optional, Tuple, TypeVar, Union

from torch import Tensor


# Subclass `object` instead of `NamedTuple` to avoid recursive tracking in
# tf.Module, which breaks on the internal cache dicts.
class _Cache:
    def __init__(self, cache, hits):
        self.cache = cache
        self.hits = hits

    def __iter__(self):
        return iter([self.cache, self.hits])


Cache = Optional[_Cache]

ReturnType = TypeVar("ReturnType")
WithCache = Tuple[ReturnType, _Cache]


def create_cache():
    """Creates a memoize cache."""
    return _Cache(
        cache=collections.defaultdict(dict),
        hits=collections.defaultdict(lambda: collections.defaultdict(int)),
    )


def _ensure_hashable(x: Union[list, tuple, dict, Tensor]):
    """Return a version of x that is hashable."""
    if isinstance(x, (list, tuple)):
        return tuple(_ensure_hashable(y) for y in x)

    if isinstance(x, dict):
        return tuple((_ensure_hashable(k), _ensure_hashable(v)) for k, v in x.items())

    if isinstance(x, Tensor):
        return hash(x)

    assert hash(x) is not None
    return x


def memoize(f):
    """
    Memoize decorator

    Uses a cache provided through `memoize.bind`
    """

    @functools.wraps(f)
    def wrapper(
        *args, _private_cache_kwarg: Cache = None, _expect_cache_hit=None, **kwargs
    ):
        cache = _private_cache_kwarg

        if cache is None:  # no caching
            assert _expect_cache_hit in [False, None]
            return f(*args, **kwargs)

        result_cache, hit_counter = cache

        # Retrieve this functions (local) cache from the global cache.
        result_cache = result_cache[f]
        hit_counter = hit_counter[f]

        # Construct the cache key from `args` and `kwargs`.

        # there could be nested memoize functions, so need to pass around
        # _Cache objects to bind the inner function -- explicitly ignore
        # such objects from the key.
        key_args = [x for x in args if not isinstance(x, _Cache)]
        key_kwargs = [
            (k, v) for (k, v) in sorted(kwargs.items()) if not isinstance(v, _Cache)
        ]
        key = (tuple(key_args), tuple(key_kwargs))

        key = _ensure_hashable(key)

        if key in result_cache:
            assert _expect_cache_hit in [True, None]
            hit_counter[key] += 1
            result = result_cache[key]
        else:
            assert _expect_cache_hit in [False, None]
            result = f(*args, **kwargs)
            result_cache[key] = result
        return result

    # for inspection/testing:
    wrapper.get_cache_hits = lambda cache, key: cache.hits[f][key]
    wrapper.get_total_cache_hits = lambda cache: sum(cache.hits[f].values())
    return wrapper


def bind(f, cache: Cache, expect_cache_hit=None) -> Callable:
    """Bind the cache `cache` to the function `f` (see module docstring).

    Args:
      f: A function which has been decorated with `tf_memoize.memoize`.
      cache: A memoize cache created by `tf_memoize.create_memoize_cache`, or
        `None` if no caching is desired.
      expect_cache_hit: To help debugging; None => no expectation.

    Returns:
      The memoized function, using `cache` as the memoize cache (or the
      non-memoized function if `cache` is None).
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(
            *args,
            _private_cache_kwarg=cache,
            _expect_cache_hit=expect_cache_hit,
            **kwargs,
        )

    return wrapper
