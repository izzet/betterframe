"""BetterFrame: write a dataframe pipeline once, run it on Dask or pandas.

Not a DataFrame implementation. It adapts the *execution semantics* that differ
between engines -- per-partition application, meta schemas, aggregation
keywords, persistence -- so a pipeline that must work on both does not fill up
with ``if is_dask:`` branches.

Public API:
- :func:`ops_for` -- operations object for whichever engine a frame belongs to
- :class:`DataFrameOps` -- the interface, and :class:`DaskOps` /
  :class:`PandasOps` to subclass for engine-specific behaviour of your own
- :func:`is_dask_frame` -- engine test that does not import Dask needlessly

Example::

    from betterframe import ops_for

    ops = ops_for(frame)
    frame = ops.apply(frame, normalise, meta=lambda: build_meta(ops.meta_source(frame)))
    result = frame.groupby("key").agg(aggs, **ops.agg_kwargs(frame))
    return ops.finalize(result)
"""

from __future__ import annotations

from .ops import DaskOps, DataFrameOps, MetaSource, PandasOps, is_dask_frame, ops_for

__all__ = [
    "DaskOps",
    "DataFrameOps",
    "MetaSource",
    "PandasOps",
    "is_dask_frame",
    "ops_for",
]

__version__ = "0.1.0"
