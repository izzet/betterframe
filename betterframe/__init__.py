"""BetterFrame: write a dataframe pipeline once, run it on Dask or pandas.

Not a DataFrame implementation. It adapts the *execution semantics* that differ
between engines -- per-partition application, meta schemas, aggregation
keywords, persistence -- so a pipeline that must work on both does not fill up
with ``if is_dask:`` branches.

Public API:
- :class:`BetterFrame` -- a frame bound to the operations its engine needs
- :class:`DaskOps` / :class:`PandasOps` -- subclass to add engine-specific
  behaviour of your own, and pass to :class:`BetterFrame`
- :func:`is_dask_frame` -- engine test that does not import Dask needlessly

Example::

    from betterframe import BetterFrame

    frame = BetterFrame(records).mutable()
    frame = frame.apply(normalise, meta=lambda: build_meta(frame.meta_source()))
    result = frame.pipe(lambda df: df.groupby("key").agg(aggs, **frame.agg_kwargs()))
    return result.finalize()
"""

from __future__ import annotations

from .frame import BetterFrame
from .ops import DaskOps, DataFrameOps, MetaSource, PandasOps, is_dask_frame

__all__ = [
    "BetterFrame",
    "DaskOps",
    "DataFrameOps",
    "MetaSource",
    "PandasOps",
    "is_dask_frame",
]

__version__ = "0.1.0"
