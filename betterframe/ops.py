"""Engine differences between Dask and pandas frames, isolated in one place.

An aggregation pipeline that must run on either engine otherwise fills up with
``if is_dask:`` branches. These classes hold the differences so the pipeline can
be written once. Most callers do not use them directly -- they go through
:class:`~betterframe.BetterFrame`, which binds a frame to the right one -- but
they are the extension point for adding engine-specific behaviour of your own.

Most of the surface is mechanical -- ``map_partitions(fn)`` versus ``fn(df)``,
``persist()`` versus nothing. The part worth reading is how the pandas side
emulates Dask's ``meta`` semantics, because getting it wrong produces frames
that differ only in dtype, only for empty inputs, and only sometimes:

* Dask coerces **every** partition to a meta schema. A function that
  short-circuits on an empty frame and returns it unchanged therefore still
  comes out with the populated schema, because Dask repairs it afterwards.
* Where no explicit meta is given, Dask *infers* one by running the function
  against a synthetic non-empty frame. So even without a meta, an empty
  partition ends up with the schema the function would have produced had there
  been rows.

pandas does neither. :class:`PandasOps` reproduces both, so a frame that is
empty on one engine is not silently a different dtype on the other.

This is not a DataFrame implementation. It adapts execution semantics for
frames that already exist.
"""

from __future__ import annotations

from typing import Any, Callable

from .aggregations import (
    dask_set_union,
    dask_set_union_flatten,
    pandas_set_union,
    pandas_set_union_flatten,
)

__all__ = ["DaskOps", "DataFrameOps", "MetaSource", "PandasOps", "is_dask_frame"]


def is_dask_frame(df: Any) -> bool:
    """True if ``df`` is a Dask DataFrame, without importing Dask unnecessarily.

    Dask is an optional dependency: a caller working purely in pandas should not
    pay an import for a question whose answer is no.
    """
    module = type(df).__module__
    if not module.startswith("dask"):
        return False
    try:
        import dask.dataframe as dd
    except ImportError:  # pragma: no cover - only when dask is half-installed
        return False
    return isinstance(df, dd.DataFrame)


class DataFrameOps:
    """Dispatch table for the operations that differ between engines.

    Subclass :class:`DaskOps` or :class:`PandasOps` to add engine-specific
    behaviour of your own -- custom aggregations, for instance -- and pass the
    subclasses to :class:`~betterframe.BetterFrame`.
    """

    #: Whether this implementation drives a Dask frame.
    is_dask: bool

    def apply(
        self,
        df: Any,
        fn: Callable,
        *args: Any,
        meta: Callable | None = None,
        **kwargs: Any,
    ) -> Any:
        """Apply a per-partition function.

        ``meta`` is a thunk rather than a value so Dask-only meta builders are
        never invoked on a pandas frame.
        """
        raise NotImplementedError

    def index_names(self, df: Any) -> Any:
        """Names of the frame's index levels."""
        raise NotImplementedError

    def agg_kwargs(self, df: Any) -> dict[str, Any]:
        """Extra keyword arguments for ``groupby().agg()`` on this engine."""
        raise NotImplementedError

    def set_union(self) -> Any:
        """Aggregation collecting a column's distinct values into a frozenset."""
        raise NotImplementedError

    def set_union_flatten(self) -> Any:
        """Aggregation unioning a column's set-valued entries into a frozenset."""
        raise NotImplementedError

    def nbytes(self, df: Any) -> int | None:
        """Materialised size in bytes, or ``None`` if it cannot be known cheaply.

        Never triggers a computation, and never waits for one. A caller sizing
        a frame to decide whether to bring it into memory must not pay for the
        answer -- that would impose a cost on exactly the large inputs the
        question exists to protect.
        """
        raise NotImplementedError

    def materialize(self, df: Any) -> Any:
        """Bring the frame into memory as pandas."""
        raise NotImplementedError

    def finalize(self, df: Any) -> Any:
        """Make the result concrete, if the engine has such a notion."""
        raise NotImplementedError

    def mutable(self, df: Any) -> Any:
        """A frame whose columns may be assigned without affecting the caller."""
        raise NotImplementedError

    def meta_source(self, df: Any) -> Any:
        """Something a Dask meta builder can read ``.columns`` and ``._meta`` from.

        Lets meta builders written against Dask be reused unchanged on pandas
        frames, rather than duplicated per engine.
        """
        raise NotImplementedError


class DaskOps(DataFrameOps):
    """Operations against a Dask DataFrame."""

    is_dask = True

    def apply(self, df, fn, *args, meta=None, **kwargs):
        if meta is not None:
            return df.map_partitions(fn, *args, meta=meta(), **kwargs)
        return df.map_partitions(fn, *args, **kwargs)

    def index_names(self, df):
        return df.index._meta.names

    def agg_kwargs(self, df):
        # one output partition per input partition
        return {"split_out": df.npartitions}

    def set_union(self):
        return dask_set_union()

    def set_union_flatten(self):
        return dask_set_union_flatten()

    def nbytes(self, df):
        """Size the scheduler already knows, or None.

        ``persist()`` is asynchronous, so partitions that have not finished are
        simply absent from the scheduler's map. That is reported as unknown
        rather than guessed at, and never waited for.
        """
        try:
            from distributed import futures_of

            futures = futures_of(df)
            if not futures:
                return None
            known = futures[0].client.nbytes(summary=False) or {}
            sizes = [known.get(f.key) for f in futures]
            if any(size is None for size in sizes):
                return None
            return int(sum(sizes))
        except Exception:  # noqa: BLE001 - any failure means the size is simply
            # not knowable cheaply, which is what None communicates
            return None

    def materialize(self, df):
        return df.compute()

    def finalize(self, df):
        return df.persist()

    def mutable(self, df):
        # assignment on a Dask frame builds a new graph; nothing is aliased
        return df

    def meta_source(self, df):
        return df


class PandasOps(DataFrameOps):
    """Operations against a pandas DataFrame, emulating Dask's meta semantics."""

    is_dask = False

    def apply(self, df, fn, *args, meta=None, **kwargs):
        out = fn(df, *args, **kwargs)
        if not getattr(out, "empty", False):
            return out
        schema = (
            meta() if meta is not None else self._infer_schema(df, fn, args, kwargs)
        )
        if schema is None:
            return out
        out = out.reindex(columns=schema.columns)
        return out.astype(
            {c: dt for c, dt in schema.dtypes.items() if c in out.columns}
        )

    @staticmethod
    def _infer_schema(df, fn, args, kwargs):
        """What Dask would have inferred: run ``fn`` on a synthetic non-empty frame."""
        try:
            from dask.dataframe.utils import meta_nonempty
        except ImportError:
            return None
        try:
            return fn(meta_nonempty(df.iloc[:0]), *args, **kwargs).iloc[:0]
        except Exception:  # noqa: BLE001 - any failure means we cannot know the
            # schema, and leaving the frame alone is safer than guessing at one
            return None

    def index_names(self, df):
        return df.index.names

    def agg_kwargs(self, df):
        # `split_out` is a Dask partitioning concern with no pandas equivalent
        return {}

    def set_union(self):
        return pandas_set_union

    def set_union_flatten(self):
        return pandas_set_union_flatten

    def nbytes(self, df):
        # already in memory, so measuring it computes nothing
        return int(df.memory_usage(deep=True).sum())

    def materialize(self, df):
        return df

    def finalize(self, df):
        return df

    def mutable(self, df):
        # column assignment must not reach back into the caller's frame
        return df.copy(deep=False)

    def meta_source(self, df):
        return MetaSource(df)


class MetaSource:
    """Adapts a pandas frame to the ``.columns`` / ``._meta`` shape Dask meta
    builders expect, so one implementation of those builders serves both
    engines rather than being duplicated per engine."""

    __slots__ = ("_meta", "columns")

    def __init__(self, df: Any) -> None:
        self.columns = df.columns
        self._meta = df.iloc[:0]
