"""Engine differences between Dask and pandas frames, isolated in one place.

An aggregation pipeline that must run on either engine otherwise fills up with
``if is_dask:`` branches. This lets the pipeline be written once: ask
:func:`ops_for` for the right operations object and work through it, so only
one small module knows which engine is in play.

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

__all__ = ["DaskOps", "DataFrameOps", "PandasOps", "is_dask_frame", "ops_for"]


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
    subclasses to :func:`ops_for`.
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


def ops_for(
    df: Any,
    *,
    dask_ops: type[DataFrameOps] = DaskOps,
    pandas_ops: type[DataFrameOps] = PandasOps,
) -> DataFrameOps:
    """Operations object for whichever engine ``df`` belongs to.

    Pass ``dask_ops`` / ``pandas_ops`` to substitute subclasses carrying your
    own engine-specific behaviour.
    """
    return dask_ops() if is_dask_frame(df) else pandas_ops()
