"""``BetterFrame`` -- a frame bound to the operations its engine needs.

Every operation that differs between Dask and pandas takes the frame as its
first argument, so binding the frame removes that argument from every call and
lets a pipeline chain instead of repeating itself.

The wrapper is deliberately thin. It does not proxy the dataframe API: real
work still happens on the frame itself, reached through :attr:`native` or
:meth:`pipe`. Wrapping exists to answer engine questions, not to replace pandas.
"""

from __future__ import annotations

from typing import Any, Callable

from .ops import DaskOps, DataFrameOps, PandasOps, is_dask_frame

__all__ = ["BetterFrame"]


class BetterFrame:
    """A dataframe together with the operations its engine requires.

    Args:
        frame: A Dask or pandas DataFrame.
        dask_ops: Operations class to use for Dask frames. Subclass
            :class:`~betterframe.DaskOps` to add behaviour of your own.
        pandas_ops: Operations class to use for pandas frames.

    Methods that produce a frame return a new ``BetterFrame``, so a pipeline
    chains. :meth:`finalize` ends the chain and hands back a native frame.
    """

    __slots__ = ("_frame", "_ops")

    def __init__(
        self,
        frame: Any,
        *,
        dask_ops: type[DataFrameOps] = DaskOps,
        pandas_ops: type[DataFrameOps] = PandasOps,
    ) -> None:
        self._frame = frame
        self._ops = dask_ops() if is_dask_frame(frame) else pandas_ops()

    # --- access ---------------------------------------------------------

    @property
    def native(self) -> Any:
        """The underlying dataframe, for operations that need no adaptation."""
        return self._frame

    @property
    def ops(self) -> DataFrameOps:
        """The operations object, for engine questions that take no frame."""
        return self._ops

    @property
    def is_dask(self) -> bool:
        return self._ops.is_dask

    def _wrap(self, frame: Any) -> BetterFrame:
        new = object.__new__(BetterFrame)
        new._frame = frame
        new._ops = self._ops
        return new

    # --- operations -----------------------------------------------------

    def apply(
        self, fn: Callable, *args: Any, meta: Callable | None = None, **kwargs: Any
    ) -> BetterFrame:
        """Apply a per-partition function.

        ``meta`` is a thunk rather than a value so Dask-only meta builders are
        never invoked on a pandas frame.
        """
        return self._wrap(self._ops.apply(self._frame, fn, *args, meta=meta, **kwargs))

    def pipe(self, fn: Callable, *args: Any, **kwargs: Any) -> BetterFrame:
        """Run a function against the whole native frame and stay wrapped.

        For operations that need no adaptation -- ``groupby``, ``rename`` and
        the like -- where dropping to :attr:`native` and re-wrapping would only
        add noise.
        """
        return self._wrap(fn(self._frame, *args, **kwargs))

    def mutable(self) -> BetterFrame:
        """A frame whose columns may be assigned without affecting the caller."""
        return self._wrap(self._ops.mutable(self._frame))

    def index_names(self) -> Any:
        """Names of the frame's index levels."""
        return self._ops.index_names(self._frame)

    def agg_kwargs(self) -> dict[str, Any]:
        """Extra keyword arguments for ``groupby().agg()`` on this engine."""
        return self._ops.agg_kwargs(self._frame)

    def meta_source(self) -> Any:
        """Something a Dask meta builder can read ``.columns`` and ``._meta`` from."""
        return self._ops.meta_source(self._frame)

    def nbytes(self) -> int | None:
        """Materialised size in bytes, or ``None`` if it cannot be known cheaply.

        Never triggers a computation and never waits for one. On Dask that means
        the answer is only available once ``persist()`` has actually finished;
        until then it is ``None``, because waiting would impose a cost on
        exactly the large frames this question exists to protect.
        """
        return self._ops.nbytes(self._frame)

    def materialize_if_under(
        self, max_bytes: int, *, fallback_bound: int | None = None
    ) -> BetterFrame:
        """Bring the frame into memory when it is provably smaller than a limit.

        Uses :meth:`nbytes` when the size is already known. When it is not --
        the common case, since ``persist()`` is asynchronous -- ``fallback_bound``
        is used instead: an upper bound the *caller* can justify, typically from
        its own partitioning policy (``npartitions * partition_size``). Without
        one, an unknown size means the frame is left alone rather than guessed at.

        Returns a pandas-backed ``BetterFrame`` when it materialises, and self
        otherwise, so callers need no branch of their own.
        """
        if not self.is_dask:
            return self
        size = self.nbytes()
        if size is None:
            size = fallback_bound
        if size is None or size > max_bytes:
            return self
        return BetterFrame(self._ops.materialize(self._frame))

    def finalize(self) -> Any:
        """Make the result concrete and return the native frame."""
        return self._ops.finalize(self._frame)

    # --- niceties -------------------------------------------------------

    def __repr__(self) -> str:
        engine = "dask" if self.is_dask else "pandas"
        return f"BetterFrame({engine}, {type(self._frame).__name__})"
