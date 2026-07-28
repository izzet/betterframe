"""Set aggregations that behave the same on Dask and pandas.

Neither engine ships one. pandas has no set aggregation at all, and Dask needs
a three-stage :class:`dask.dataframe.Aggregation` whose chunk and combine steps
have to agree -- easy to get subtly wrong, and the kind of engine knowledge
this package exists to hold.

Both are built on :class:`betterset.BetterSet`, whose ``flatten`` handles the
cases a naive ``set().union(*values)`` gets wrong:

* strings stay atomic -- ``union(*["abc"])`` would otherwise yield
  ``{"a", "b", "c"}``, quietly turning a set of names into a set of letters
* scalars mixed in with collections do not raise
* ``None`` is skipped rather than poisoning the result

Both engines return :class:`frozenset`, so a value aggregated one way compares
equal to the same value aggregated the other.

.. warning::

   Dask's ``dataframe.convert-string`` is on by default and rewrites object
   columns to its string dtype at construction, turning a column of sets into a
   column of their reprs before any aggregation runs. Set
   ``dask.config.set({"dataframe.convert-string": False})`` when a column holds
   sets; nothing downstream can recover the values otherwise.
"""

from __future__ import annotations

from typing import Any

from betterset import BetterSet

__all__ = [
    "dask_set_union",
    "dask_set_union_flatten",
    "pandas_set_union",
    "pandas_set_union_flatten",
]

# `groupby().agg()` names the output column after the aggregation, so both
# engines have to agree on this for their results to line up.
AGGREGATION_NAME = "unique"


def pandas_set_union(series: Any) -> frozenset:
    """Distinct non-null values of a series, as a frozenset."""
    return frozenset(BetterSet(series.dropna().unique().tolist()))


def pandas_set_union_flatten(series: Any) -> frozenset:
    """Union of the set-valued entries of a series, as a frozenset."""
    return frozenset(BetterSet.flatten(series.dropna()))


pandas_set_union.__name__ = AGGREGATION_NAME
pandas_set_union_flatten.__name__ = AGGREGATION_NAME


def dask_set_union() -> Any:
    """Dask aggregation collecting distinct values into a frozenset."""
    import dask.dataframe as dd

    return dd.Aggregation(
        AGGREGATION_NAME,
        lambda s: s.apply(lambda x: BetterSet(x.dropna().unique().tolist())),
        lambda s0: s0.apply(BetterSet.flatten),
        lambda s1: s1.apply(frozenset),
    )


def dask_set_union_flatten() -> Any:
    """Dask aggregation unioning set-valued entries into a frozenset."""
    import dask.dataframe as dd

    return dd.Aggregation(
        AGGREGATION_NAME,
        lambda s: s.apply(BetterSet.flatten),
        lambda s0: s0.agg(BetterSet.flatten),
        lambda s1: s1.apply(frozenset),
    )
