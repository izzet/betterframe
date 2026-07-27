# BetterFrame

Write a dataframe pipeline once, run it on Dask or pandas.

Not a DataFrame implementation. BetterFrame adapts the *execution semantics*
that differ between the two engines — per-partition application, meta schemas,
aggregation keywords, persistence — so a pipeline that has to work on both does
not fill up with `if is_dask:` branches.

## Why

Dask is the right tool when data outgrows memory, and a needless tax when it
does not. A pipeline that wants both ends up either duplicated or littered with
engine checks. The differences are few but awkward:

| | Dask | pandas |
|---|---|---|
| apply a function per partition | `df.map_partitions(fn, meta=…)` | `fn(df)` |
| index level names | `df.index._meta.names` | `df.index.names` |
| `groupby().agg()` extras | `split_out=…` | — |
| make the result concrete | `df.persist()` | already is |
| column assignment | builds a graph | mutates in place |

## Install

```bash
pip install betterframe          # pandas only
pip install "betterframe[dask]"  # to drive Dask frames as well
```

## Use

```python
from betterframe import ops_for


def compute(frame):
    ops = ops_for(frame)  # frame may be Dask or pandas
    frame = ops.mutable(frame)
    frame["scaled"] = frame["value"] * 2
    frame = ops.apply(frame, normalise, meta=lambda: build_meta(ops.meta_source(frame)))
    result = frame.groupby("key").agg({"scaled": "sum"}, **ops.agg_kwargs(frame))
    return ops.finalize(result)
```

The same function now runs on either engine, and there is exactly one place —
this package — that knows the difference.

## The part worth knowing about

Most of the surface is mechanical. The subtle part is Dask's `meta` handling,
because getting it wrong produces frames that differ **only in dtype, only for
empty inputs, and only sometimes**:

* Dask coerces *every* partition to a meta schema. A helper that short-circuits
  on an empty frame and returns it unchanged still comes out with the populated
  schema, because Dask repairs it afterwards.
* Where no explicit meta is given, Dask *infers* one by running the function
  against a synthetic non-empty frame. So even without a meta, an empty
  partition ends up with the schema the function would have produced had there
  been rows.

pandas does neither. `PandasOps.apply` reproduces both, so an empty frame is
not silently a different dtype depending on which engine produced it. This is
the failure mode BetterFrame exists to prevent: it is invisible in every test
whose fixtures happen to be fully populated.

## Extending

Subclass `DaskOps` / `PandasOps` for engine-specific behaviour of your own —
custom aggregations, say — and hand the subclasses to `ops_for`:

```python
from betterframe import DaskOps, PandasOps, ops_for


class MyDaskOps(DaskOps):
    def set_union(self):
        return some_dask_aggregation()


class MyPandasOps(PandasOps):
    def set_union(self):
        return some_pandas_callable


ops = ops_for(frame, dask_ops=MyDaskOps, pandas_ops=MyPandasOps)
```

Domain-specific aggregations are deliberately left out of the core so the
package stays about engine semantics.

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## License

MIT
