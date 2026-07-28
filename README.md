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
from betterframe import BetterFrame


def compute(records):
    frame = BetterFrame(records).mutable()  # records may be Dask or pandas
    frame = frame.apply(normalise, meta=lambda: build_meta(frame.meta_source()))
    result = frame.pipe(
        lambda df, kw=frame.agg_kwargs(): df.groupby("key").agg({"scaled": "sum"}, **kw)
    )
    return result.finalize()
```

`BetterFrame` binds a frame to the operations its engine needs, so the frame
stops being an argument to every call. Methods that produce a frame return a
`BetterFrame`, so a pipeline chains; `finalize()` ends the chain and hands back
a native frame, and `native` reaches the underlying one at any point.

It is deliberately thin — it does **not** proxy the dataframe API. Real work
still happens on the frame itself, through `pipe()` or `native`. Wrapping exists
to answer engine questions, not to replace pandas.

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

## Set aggregations

Neither engine ships one — pandas has no set aggregation at all, and Dask needs
a three-stage `Aggregation` whose chunk and combine steps have to agree. Both
are provided, and both return `frozenset`, so a value aggregated one way
compares equal to the same value aggregated the other:

```python
bf = BetterFrame(records)
grouped = records.groupby("key").agg(
    {
        "name": bf.ops.set_union(),  # distinct values -> frozenset
        "tags": bf.ops.set_union_flatten(),
    },  # union of set-valued cells
    **bf.agg_kwargs(),
)
```

Flattening is built on [betterset](https://github.com/izzet/betterset), which
handles what a naive `set().union(*values)` gets wrong:

| input | `set().union(*values)` | `set_union_flatten` |
|---|---|---|
| `["abc"]` | `{"a", "b", "c"}` — string shredded | `{"abc"}` |
| `[42, {"d"}]` | `TypeError` | `{42, "d"}` |
| `[None, {"d"}]` | `TypeError` | `{"d"}` |

The string case is the one that bites: it silently turns a set of names into a
set of letters, and nothing raises.

### Dask stringifies set-valued columns unless you stop it

Dask's `dataframe.convert-string` is **on by default**, and it rewrites object
columns to its own string dtype when a frame is constructed. A column holding
sets is turned into a column holding their *reprs*, before any aggregation runs:

```python
frame = pd.DataFrame({"g": ["x", "x"], "tags": ["abc", frozenset({"d"})]})

dd.from_pandas(frame, npartitions=1)  # tags dtype -> string
# set_union_flatten now yields {"abc", "frozenset({'d'})"}

with dask.config.set({"dataframe.convert-string": False}):
    dd.from_pandas(frame, npartitions=1)  # tags dtype -> object
# set_union_flatten yields {"abc", "d"}
```

No library can recover the values once that has happened -- the conversion
occurs at construction, upstream of anything BetterFrame sees. If you hold sets
in a column, turn the conversion off.

## Deciding when to defer

Dask earns its overhead on data that does not fit in memory and charges it
regardless. Choosing between the engines needs a size, and the obvious ways to
get one defeat the purpose -- on a lazy frame they execute the very work you
were trying to avoid:

| | partitions executed |
|---|---|
| `len(frame)` | 4 |
| `frame.memory_usage_per_partition().sum().compute()` | 4 |
| `BetterFrame(frame).nbytes()` | **0** |

`nbytes()` returns what the scheduler already knows and `None` otherwise. It
never computes and never waits -- and since `persist()` is asynchronous, `None`
is the common answer, not an edge case.

So `materialize_if_under` lets the caller supply a bound it can justify:

```python
frame = BetterFrame(records).materialize_if_under(
    256 * 1024**2,
    fallback_bound=records.npartitions
    * PARTITION_SIZE_BYTES,  # your partitioning policy
)
```

It returns a pandas-backed `BetterFrame` when it materialises and the original
otherwise, so callers need no branch. With neither a known size nor a bound, the
frame is left alone rather than guessed at.

## Extending

Subclass `DaskOps` / `PandasOps` for engine-specific behaviour of your own —
custom aggregations, say — and hand the subclasses to `BetterFrame`:

```python
from betterframe import BetterFrame, DaskOps, PandasOps


class MyDaskOps(DaskOps):
    def set_union(self):
        return some_dask_aggregation()


class MyPandasOps(PandasOps):
    def set_union(self):
        return some_pandas_callable


frame = BetterFrame(records, dask_ops=MyDaskOps, pandas_ops=MyPandasOps)
frame.ops.set_union()  # `ops` reaches engine questions that take no frame
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
