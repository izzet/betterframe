import pandas as pd
import pytest

from betterframe import DaskOps, PandasOps, is_dask_frame, ops_for

dask = pytest.importorskip("dask", reason="dask is an optional dependency")
dd = pytest.importorskip("dask.dataframe", reason="dask is an optional dependency")


@pytest.fixture(autouse=True)
def _no_string_conversion():
    """Dask rewrites object columns to its own string dtype when a frame is
    constructed. That is a property of frame construction, not of the
    operations under test, and comparing dtypes across engines is meaningless
    while it is on."""
    with dask.config.set({"dataframe.convert-string": False}):
        yield


@pytest.fixture
def frame():
    return pd.DataFrame({"g": ["x", "x", "y"], "v": [1.0, 2.0, 3.0]})


@pytest.fixture
def dask_frame(frame):
    return dd.from_pandas(frame, npartitions=2)


def add_column(df, value=1):
    out = df.copy()
    out["extra"] = value
    return out


def widen_but_bail_when_empty(df):
    """Mirrors the real-world helpers this package exists for: it short-circuits
    on an empty frame and returns a *different* schema than it would otherwise."""
    if df.empty:
        return df
    out = df.copy()
    out["mean_v"] = out["v"] / 2.0
    return out.drop(columns=["v"])


# --- engine detection --------------------------------------------------------


def test_is_dask_frame_distinguishes_engines(frame, dask_frame):
    assert is_dask_frame(dask_frame)
    assert not is_dask_frame(frame)


def test_is_dask_frame_tolerates_non_frames():
    assert not is_dask_frame(None)
    assert not is_dask_frame([1, 2, 3])
    assert not is_dask_frame("frame")


def test_ops_for_picks_the_right_implementation(frame, dask_frame):
    assert isinstance(ops_for(frame), PandasOps)
    assert isinstance(ops_for(dask_frame), DaskOps)
    assert ops_for(dask_frame).is_dask
    assert not ops_for(frame).is_dask


def test_ops_for_accepts_subclasses(frame, dask_frame):
    class MyPandas(PandasOps):
        pass

    class MyDask(DaskOps):
        pass

    assert isinstance(ops_for(frame, pandas_ops=MyPandas), MyPandas)
    assert isinstance(ops_for(dask_frame, dask_ops=MyDask), MyDask)


# --- apply -------------------------------------------------------------------


def test_apply_matches_across_engines(frame, dask_frame):
    on_pandas = ops_for(frame).apply(frame, add_column, value=7)
    on_dask = ops_for(dask_frame).apply(dask_frame, add_column, value=7).compute()
    pd.testing.assert_frame_equal(on_pandas, on_dask)


def test_apply_passes_positional_and_keyword_arguments(frame):
    ops = ops_for(frame)
    out = ops.apply(frame, lambda df, a, b=0: df.assign(total=a + b), 10, b=5)
    assert out["total"].tolist() == [15, 15, 15]


# --- meta semantics: the reason this package exists --------------------------


def test_empty_frame_gets_the_populated_schema_on_both_engines(frame, dask_frame):
    """A helper that bails on empty input must not leave the two engines with
    different columns. Dask repairs it via meta; pandas has to be told to."""
    empty = frame.iloc[:0]
    empty_dask = dd.from_pandas(empty, npartitions=1)

    on_pandas = ops_for(empty).apply(empty, widen_but_bail_when_empty)
    on_dask = ops_for(empty_dask).apply(empty_dask, widen_but_bail_when_empty).compute()

    assert list(on_pandas.columns) == list(on_dask.columns)
    assert "mean_v" in on_pandas.columns, (
        "schema should be the one the helper produces when populated"
    )
    assert "v" not in on_pandas.columns


def test_empty_frame_dtypes_match_across_engines(frame, dask_frame):
    empty = frame.iloc[:0]
    empty_dask = dd.from_pandas(empty, npartitions=1)

    on_pandas = ops_for(empty).apply(empty, widen_but_bail_when_empty)
    on_dask = ops_for(empty_dask).apply(empty_dask, widen_but_bail_when_empty).compute()

    assert on_pandas.dtypes.to_dict() == on_dask.dtypes.to_dict()


def test_explicit_meta_is_used_for_empty_pandas_frames(frame):
    empty = frame.iloc[:0]
    meta = pd.DataFrame({"only": pd.Series(dtype="int64")})

    out = ops_for(empty).apply(empty, widen_but_bail_when_empty, meta=lambda: meta)

    assert list(out.columns) == ["only"]
    assert out.dtypes["only"] == "int64"


def test_meta_thunk_is_not_called_on_the_pandas_path_when_rows_exist(frame):
    calls = []

    def meta():
        calls.append(1)
        return pd.DataFrame()

    ops_for(frame).apply(frame, add_column, meta=meta)
    assert calls == [], "meta must not be built when the result is non-empty"


def test_meta_thunk_is_called_on_the_dask_path(dask_frame):
    calls = []

    def meta():
        calls.append(1)
        return add_column(dask_frame._meta)

    ops_for(dask_frame).apply(dask_frame, add_column, meta=meta)
    assert calls, "dask needs the meta up front"


def test_apply_leaves_the_frame_alone_when_no_schema_can_be_inferred(frame):
    """A helper that cannot run on a synthetic frame must not have a schema
    guessed for it."""
    empty = frame.iloc[:0]

    def unrunnable(df):
        if df.empty:
            return df
        raise RuntimeError("cannot run on populated frames")

    out = ops_for(empty).apply(empty, unrunnable)
    assert list(out.columns) == list(empty.columns)


# --- index names -------------------------------------------------------------


def test_index_names_match_across_engines(frame, dask_frame):
    """A single-level index, which is what `from_pandas` supports."""
    indexed = frame.set_index("g")
    indexed_dask = dask_frame.set_index("g")

    assert list(ops_for(indexed).index_names(indexed)) == ["g"]
    assert list(ops_for(indexed_dask).index_names(indexed_dask)) == ["g"]


def test_index_names_reads_a_multiindex(frame, dask_frame):
    """Dask cannot build a MultiIndex frame directly, but a groupby produces
    one -- which is how these frames arise in practice."""
    grouped = frame.groupby(["g", "v"]).sum()
    grouped_dask = dask_frame.groupby(["g", "v"]).sum()

    assert list(ops_for(grouped).index_names(grouped)) == ["g", "v"]
    assert list(ops_for(grouped_dask).index_names(grouped_dask)) == ["g", "v"]


# --- aggregation keywords ----------------------------------------------------


def test_agg_kwargs_are_engine_appropriate(frame, dask_frame):
    assert ops_for(frame).agg_kwargs(frame) == {}
    assert ops_for(dask_frame).agg_kwargs(dask_frame) == {
        "split_out": dask_frame.npartitions
    }


def test_agg_kwargs_can_be_splatted_into_groupby(frame, dask_frame):
    for target in (frame, dask_frame):
        ops = ops_for(target)
        out = target.groupby("g").agg({"v": "sum"}, **ops.agg_kwargs(target))
        out = out.compute() if ops.is_dask else out
        assert out.loc["x", "v"] == 3.0
        assert out.loc["y", "v"] == 3.0


# --- finalize and mutability -------------------------------------------------


def test_finalize_is_a_noop_for_pandas(frame):
    assert ops_for(frame).finalize(frame) is frame


def test_finalize_persists_a_dask_frame(dask_frame):
    out = ops_for(dask_frame).finalize(dask_frame)
    pd.testing.assert_frame_equal(out.compute(), dask_frame.compute())


def test_mutable_protects_the_callers_pandas_frame(frame):
    before = frame.copy()
    working = ops_for(frame).mutable(frame)
    working["injected"] = 1
    pd.testing.assert_frame_equal(frame, before)


def test_mutable_is_a_passthrough_for_dask(dask_frame):
    assert ops_for(dask_frame).mutable(dask_frame) is dask_frame


# --- meta_source -------------------------------------------------------------


def test_meta_source_exposes_the_dask_shape_for_pandas_frames(frame):
    source = ops_for(frame).meta_source(frame)
    assert list(source.columns) == list(frame.columns)
    assert source._meta.empty
    assert source._meta.dtypes.to_dict() == frame.dtypes.to_dict()


def test_meta_source_returns_a_dask_frame_unchanged(dask_frame):
    assert ops_for(dask_frame).meta_source(dask_frame) is dask_frame


def test_meta_source_lets_one_meta_builder_serve_both_engines(frame, dask_frame):
    """A builder written against Dask should work unchanged on pandas."""

    def build_meta(source):
        return pd.DataFrame(
            {c: pd.Series(dtype=source._meta[c].dtype) for c in source.columns}
        )

    from_pandas = build_meta(ops_for(frame).meta_source(frame))
    from_dask = build_meta(ops_for(dask_frame).meta_source(dask_frame))
    pd.testing.assert_frame_equal(from_pandas, from_dask)


# --- the interface itself ----------------------------------------------------


def test_base_class_refuses_to_guess():
    from betterframe import DataFrameOps

    ops = DataFrameOps()
    for call in (
        lambda: ops.apply(None, None),
        lambda: ops.index_names(None),
        lambda: ops.agg_kwargs(None),
        lambda: ops.finalize(None),
        lambda: ops.mutable(None),
        lambda: ops.meta_source(None),
    ):
        with pytest.raises(NotImplementedError):
            call()
