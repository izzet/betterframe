import pandas as pd
import pytest

from betterframe import BetterFrame, DaskOps, PandasOps, is_dask_frame

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


def test_betterframe_picks_the_right_implementation(frame, dask_frame):
    assert isinstance(BetterFrame(frame).ops, PandasOps)
    assert isinstance(BetterFrame(dask_frame).ops, DaskOps)
    assert BetterFrame(dask_frame).is_dask
    assert not BetterFrame(frame).is_dask


def test_betterframe_accepts_ops_subclasses(frame, dask_frame):
    class MyPandas(PandasOps):
        pass

    class MyDask(DaskOps):
        pass

    assert isinstance(BetterFrame(frame, pandas_ops=MyPandas).ops, MyPandas)
    assert isinstance(BetterFrame(dask_frame, dask_ops=MyDask).ops, MyDask)


# --- apply -------------------------------------------------------------------


def test_apply_matches_across_engines(frame, dask_frame):
    on_pandas = BetterFrame(frame).apply(add_column, value=7).native
    on_dask = BetterFrame(dask_frame).apply(add_column, value=7).native.compute()
    pd.testing.assert_frame_equal(on_pandas, on_dask)


def test_apply_passes_positional_and_keyword_arguments(frame):
    out = (
        BetterFrame(frame)
        .apply(lambda df, a, b=0: df.assign(total=a + b), 10, b=5)
        .native
    )
    assert out["total"].tolist() == [15, 15, 15]


# --- meta semantics: the reason this package exists --------------------------


def test_empty_frame_gets_the_populated_schema_on_both_engines(frame, dask_frame):
    """A helper that bails on empty input must not leave the two engines with
    different columns. Dask repairs it via meta; pandas has to be told to."""
    empty = frame.iloc[:0]
    empty_dask = dd.from_pandas(empty, npartitions=1)

    on_pandas = BetterFrame(empty).apply(widen_but_bail_when_empty).native
    on_dask = BetterFrame(empty_dask).apply(widen_but_bail_when_empty).native.compute()

    assert list(on_pandas.columns) == list(on_dask.columns)
    assert "mean_v" in on_pandas.columns, (
        "schema should be the one the helper produces when populated"
    )
    assert "v" not in on_pandas.columns


def test_empty_frame_dtypes_match_across_engines(frame, dask_frame):
    empty = frame.iloc[:0]
    empty_dask = dd.from_pandas(empty, npartitions=1)

    on_pandas = BetterFrame(empty).apply(widen_but_bail_when_empty).native
    on_dask = BetterFrame(empty_dask).apply(widen_but_bail_when_empty).native.compute()

    assert on_pandas.dtypes.to_dict() == on_dask.dtypes.to_dict()


def test_explicit_meta_is_used_for_empty_pandas_frames(frame):
    empty = frame.iloc[:0]
    meta = pd.DataFrame({"only": pd.Series(dtype="int64")})

    out = BetterFrame(empty).apply(widen_but_bail_when_empty, meta=lambda: meta).native

    assert list(out.columns) == ["only"]
    assert out.dtypes["only"] == "int64"


def test_meta_thunk_is_not_called_on_the_pandas_path_when_rows_exist(frame):
    calls = []

    def meta():
        calls.append(1)
        return pd.DataFrame()

    BetterFrame(frame).apply(add_column, meta=meta)
    assert calls == [], "meta must not be built when the result is non-empty"


def test_meta_thunk_is_called_on_the_dask_path(dask_frame):
    calls = []

    def meta():
        calls.append(1)
        return add_column(dask_frame._meta)

    BetterFrame(dask_frame).apply(add_column, meta=meta)
    assert calls, "dask needs the meta up front"


def test_apply_leaves_the_frame_alone_when_no_schema_can_be_inferred(frame):
    """A helper that cannot run on a synthetic frame must not have a schema
    guessed for it."""
    empty = frame.iloc[:0]

    def unrunnable(df):
        if df.empty:
            return df
        raise RuntimeError("cannot run on populated frames")

    out = BetterFrame(empty).apply(unrunnable).native
    assert list(out.columns) == list(empty.columns)


# --- index names -------------------------------------------------------------


def test_index_names_match_across_engines(frame, dask_frame):
    """A single-level index, which is what `from_pandas` supports."""
    indexed = frame.set_index("g")
    indexed_dask = dask_frame.set_index("g")

    assert list(BetterFrame(indexed).index_names()) == ["g"]
    assert list(BetterFrame(indexed_dask).index_names()) == ["g"]


def test_index_names_reads_a_multiindex(frame, dask_frame):
    """Dask cannot build a MultiIndex frame directly, but a groupby produces
    one -- which is how these frames arise in practice."""
    grouped = frame.groupby(["g", "v"]).sum()
    grouped_dask = dask_frame.groupby(["g", "v"]).sum()

    assert list(BetterFrame(grouped).index_names()) == ["g", "v"]
    assert list(BetterFrame(grouped_dask).index_names()) == ["g", "v"]


# --- aggregation keywords ----------------------------------------------------


def test_agg_kwargs_are_engine_appropriate(frame, dask_frame):
    assert BetterFrame(frame).agg_kwargs() == {}
    assert BetterFrame(dask_frame).agg_kwargs() == {"split_out": dask_frame.npartitions}


def test_agg_kwargs_can_be_splatted_into_groupby(frame, dask_frame):
    for target in (frame, dask_frame):
        bf = BetterFrame(target)
        out = target.groupby("g").agg({"v": "sum"}, **bf.agg_kwargs())
        out = out.compute() if bf.is_dask else out
        assert out.loc["x", "v"] == 3.0
        assert out.loc["y", "v"] == 3.0


# --- finalize and mutability -------------------------------------------------


def test_finalize_is_a_noop_for_pandas(frame):
    assert BetterFrame(frame).finalize() is frame


def test_finalize_persists_a_dask_frame(dask_frame):
    out = BetterFrame(dask_frame).finalize()
    pd.testing.assert_frame_equal(out.compute(), dask_frame.compute())


def test_mutable_protects_the_callers_pandas_frame(frame):
    before = frame.copy()
    working = BetterFrame(frame).mutable().native
    working["injected"] = 1
    pd.testing.assert_frame_equal(frame, before)


def test_mutable_is_a_passthrough_for_dask(dask_frame):
    assert BetterFrame(dask_frame).mutable().native is dask_frame


# --- meta_source -------------------------------------------------------------


def test_meta_source_exposes_the_dask_shape_for_pandas_frames(frame):
    source = BetterFrame(frame).meta_source()
    assert list(source.columns) == list(frame.columns)
    assert source._meta.empty
    assert source._meta.dtypes.to_dict() == frame.dtypes.to_dict()


def test_meta_source_returns_a_dask_frame_unchanged(dask_frame):
    assert BetterFrame(dask_frame).meta_source() is dask_frame


def test_meta_source_lets_one_meta_builder_serve_both_engines(frame, dask_frame):
    """A builder written against Dask should work unchanged on pandas."""

    def build_meta(source):
        return pd.DataFrame(
            {c: pd.Series(dtype=source._meta[c].dtype) for c in source.columns}
        )

    from_pandas = build_meta(BetterFrame(frame).meta_source())
    from_dask = build_meta(BetterFrame(dask_frame).meta_source())
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


# --- the wrapper itself ------------------------------------------------------


def test_native_returns_the_underlying_frame(frame, dask_frame):
    assert BetterFrame(frame).native is frame
    assert BetterFrame(dask_frame).native is dask_frame


def test_operations_chain_and_stay_wrapped(frame):
    out = (
        BetterFrame(frame)
        .mutable()
        .apply(add_column, value=2)
        .apply(lambda df: df.assign(doubled=df["v"] * 2))
    )
    assert isinstance(out, BetterFrame)
    assert out.native["extra"].tolist() == [2, 2, 2]
    assert out.native["doubled"].tolist() == [2.0, 4.0, 6.0]


def test_pipe_runs_against_the_native_frame_and_rewraps(frame, dask_frame):
    for target in (frame, dask_frame):
        bf = BetterFrame(target)
        agg_kwargs = bf.agg_kwargs()
        out = bf.pipe(lambda df, kw=agg_kwargs: df.groupby("g").agg({"v": "sum"}, **kw))
        assert isinstance(out, BetterFrame)
        assert out.is_dask == bf.is_dask, "engine must survive a pipe"
        native = out.finalize()
        native = native.compute() if bf.is_dask else native
        assert native.loc["x", "v"] == 3.0


def test_chaining_preserves_the_engine_across_frames(dask_frame):
    """The ops object is reused as the chain moves between frames."""
    bf = BetterFrame(dask_frame)
    derived = bf.apply(add_column).pipe(lambda df: df.groupby("g").sum())
    assert derived.is_dask
    assert isinstance(derived.ops, DaskOps)


def test_repr_names_the_engine(frame, dask_frame):
    assert "pandas" in repr(BetterFrame(frame))
    assert "dask" in repr(BetterFrame(dask_frame))


def test_ops_subclass_survives_chaining(frame):
    class Tagged(PandasOps):
        tag = "mine"

    out = BetterFrame(frame, pandas_ops=Tagged).apply(add_column)
    assert isinstance(out.ops, Tagged)
    assert out.ops.tag == "mine"


# --- set aggregations --------------------------------------------------------


@pytest.fixture
def set_frame():
    return pd.DataFrame(
        {
            "g": ["x", "x", "y", "y"],
            "name": ["alpha", "beta", "gamma", None],
            "tags": [
                frozenset({"a", "b"}),
                frozenset({"b", "c"}),
                frozenset({"d"}),
                None,
            ],
        }
    )


def _agg(frame, column, aggregation_name):
    bf = BetterFrame(frame)
    agg = getattr(bf.ops, aggregation_name)()
    out = frame.groupby("g").agg({column: agg}, **bf.agg_kwargs())
    return out.compute() if bf.is_dask else out


def test_set_union_matches_across_engines(set_frame):
    on_pandas = _agg(set_frame, "name", "set_union")
    on_dask = _agg(dd.from_pandas(set_frame, npartitions=2), "name", "set_union")
    assert on_pandas["name"].to_dict() == on_dask["name"].to_dict()
    assert on_pandas["name"]["x"] == frozenset({"alpha", "beta"})


def test_set_union_flatten_matches_across_engines(set_frame):
    on_pandas = _agg(set_frame, "tags", "set_union_flatten")
    on_dask = _agg(
        dd.from_pandas(set_frame, npartitions=2), "tags", "set_union_flatten"
    )
    assert on_pandas["tags"].to_dict() == on_dask["tags"].to_dict()
    assert on_pandas["tags"]["x"] == frozenset({"a", "b", "c"})


def test_both_engines_return_frozensets(set_frame):
    on_pandas = _agg(set_frame, "tags", "set_union_flatten")
    on_dask = _agg(
        dd.from_pandas(set_frame, npartitions=2), "tags", "set_union_flatten"
    )
    assert all(isinstance(v, frozenset) for v in on_pandas["tags"])
    assert all(isinstance(v, frozenset) for v in on_dask["tags"])


def test_nulls_are_dropped_not_propagated(set_frame):
    """Group 'y' has a null in both columns; it must not poison the result."""
    assert _agg(set_frame, "name", "set_union")["name"]["y"] == frozenset({"gamma"})
    assert _agg(set_frame, "tags", "set_union_flatten")["tags"]["y"] == frozenset({"d"})


def test_strings_stay_atomic_when_flattened():
    """The trap a naive set().union(*values) falls into: a string is iterable,
    so unioning would turn a set of names into a set of letters."""
    frame = pd.DataFrame({"g": ["x", "x"], "tags": ["abc", frozenset({"d"})]})
    result = _agg(frame, "tags", "set_union_flatten")["tags"]["x"]
    assert result == frozenset({"abc", "d"}), result
    assert "a" not in result


def test_scalars_mixed_with_collections_do_not_raise():
    """set().union(*values) raises on a bare scalar; flattening must not."""
    frame = pd.DataFrame({"g": ["x", "x"], "tags": [42, frozenset({"d"})]})
    assert _agg(frame, "tags", "set_union_flatten")["tags"]["x"] == frozenset({42, "d"})


def test_aggregations_survive_repartitioning(set_frame):
    """The Dask chunk and combine stages must agree however the data is split."""
    results = [
        _agg(dd.from_pandas(set_frame, npartitions=n), "tags", "set_union_flatten")[
            "tags"
        ].to_dict()
        for n in (1, 2, 4)
    ]
    assert results[0] == results[1] == results[2]


def test_flattening_mixed_types_matches_across_engines():
    """The mixed scalar/string/set case, on both engines rather than pandas alone.

    Note this relies on the column being genuine object dtype. Dask's
    `dataframe.convert-string` (on by default, disabled for these tests) would
    otherwise rewrite the column to its string dtype at construction, turning
    each set into its repr before any aggregation runs.
    """
    frame = pd.DataFrame({"g": ["x", "x", "x"], "tags": ["abc", frozenset({"d"}), 42]})
    expected = frozenset({"abc", "d", 42})

    on_pandas = _agg(frame, "tags", "set_union_flatten")["tags"]["x"]
    on_dask = _agg(dd.from_pandas(frame, npartitions=2), "tags", "set_union_flatten")[
        "tags"
    ]["x"]

    assert on_pandas == expected, on_pandas
    assert on_dask == expected, on_dask


def test_convert_string_corrupts_set_columns_before_aggregation():
    """Documents the trap: with Dask's default string conversion, a set-valued
    object column is stringified at construction, so no aggregation can recover
    it. Callers holding sets in a column must turn that conversion off."""
    frame = pd.DataFrame({"g": ["x", "x"], "tags": ["abc", frozenset({"d"})]})

    with dask.config.set({"dataframe.convert-string": True}):
        converted = dd.from_pandas(frame, npartitions=1)
        assert str(converted.dtypes["tags"]) == "string"
        mangled = _agg(converted, "tags", "set_union_flatten")["tags"]["x"]

    assert "d" not in mangled, (
        "if this passes, dask stopped stringifying and the docs can drop the warning"
    )
    assert any("frozenset" in str(v) for v in mangled)
