import pandas as pd

from app.chunkers.timewindow_chunker import build_time_windows, detect_timeseries


def _snapshot_df():
    # 3 different funds, ALL at the same single date — classic snapshot file.
    return pd.DataFrame(
        {
            "Scheme Name": ["HDFC Top 100", "Axis Bluechip", "ICICI Bluechip"],
            "NAV Date": pd.to_datetime(["2026-04-30"] * 3),
            "NAV": [100.0, 200.0, 300.0],
        }
    )


def _series_df():
    # 1 fund, 5 distinct monthly observations — clear time-series.
    dates = pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30"])
    return pd.DataFrame(
        {
            "Scheme Name": ["HDFC Top 100"] * 5,
            "NAV Date": dates,
            "NAV": [100.0, 110.0, 105.0, 115.0, 120.0],
        }
    )


def test_detect_timeseries_returns_none_for_snapshot():
    df = _snapshot_df()
    assert detect_timeseries(df, "Scheme Name") is None


def test_detect_timeseries_finds_date_column():
    df = _series_df()
    assert detect_timeseries(df, "Scheme Name") == "NAV Date"


def test_build_time_windows_emits_chunks_for_series():
    df = _series_df()
    out = build_time_windows(
        df,
        file="x.csv",
        file_id="fid",
        sheet=None,
        name_col="Scheme Name",
        date_col="NAV Date",
        value_cols=["NAV"],
    )
    assert len(out) >= 1
    types = {meta["chunk_type"] for meta, _ in out}
    assert types == {"time_window"}
    # At least the 365d window should be present given 5 months span.
    windows = {meta["window"] for meta, _ in out}
    assert "365d" in windows


def test_cagr_present_only_in_365d_window():
    df = _series_df()
    out = build_time_windows(
        df,
        file="x.csv",
        file_id="fid",
        sheet=None,
        name_col="Scheme Name",
        date_col="NAV Date",
        value_cols=["NAV"],
    )
    saw_365_with_cagr = False
    for meta, _ in out:
        stats = meta.get("stats", {})
        nav = stats.get("NAV", {})
        if meta["window"] == "365d":
            if "cagr" in nav:
                saw_365_with_cagr = True
        else:
            assert "cagr" not in nav
    assert saw_365_with_cagr


def test_cagr_value_close_to_manual():
    df = _series_df()
    out = build_time_windows(
        df,
        file="x.csv",
        file_id="fid",
        sheet=None,
        name_col="Scheme Name",
        date_col="NAV Date",
        value_cols=["NAV"],
    )
    # 100 -> 120 over 4 months ≈ 0.333 days/year ratio
    span_days = (df["NAV Date"].iloc[-1] - df["NAV Date"].iloc[0]).days
    yrs = span_days / 365.25
    expected = (120.0 / 100.0) ** (1.0 / yrs) - 1.0
    cagr_seen = None
    for meta, _ in out:
        if meta["window"] == "365d":
            cagr_seen = meta["stats"]["NAV"].get("cagr")
            break
    assert cagr_seen is not None
    assert abs(cagr_seen - expected) < 1e-6


def test_snapshot_input_emits_no_chunks():
    df = _snapshot_df()
    out = build_time_windows(
        df,
        file="x.csv",
        file_id="fid",
        sheet=None,
        name_col="Scheme Name",
        date_col="NAV Date",
        value_cols=["NAV"],
    )
    # Single observation per entity per window — no chunks emitted (need >= 2).
    assert out == []
