"""
engine/forecast_signals.py

Backend glue for the TCN pipeline: takes the DataFrame /forecast already
builds (candles + ema9 + ema20 + the TCN's predicted_next_open) and
produces the `signals` payload frontend/app.js expects in the /forecast
response.

This file does NOT duplicate any signal logic — it only imports and calls
signal_logic.py, then formats the result for the wire. If you improve the
strategy later, you only ever touch signal_logic.py; this file does not
need to change.
"""

from __future__ import annotations

import pandas as pd

from .signal_logic import SignalEngine, to_lightweight_charts_markers


def build_signals_payload(
    df: pd.DataFrame,
    time_column: str = "time",
    engine: SignalEngine | None = None,
) -> list[dict]:
    """
    Run the signal engine over `df` and return markers ready to drop
    straight into the /forecast JSON response as `signals`.

    signal_logic.py expects columns: current_price, ema9, ema20,
    predicted_next_open. If the incoming df instead has the older
    'close' / 'prediction' names (e.g. from an existing /forecast
    builder), they're renamed here so callers don't need to change —
    this is pure column-naming glue, not signal logic.

    df must contain (after renaming): time_column, 'current_price',
    'ema9', 'ema20', 'predicted_next_open'. Only rows present in ALL of
    those (after any merge you do to build df) can produce a signal —
    rows missing a prediction simply won't appear.
    """
    if df.empty:
        return []

    rename_map = {}
    if "current_price" not in df.columns and "close" in df.columns:
        rename_map["close"] = "current_price"
    if "predicted_next_open" not in df.columns and "prediction" in df.columns:
        rename_map["prediction"] = "predicted_next_open"

    if rename_map:
        df = df.rename(columns=rename_map)

    engine = engine or SignalEngine()
    result_df = engine.generate(df)
    return to_lightweight_charts_markers(result_df, time_column=time_column)