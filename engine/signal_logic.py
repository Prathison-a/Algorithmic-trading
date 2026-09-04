"""
signal_logic.py

Converts the TCN model's predicted next-bar Open into BUY / SELL / HOLD.

Responsibility of this module (and ONLY this module):
    Given current price, EMA9, EMA20, and predicted_next_open (from the
    TCN), decide BUY / SELL / HOLD.

This module NEVER trains or calls the TCN. It only consumes the
prediction the model already produced. This keeps the trading logic
fully swappable without ever touching the trained model.

Decision path
--------------
    1. predicted_move_pct = (predicted_next_open - current_price) / current_price
    2. trend = "UP" if ema9 > ema20, "DOWN" if ema9 < ema20, else "FLAT"
    3. BUY  when predicted_move_pct >  threshold_pct AND trend == "UP"
       SELL when predicted_move_pct < -threshold_pct AND trend == "DOWN"
       HOLD otherwise (move too small, or prediction disagrees with trend)

Edge-triggered signals
-----------------------
By default, the engine only emits BUY/SELL on the first candle a state
becomes true, then HOLD for every candle after that (even if the raw
condition is still true) until the opposite condition fires. This gives:

    BUY -> HOLD -> HOLD -> ... -> SELL -> HOLD -> ... -> BUY ...

instead of repeating BUY/SELL every candle the raw condition happens to
still hold. Set `SignalLogicConfig.edge_triggered = False` for the raw,
non-deduped behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# current_price is whatever you consider "now" -- typically the latest
# close. predicted_next_open comes straight out of the TCN (already
# inverse-scaled back to real price units).
REQUIRED_COLUMNS = ("current_price", "ema9", "ema20", "predicted_next_open")


@dataclass
class SignalLogicConfig:
    """
    Tunable parameters. Nothing here touches the model -- safe to change
    any time.
    """

    # Minimum predicted move, as a fraction of current_price, required to
    # act instead of HOLD. 0.0001 = 0.01% -- realistic for 1-minute bars
    # (0.1% on a ~$750 SPY bar would be a 75 cent move in one minute,
    # which basically never happens, so that threshold always HOLDs).
    threshold_pct: float = 0.0001

    # Optional minimum EMA9/EMA20 separation (as a fraction of price) to
    # avoid signaling on a razor-thin, noisy crossover. 0.0 disables this.
    min_ema_gap_pct: float = 0.0

    # If True (default), only emit BUY/SELL on the candle where the signal
    # state actually changes; every other candle is HOLD. If False, BUY/SELL
    # fires on every candle where the raw condition holds.
    edge_triggered: bool = True


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Signal logic input is missing required columns: {missing}. "
            f"Expected at least: {REQUIRED_COLUMNS}"
        )


def compute_predicted_move_pct(df: pd.DataFrame) -> pd.Series:
    """
    (predicted_next_open - current_price) / current_price

    Positive => model expects price to rise.
    Negative => model expects price to fall.
    """
    return (df["predicted_next_open"] - df["current_price"]) / df["current_price"]


def trend_state(df: pd.DataFrame, min_gap_pct: float = 0.0) -> pd.Series:
    """
    "UP"   when ema9 is above ema20 by at least min_gap_pct
    "DOWN" when ema9 is below ema20 by at least min_gap_pct
    "FLAT" otherwise
    """
    gap_pct = (df["ema9"] - df["ema20"]) / df["current_price"]

    state = pd.Series("FLAT", index=df.index)
    state[gap_pct > min_gap_pct] = "UP"
    state[gap_pct < -min_gap_pct] = "DOWN"
    return state


# ---------------------------------------------------------------------------
# Edge-triggering: collapse repeated BUY/BUY/BUY... or SELL/SELL/SELL... runs
# down to a single signal on the first candle of the run. A HOLD candle does
# NOT reset the position state -- only an opposite BUY/SELL flips it.
# ---------------------------------------------------------------------------

def _dedupe_signal_transitions(raw_signals: pd.Series) -> pd.Series:
    """
    Example:
        raw:    HOLD BUY BUY BUY HOLD SELL SELL BUY
        result: HOLD BUY HOLD HOLD HOLD SELL HOLD BUY
    """
    out = []
    last_actionable: Optional[str] = None

    for value in raw_signals:
        if value == Signal.HOLD.value:
            out.append(Signal.HOLD.value)
        elif value != last_actionable:
            out.append(value)
            last_actionable = value
        else:
            out.append(Signal.HOLD.value)

    return pd.Series(out, index=raw_signals.index)


# ---------------------------------------------------------------------------
# Strategy: predicted move vs current price, filtered by EMA9/EMA20 trend
# ---------------------------------------------------------------------------

def tcn_prediction_strategy(df: pd.DataFrame, config: SignalLogicConfig) -> pd.Series:
    """
    Raw condition (before edge-triggering):
        BUY  when predicted_move_pct >  threshold_pct AND trend == "UP"
        SELL when predicted_move_pct < -threshold_pct AND trend == "DOWN"
        HOLD otherwise

    If config.edge_triggered is True (default), repeated BUY/SELL candles
    are collapsed so only the first candle of each new state fires.
    """
    _validate_columns(df)

    move_pct = compute_predicted_move_pct(df)
    trend = trend_state(df, config.min_ema_gap_pct)

    bullish_move = move_pct > config.threshold_pct
    bearish_move = move_pct < -config.threshold_pct

    raw_signals = pd.Series(Signal.HOLD.value, index=df.index)
    raw_signals[bullish_move & (trend == "UP")] = Signal.BUY.value
    raw_signals[bearish_move & (trend == "DOWN")] = Signal.SELL.value

    if config.edge_triggered:
        return _dedupe_signal_transitions(raw_signals)
    return raw_signals


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SignalEngine:
    """
    Thin wrapper so the strategy function can be swapped later (e.g. an
    RSI/MACD confirmation layer on top) without changing caller code.

    Usage:
        engine = SignalEngine()
        result_df = engine.generate(df)  # needs current_price/ema9/ema20/predicted_next_open
    """

    def __init__(
        self,
        strategy=tcn_prediction_strategy,
        config: Optional[SignalLogicConfig] = None,
    ):
        self.strategy = strategy
        self.config = config or SignalLogicConfig()

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a copy of df with two new columns:
            - 'predicted_move_pct': the raw move % used in the decision
            - 'trend'             : "UP" / "DOWN" / "FLAT"
            - 'signal'            : "BUY" / "SELL" / "HOLD"
        """
        _validate_columns(df)
        out = df.copy()
        out["predicted_move_pct"] = compute_predicted_move_pct(out)
        out["trend"] = trend_state(out, self.config.min_ema_gap_pct)
        out["signal"] = self.strategy(out, self.config)
        return out


# ---------------------------------------------------------------------------
# Convert signals into Lightweight Charts markers
# ---------------------------------------------------------------------------

def to_lightweight_charts_markers(df: pd.DataFrame, time_column: str = "time") -> list[dict]:
    """
    Convert a DataFrame that already has a 'signal' column (from
    SignalEngine.generate) into the marker format Lightweight Charts expects
    for `series.setMarkers([...])`.

    Only BUY and SELL rows produce markers; HOLD rows are skipped.
    """
    if "signal" not in df.columns:
        raise ValueError("DataFrame must have a 'signal' column. Run SignalEngine.generate() first.")
    if time_column not in df.columns:
        raise ValueError(f"DataFrame must have a '{time_column}' column for chart timestamps.")

    markers = []
    for _, row in df.iterrows():
        if row["signal"] == Signal.BUY.value:
            markers.append(
                {
                    "time": row[time_column],
                    "position": "belowBar",
                    "color": "#26a69a",
                    "shape": "arrowUp",
                    "text": "BUY",
                }
            )
        elif row["signal"] == Signal.SELL.value:
            markers.append(
                {
                    "time": row[time_column],
                    "position": "aboveBar",
                    "color": "#ef5350",
                    "shape": "arrowDown",
                    "text": "SELL",
                }
            )
    return markers


if __name__ == "__main__":
    # Minimal smoke test using a tiny, clearly-labeled synthetic frame.
    # This block only sanity-checks the logic in isolation -- never used
    # by the real pipeline.
    #
    # Candles 1-3: ema9 > ema20 (UP trend) and predicted_next_open is
    # meaningfully above current_price -- with edge_triggered=True you
    # should see exactly one BUY (candle 1), then HOLD, then a SELL once
    # the trend flips down and the prediction turns bearish.
    sample = pd.DataFrame(
        {
            "time": range(8),
            "current_price":      [100, 101, 102, 103, 103, 102, 101, 100],
            "ema9":                [99, 100, 101, 102, 102, 101.5, 101, 100.5],
            "ema20":               [100, 99.5, 99.8, 100, 100.5, 101.2, 101.5, 101.8],
            "predicted_next_open": [100.3, 101.4, 102.6, 103.5, 103.2, 101.5, 100.2, 99.5],
        }
    )
    engine = SignalEngine()
    result = engine.generate(sample)
    print(result[["current_price", "ema9", "ema20", "predicted_next_open",
                   "predicted_move_pct", "trend", "signal"]])
    print(to_lightweight_charts_markers(result))