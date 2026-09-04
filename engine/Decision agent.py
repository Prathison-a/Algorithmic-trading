"""
decision_agent.py

Turns a predictor's multi-step forecast + EMA9/EMA20 into a
BUY / SELL / HOLD decision:

    EMA9 > EMA20  AND  prediction line rising   -> BUY
    EMA9 < EMA20  AND  prediction line falling  -> SELL
    otherwise                                    -> HOLD

Model-agnostic: works with either engine.tcn_predictor.TCNPredictor or
engine.gru_predictor.GRUPredictor, AS LONG AS both expose the same
.predict(df, future_steps) -> (future_times, prediction_line)
interface. The current TCN pipeline is single-target (next_open only,
no classification head), so predict() returns a 2-tuple. If GRU still
returns a 3-tuple (with a direction-probability list), this dispatch
will break for "GRU" until GRU is trimmed to match the same
single-target interface.

EMA9/EMA20 are computed directly here with the same formula used
everywhere else in this codebase (df["close"].ewm(span=N,
adjust=False).mean()), rather than through either predictor's
internal indicator machinery. Keeps this module decoupled from
TCN-vs-GRU-specific internals and means it works even for a
checkpoint trained without EMA9/EMA20 in its indicator list.

NOTE: this is a SEPARATE signal from signal_logic.py's single-bar
prediction-vs-current-price check (used by forecast_signals.py for the
historical chart markers). This module is the forward-looking live
decision (EMA9-vs-EMA20 + future prediction-line direction). These two
rules can disagree on a given bar -- decide whether to reconcile them
into one before both are live in production.
"""

from dataclasses import dataclass

from engine.tcn_predictor import TCNPredictor
from engine.gru_predictor import GRUPredictor


# =========================================================
# CONFIG
# =========================================================

# How many bars ahead to forecast for the "prediction line" used in
# the rising/falling check.
LINE_STEPS = 5


# =========================================================
# RESULT CONTAINER
# =========================================================

@dataclass
class Decision:
    action: str                          # "BUY" | "SELL" | "HOLD"
    ema9: float
    ema20: float
    ema_bullish: bool                    # EMA9 > EMA20
    ema_bearish: bool                    # EMA9 < EMA20
    prediction_line: list                # [open+1, ..., open+LINE_STEPS], real price units
    prediction_rising: bool
    prediction_falling: bool
    last_close: float


# =========================================================
# MODEL SELECTION (mirrors app.py's TCN/GRU dispatch in /forecast)
# =========================================================

def get_predictor(model_name, symbol):
    model_name = model_name.upper()

    if model_name == "TCN":
        return TCNPredictor(symbol)
    elif model_name == "GRU":
        return GRUPredictor(symbol)

    raise ValueError(f"Unknown model: {model_name}")


# =========================================================
# EMA EXTRACTION
# =========================================================

def get_latest_ema(dataframe):
    """
    Computes EMA9/EMA20 directly from the dataframe's close column,
    using the same span/adjust settings used everywhere else in this
    codebase, and returns the most recent values.
    """
    df = dataframe.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    ema9_series = df["close"].ewm(span=9, adjust=False).mean()
    ema20_series = df["close"].ewm(span=20, adjust=False).mean()

    return (
        float(ema9_series.iloc[-1]),
        float(ema20_series.iloc[-1]),
        float(df["close"].iloc[-1]),
    )


# =========================================================
# DECISION LOGIC (per the diagram)
# =========================================================

def decide_action(ema9, ema20, prediction_line):
    """
    EMA9 > EMA20  AND  prediction line rising   -> BUY
    EMA9 < EMA20  AND  prediction line falling  -> SELL
    otherwise                                    -> HOLD

    "Rising"/"falling" = net direction over the line (last predicted
    step vs first predicted step). Swap in a strict monotonic check
    (all(later >= earlier)) if you want zero tolerance for a single
    wobble mid-line.
    """
    ema_bullish = ema9 > ema20
    ema_bearish = ema9 < ema20

    prediction_rising = prediction_line[-1] > prediction_line[0]
    prediction_falling = prediction_line[-1] < prediction_line[0]

    if ema_bullish and prediction_rising:
        action = "BUY"
    elif ema_bearish and prediction_falling:
        action = "SELL"
    else:
        action = "HOLD"

    return action, ema_bullish, ema_bearish, prediction_rising, prediction_falling


# =========================================================
# TOP-LEVEL ENTRY POINT
# =========================================================

def run_decision_agent(symbol, dataframe, model="TCN", steps=LINE_STEPS):
    """
    dataframe: raw OHLCV, oldest -> newest, columns
    open/high/low/close/volume (+ datetime, case-insensitive) --
    e.g. straight from engine.load_data(symbol) in app.py.
    """
    predictor = get_predictor(model, symbol)

    ema9, ema20, last_close = get_latest_ema(dataframe)

    # Single-target predictor interface: (future_times, prediction_line).
    _future_times, prediction_line = predictor.predict(dataframe, steps)

    action, ema_bullish, ema_bearish, rising, falling = decide_action(
        ema9, ema20, prediction_line
    )

    return Decision(
        action=action,
        ema9=ema9,
        ema20=ema20,
        ema_bullish=ema_bullish,
        ema_bearish=ema_bearish,
        prediction_line=prediction_line,
        prediction_rising=rising,
        prediction_falling=falling,
        last_close=last_close,
    )


if __name__ == "__main__":
    # Example manual test.
    #
    # from engine.inference_engine import InferenceEngine
    # engine = InferenceEngine()
    # df = engine.load_data("SPY")
    # result = run_decision_agent("SPY", df, model="TCN")
    # print(f"Action: {result.action}")
    # print(f"EMA9={result.ema9:.4f}  EMA20={result.ema20:.4f}")
    # print(f"Prediction line: {result.prediction_line}")
    pass