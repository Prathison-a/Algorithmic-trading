import threading
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine.inference_engine import InferenceEngine
from engine.tcn_predictor import TCNPredictor
from engine.gru_predictor import GRUPredictor


# ---------------------------------------------------------
# Basic project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_FILE = FRONTEND_DIR / "index.html"


# ---------------------------------------------------------
# Create FastAPI app
# ---------------------------------------------------------

app = FastAPI(title="Smart Trading Dashboard")


# ---------------------------------------------------------
# Connect frontend static files
# Example:
# /static/style.css
# /static/app.js
# ---------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static"
    )


# ---------------------------------------------------------
# Training status data
# This stores live training progress
# ---------------------------------------------------------

training_state = {
    "running": False,
    "epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "symbol": "",
    "model": ""
}


# ---------------------------------------------------------
# Threading variables
# These help training run without blocking the website
# ---------------------------------------------------------

training_lock = threading.Lock()
training_thread = None
stop_requested = False


# ---------------------------------------------------------
# Request body for forecast API
# ---------------------------------------------------------

class ForecastRequest(BaseModel):
    symbol: str
    model: str
    data_source: str = "ftp"
    data_path: str = ""
    timeframe: int = 1
    forecast_steps: int = 15
    history: int = 500
    indicators: list[str] = Field(default_factory=list)
    show_ema9: bool = True
    show_ema20: bool = True
    show_ma: bool = True
    show_vwap: bool = True
    show_rsi: bool = True
    show_volume_delta: bool = True
    ma_period: int = 20
    rsi_period: int = 14


# ---------------------------------------------------------
# Request body for training API
# ---------------------------------------------------------

class TrainRequest(BaseModel):
    symbol: str
    model: str
    data_source: str = "ftp"
    data_path: str = ""
    indicators: list[str] = Field(default_factory=list)


# ---------------------------------------------------------
# Create inference engine object
# ---------------------------------------------------------

engine = InferenceEngine()


# ---------------------------------------------------------
# Calculate EMA 9, EMA 20, and ATR indicators
# ---------------------------------------------------------

def calculate_indicators(df):
    df = df.copy()

    df["datetime"] = pd.to_datetime(df["datetime"])

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

    high_low = df["high"] - df["low"]
    high_close = df["high"] - df["close"].shift(1)
    high_close = high_close.abs()

    low_close = df["low"] - df["close"].shift(1)
    low_close = low_close.abs()

    true_range_data = [high_low, high_close, low_close]

    true_range = pd.concat(true_range_data, axis=1)
    true_range = true_range.max(axis=1)

    df["atr"] = true_range.rolling(14).mean()

    df.dropna(inplace=True)

    return df


# ---------------------------------------------------------
# Calculate MA (Simple Moving Average) and VWAP
# MA: rolling mean of close over "period" bars
# VWAP: cumulative (typical price * volume) / cumulative volume,
#       reset each day so it behaves like a standard session VWAP
# ---------------------------------------------------------

def calculate_ma_vwap(df, ma_period=20):
    df = df.copy()

    df["ma"] = df["close"].rolling(window=ma_period).mean()

    if "volume" in df.columns:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        pv = typical_price * df["volume"]

        # Reset VWAP at the start of each trading day
        day_key = df["datetime"].dt.date

        cum_pv = pv.groupby(day_key).cumsum()
        cum_vol = df["volume"].groupby(day_key).cumsum()

        df["vwap"] = cum_pv / cum_vol.replace(0, pd.NA)
    else:
        df["vwap"] = pd.NA

    return df


# ---------------------------------------------------------
# Calculate RSI (Relative Strength Index)
# Standard 14-period RSI using Wilder's smoothing
# (an EWM with alpha = 1/period, which is what most charting
# platforms - TradingView included - use by default)
# ---------------------------------------------------------

def calculate_rsi(df, period=14):
    df = df.copy()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)

    df["rsi"] = 100 - (100 / (1 + rs))

    # Where avg_loss is 0 (pure uptrend), RS is undefined -> RSI should read 100
    df.loc[avg_loss == 0, "rsi"] = 100

    return df


# ---------------------------------------------------------
# Calculate Volume Delta
# Approximate signed (buy - sell) volume per bar from OHLCV only.
# Same formula used across GRU_Training.py / TCN_Training.py /
# gru_predictor.py / tcn_predictor.py, so the chart matches what
# the models are actually trained on:
#     close > open  -> +volume  (counted as buying)
#     close < open  -> -volume  (counted as selling)
#     close == open -> 0
# ---------------------------------------------------------

def calculate_volume_delta(df):
    df = df.copy()

    direction = df["close"] - df["open"]
    direction = direction.apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )

    df["volume_delta"] = direction * df["volume"]

    return df


# ---------------------------------------------------------
# Bollinger Bands / MACD / Stochastic / MFI / ADX / OBV
#
# Formulas here are copied verbatim from GRU_Training.py / TCN_Training.py
# so the chart display matches what the models were actually trained on.
# ---------------------------------------------------------

def calculate_bollinger_bands(df, period=20, std=2.0):
    df = df.copy()

    mid = df["close"].rolling(period).mean()
    dev = df["close"].rolling(period).std()

    df["bb_upper"] = mid + (std * dev)
    df["bb_middle"] = mid
    df["bb_lower"] = mid - (std * dev)

    return df


def calculate_macd(df, fast=12, slow=26, signal=9):
    df = df.copy()

    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    df["macd_line"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_line - signal_line

    return df


def calculate_stochastic(df, k_period=14, d_period=3):
    df = df.copy()

    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    stoch_range = (high_max - low_min).replace(0, pd.NA)

    k = 100 * (df["close"] - low_min) / stoch_range
    d = k.rolling(d_period).mean()

    df["stoch_k"] = k
    df["stoch_d"] = d

    return df


def calculate_mfi(df, period=14):
    df = df.copy()

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical_price * df["volume"]
    direction = typical_price.diff()

    positive_flow = money_flow.where(direction > 0, 0.0).rolling(period).sum()
    negative_flow = money_flow.where(direction < 0, 0.0).rolling(period).sum()

    money_ratio = positive_flow / negative_flow.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))
    mfi[negative_flow == 0] = 100

    df["mfi"] = mfi

    return df


def calculate_adx(df, period=14):
    df = df.copy()

    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr_smooth = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_smooth
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_smooth

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum

    df["adx"] = dx.ewm(alpha=1 / period, adjust=False).mean()

    return df


def calculate_obv(df):
    df = df.copy()

    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (direction * df["volume"]).cumsum()

    return df


# ---------------------------------------------------------
# Convert 1-minute candles into selected timeframe candles
# Example:
# timeframe = 5 means 5-minute candles
# ---------------------------------------------------------

def resample_timeframe(df, timeframe):
    if timeframe == 1:
        return df

    df = df.copy()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")

    rule = str(timeframe) + "min"

    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    resampled = resampled.dropna()
    resampled.reset_index(inplace=True)

    return resampled


# ---------------------------------------------------------
# Update training progress
# This function is called from training file
# ---------------------------------------------------------

def progress_callback(epoch, total_epochs, train_loss, val_loss):
    with training_lock:
        training_state["epoch"] = epoch
        training_state["total_epochs"] = total_epochs
        training_state["train_loss"] = float(train_loss)
        training_state["val_loss"] = float(val_loss)


# ---------------------------------------------------------
# Check whether user clicked stop training
# ---------------------------------------------------------

def stop_callback():
    return stop_requested


# ---------------------------------------------------------
# Training function
# This runs inside a separate thread
# ---------------------------------------------------------

def training_worker(symbol, model_name, indicators=None, data_source="ftp", data_path=""):
    global stop_requested

    try:
        with training_lock:
            training_state["running"] = True
            training_state["epoch"] = 0
            training_state["total_epochs"] = 0
            training_state["train_loss"] = 0.0
            training_state["val_loss"] = 0.0
            training_state["symbol"] = symbol
            training_state["model"] = model_name

        stop_requested = False

        model_name = model_name.upper()

        if model_name == "TCN":
            from train.TCN_Training import train_model
        elif model_name == "GRU":
            from train.GRU_Training import train_model
        else:
            raise RuntimeError("Unknown model: " + model_name)

        train_model(
            symbol=symbol,
            data_source=data_source,
            data_path=data_path,
            indicators=indicators,
            progress_callback=progress_callback,
            stop_callback=stop_callback
        )

    except Exception as error:
        print("TRAINING ERROR:", str(error))

    finally:
        with training_lock:
            training_state["running"] = False


# ---------------------------------------------------------
# Home page route
# Opens frontend/index.html
# ---------------------------------------------------------

@app.get("/")
def home():
    if not INDEX_FILE.exists():
        return {
            "message": "FastAPI running",
            "error": "frontend/index.html not found"
        }

    return FileResponse(str(INDEX_FILE))


# ---------------------------------------------------------
# Health check route
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "online"
    }


# ---------------------------------------------------------
# Available models
# ---------------------------------------------------------

@app.get("/models")
def models():
    return ["TCN", "GRU"]


# ---------------------------------------------------------
# Available symbols
# ---------------------------------------------------------

@app.get("/symbols")
def symbols():
    return ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"]


# ---------------------------------------------------------
# Get current training status
# ---------------------------------------------------------

@app.get("/training-status")
def training_status():
    with training_lock:
        current_state = dict(training_state)

    return current_state


# ---------------------------------------------------------
# Test data loading
# ---------------------------------------------------------

@app.get("/test-load")
def test_load():
    try:
        df = engine.load_data("SPY")

        return {
            "rows": len(df),
            "columns": list(df.columns)
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


# ---------------------------------------------------------
# Start model training
# ---------------------------------------------------------

@app.post("/train")
def train(request: TrainRequest):
    global training_thread

    try:
        # Prevent multiple training jobs
        if training_thread is not None and training_thread.is_alive():
            return {
                "success": False,
                "message": "Training already running"
            }

        print("======================================")
        print("Training Request Received")
        print("Symbol     :", request.symbol)
        print("Model      :", request.model)
        print("Data Source:", request.data_source)
        print("Data Path  :", request.data_path)
        print("Indicators :", request.indicators)
        print("======================================")

        training_thread = threading.Thread(
            target=training_worker,
            args=(
                request.symbol,
                request.model,
                request.indicators,
                request.data_source,
                request.data_path
            ),
            daemon=True
        )

        training_thread.start()

        return {
            "success": True,
            "message": "Training started",
            "symbol": request.symbol,
            "model": request.model,
            "data_source": request.data_source,
            "data_path": request.data_path,
            "indicators": request.indicators
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
# ---------------------------------------------------------
# Stop model training
# ---------------------------------------------------------

@app.post("/stop-training")
def stop_training():
    global stop_requested

    stop_requested = True

    return {
        "success": True
    }


# ---------------------------------------------------------
# Forecast route
# This returns candles, EMA lines, MA, VWAP, RSI, Volume Delta,
# Bollinger Bands, MACD, Stochastic, MFI, ADX, OBV, historical
# prediction, and future prediction
#
# NOTE: every output loop below now guards against NaN with
# pd.isna(...)/skip. Without this, a single NaN (e.g. from a bad
# model prediction, or an early-window indicator) gets serialized
# as JSON null, and lightweight-charts throws an uncaught
# "Value is null" error the moment setData() hits it on the
# frontend -- which aborts the rest of runForecast() in app.js
# (including chart.timeScale().fitContent()), leaving the charts
# looking blank/empty even though the network call itself succeeded.
# ---------------------------------------------------------

@app.post("/forecast")
def forecast(request: ForecastRequest):
    try:
        df = engine.load_data(request.symbol, request.data_source, request.data_path)

        df = calculate_indicators(df)

        model_df = df.copy()

        df_display = model_df.tail(request.history).copy()

        df_display = resample_timeframe(
            df_display,
            request.timeframe
        )

        df_display["ema9"] = df_display["close"].ewm(
            span=9,
            adjust=False
        ).mean()

        df_display["ema20"] = df_display["close"].ewm(
            span=20,
            adjust=False
        ).mean()

        df_display = calculate_ma_vwap(
            df_display,
            ma_period=request.ma_period
        )

        df_display = calculate_rsi(
            df_display,
            period=request.rsi_period
        )

        df_display = calculate_volume_delta(df_display)

        df_display = calculate_bollinger_bands(df_display)
        df_display = calculate_macd(df_display)
        df_display = calculate_stochastic(df_display)
        df_display = calculate_mfi(df_display)
        df_display = calculate_adx(df_display)
        df_display = calculate_obv(df_display)

        model_name = request.model.upper()

        if model_name == "TCN":
            predictor = TCNPredictor(request.symbol)
        elif model_name == "GRU":
            predictor = GRUPredictor(request.symbol)
        else:
            raise RuntimeError("Unknown model: " + request.model)

        predict_res = predictor.predict(
            model_df,
            request.forecast_steps
        )
        future_times = predict_res[0]
        future_predictions = predict_res[1]

        historical_times, historical_predictions = predictor.historical_predictions(
            model_df
        )

        keep = request.history

        historical_times = historical_times[-keep:]
        historical_predictions = historical_predictions[-keep:]

        # --- Future predictions (anchor point + model output) ---
        # NEW: skip NaN so a single bad model output can't null-crash the chart
        future_output = []

        last_time = model_df["datetime"].iloc[-1]
        last_open = model_df["open"].iloc[-1]

        if not pd.isna(last_open):
            future_output.append({
                "time": int(last_time.timestamp()),
                "value": float(last_open)
            })

        for time_value, prediction_value in zip(future_times, future_predictions):
            if pd.isna(prediction_value):
                continue
            future_output.append({
                "time": int(time_value.timestamp()),
                "value": float(prediction_value)
            })

        # --- Candles ---
        # NEW: skip any row with a NaN OHLC value
        candles_output = []

        for index, row in df_display.iterrows():
            if (
                pd.isna(row["open"]) or pd.isna(row["high"])
                or pd.isna(row["low"]) or pd.isna(row["close"])
            ):
                continue
            candles_output.append({
                "time": int(row["datetime"].timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]) if "volume" in row and not pd.isna(row["volume"]) else 0.0
            })

        # --- EMA9 ---
        # NEW: skip NaN
        ema9_output = []

        for index, row in df_display.iterrows():
            if pd.isna(row["ema9"]):
                continue
            ema9_output.append({
                "time": int(row["datetime"].timestamp()),
                "value": float(row["ema9"])
            })

        # --- EMA20 ---
        # NEW: skip NaN
        ema20_output = []

        for index, row in df_display.iterrows():
            if pd.isna(row["ema20"]):
                continue
            ema20_output.append({
                "time": int(row["datetime"].timestamp()),
                "value": float(row["ema20"])
            })

        ma_output = []

        for index, row in df_display.iterrows():
            if pd.isna(row["ma"]):
                continue
            ma_output.append({
                "time": int(row["datetime"].timestamp()),
                "value": float(row["ma"])
            })

        vwap_output = []

        for index, row in df_display.iterrows():
            if pd.isna(row["vwap"]):
                continue
            vwap_output.append({
                "time": int(row["datetime"].timestamp()),
                "value": float(row["vwap"])
            })

        rsi_output = []

        for index, row in df_display.iterrows():
            if pd.isna(row["rsi"]):
                continue
            rsi_output.append({
                "time": int(row["datetime"].timestamp()),
                "value": float(row["rsi"])
            })

        # Histogram bars: green for buying pressure, red for selling
        # pressure, matching the candle up/down colors.
        volume_delta_output = []

        for index, row in df_display.iterrows():
            if pd.isna(row["volume_delta"]):
                continue

            value = float(row["volume_delta"])
            bar_color = "#26a69a" if value >= 0 else "#ef5350"

            volume_delta_output.append({
                "time": int(row["datetime"].timestamp()),
                "value": value,
                "color": bar_color
            })

        # --- Bollinger / MACD / Stochastic / MFI / ADX / OBV ---

        bb_upper_output = []
        bb_middle_output = []
        bb_lower_output = []

        for index, row in df_display.iterrows():
            if pd.isna(row["bb_upper"]) or pd.isna(row["bb_middle"]) or pd.isna(row["bb_lower"]):
                continue

            timestamp = int(row["datetime"].timestamp())

            bb_upper_output.append({"time": timestamp, "value": float(row["bb_upper"])})
            bb_middle_output.append({"time": timestamp, "value": float(row["bb_middle"])})
            bb_lower_output.append({"time": timestamp, "value": float(row["bb_lower"])})

        macd_line_output = []
        macd_signal_output = []
        macd_hist_output = []

        for index, row in df_display.iterrows():
            if pd.isna(row["macd_line"]) or pd.isna(row["macd_signal"]) or pd.isna(row["macd_hist"]):
                continue

            timestamp = int(row["datetime"].timestamp())
            hist_value = float(row["macd_hist"])

            macd_line_output.append({"time": timestamp, "value": float(row["macd_line"])})
            macd_signal_output.append({"time": timestamp, "value": float(row["macd_signal"])})
            macd_hist_output.append({
                "time": timestamp,
                "value": hist_value,
                "color": "#26a69a" if hist_value >= 0 else "#ef5350"
            })

        stoch_k_output = []
        stoch_d_output = []

        for index, row in df_display.iterrows():
            if pd.isna(row["stoch_k"]) or pd.isna(row["stoch_d"]):
                continue

            timestamp = int(row["datetime"].timestamp())

            stoch_k_output.append({"time": timestamp, "value": float(row["stoch_k"])})
            stoch_d_output.append({"time": timestamp, "value": float(row["stoch_d"])})

        mfi_output = []

        for index, row in df_display.iterrows():
            if pd.isna(row["mfi"]):
                continue
            mfi_output.append({
                "time": int(row["datetime"].timestamp()),
                "value": float(row["mfi"])
            })

        adx_output = []

        for index, row in df_display.iterrows():
            if pd.isna(row["adx"]):
                continue
            adx_output.append({
                "time": int(row["datetime"].timestamp()),
                "value": float(row["adx"])
            })

        obv_output = []

        for index, row in df_display.iterrows():
            if pd.isna(row["obv"]):
                continue
            obv_output.append({
                "time": int(row["datetime"].timestamp()),
                "value": float(row["obv"])
            })

        # --- Historical predictions ---
        # NEW: skip NaN model outputs
        historical_output = []

        for time_value, prediction_value in zip(historical_times, historical_predictions):
            if pd.isna(prediction_value):
                continue
            historical_output.append({
                "time": int(time_value.timestamp()),
                "value": float(prediction_value)
            })

        # --- ML Buy/Sell Signals via Engine ---
        from engine.forecast_signals import build_signals_payload

        hist_pred_map = {
            int(t.timestamp()): float(p)
            for t, p in zip(historical_times, historical_predictions)
            if not pd.isna(p)
        }

        signal_df_list = []
        for index, row in df_display.iterrows():
            ts = int(row["datetime"].timestamp())
            if pd.isna(row["ema9"]) or pd.isna(row["ema20"]):
                continue

            curr_pred = hist_pred_map.get(ts)
            if curr_pred is not None:
                signal_df_list.append({
                    "time": ts,
                    "close": float(row["close"]),
                    "ema9": float(row["ema9"]),
                    "ema20": float(row["ema20"]),
                    "prediction": float(curr_pred)
                })

        if signal_df_list:
            signal_df = pd.DataFrame(signal_df_list)
            signals_output = build_signals_payload(signal_df, time_column="time")
        else:
            signals_output = []

        return {
            "candles": candles_output,
            "ema9": ema9_output,
            "ema20": ema20_output,
            "ma": ma_output,
            "vwap": vwap_output,
            "rsi": rsi_output,
            "volume_delta": volume_delta_output,

            # NOTE: key names here must match what frontend/app.js reads
            # from the /forecast response (e.g. data.bollinger_upper,
            # data.macd_histogram, data.stochastic_k) -- NOT the internal
            # DataFrame column names used above, which stay bb_upper/
            # macd_hist/stoch_k for consistency with the training scripts.
            "bollinger_upper": bb_upper_output,
            "bollinger_middle": bb_middle_output,
            "bollinger_lower": bb_lower_output,
            "macd_line": macd_line_output,
            "macd_signal": macd_signal_output,
            "macd_histogram": macd_hist_output,
            "stochastic_k": stoch_k_output,
            "stochastic_d": stoch_d_output,
            "mfi": mfi_output,
            "adx": adx_output,
            "obv": obv_output,

            "historical_predictions": historical_output,
            "future_predictions": future_output,
            "signals": signals_output,
            "show_ema9": request.show_ema9,
            "show_ema20": request.show_ema20,
            "show_ma": request.show_ma,
            "show_vwap": request.show_vwap,
            "show_rsi": request.show_rsi,
            "show_volume_delta": request.show_volume_delta
        }

    except Exception as error:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(error))


# ---------------------------------------------------------
# Run FastAPI server
# ---------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )