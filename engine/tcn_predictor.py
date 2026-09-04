import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DROPOUT_RATE = 0.2

RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Must match TCN_Training.py's BASE_FEATURES exactly (same order) --
# the scaler and saved "features" list depend on this ordering.
BASE_FEATURES = ["high", "low", "close", "volume", "open"]

INDICATOR_COLUMN_MAP = {
    "EMA9":  ["ema9"],
    "EMA20": ["ema20"],
    "EMA50": ["ema50"],
    "VWAP":  ["vwap"],
    "RSI14": ["rsi14"],
    "ATR14": ["atr14"],
    "MACD":  ["macd_line", "macd_signal", "macd_hist"],
}


# =============================================================
# INDICATOR NAME NORMALIZATION
# =============================================================

def _normalize_key(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _build_alias_table(canonical_names) -> dict:
    return {_normalize_key(name): name for name in canonical_names}


INDICATOR_ALIASES = _build_alias_table(INDICATOR_COLUMN_MAP.keys())


def normalize_indicators(indicators: list) -> list:
    resolved = []
    for raw_name in indicators:
        key = _normalize_key(raw_name)
        if key not in INDICATOR_ALIASES:
            raise ValueError(
                f"Unknown indicator: {raw_name!r}. This pipeline only "
                f"supports: {list(INDICATOR_COLUMN_MAP.keys())}"
            )
        resolved.append(INDICATOR_ALIASES[key])
    return resolved


# ---------------------------------------------------------------
# TARGETS
#
# TCN_Training.py's TCNDualHead predicts:
#   target_next_return = (next_open - close) / close   (regression head)
#   target_direction    = SELL(0) / HOLD(1) / BUY(2)    (classification head)
#
# These defaults are only used as a fallback when an older checkpoint
# doesn't carry the corresponding metadata -- the checkpoint's own
# values are always preferred (see TCNPredictor.__init__).
# ---------------------------------------------------------------

DEFAULT_TARGET_COLUMNS = [
    "target_next_return",
    "target_direction",
]

DEFAULT_DIRECTION_LABELS = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}

DEFAULT_DIRECTION_THRESHOLD = 0.0015
DEFAULT_MIN_SIGNAL_CONFIDENCE = 0.70

# Caps how far a single recursive step can move away from the previous
# predicted (or last real) price, expressed as a multiple of recent ATR.
MAX_STEP_MOVE_ATR_MULTIPLE = 1.5


# =========================================================
# INDICATOR FORMULAS -- must match TCN_Training.py EXACTLY
# =========================================================

def compute_rsi(close, period=RSI_PERIOD):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs)), avg_gain, avg_loss


def compute_rsi_series(close, period=RSI_PERIOD):
    """Wrapper matching training's public signature (RSI series only)."""
    rsi, _, _ = compute_rsi(close, period)
    return rsi


def compute_atr(df, period=ATR_PERIOD):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def compute_vwap(df):
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_pv = (typical_price * df["volume"]).cumsum()
    cumulative_volume = df["volume"].cumsum().replace(0, np.nan)

    return cumulative_pv / cumulative_volume


def compute_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    return pd.DataFrame({
        "macd_line":   macd_line,
        "macd_signal": signal_line,
        "macd_hist":   macd_line - signal_line,
    }), ema_fast, ema_slow


def add_missing_features(df, indicators):
    df = df.copy()

    for name in indicators:
        cols = INDICATOR_COLUMN_MAP[name]

        if all(c in df.columns for c in cols):
            continue

        if name == "EMA9":
            df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
        elif name == "EMA20":
            df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        elif name == "EMA50":
            df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        elif name == "VWAP":
            df["vwap"] = compute_vwap(df)
        elif name == "RSI14":
            df["rsi14"] = compute_rsi_series(df["close"], period=RSI_PERIOD)
        elif name == "ATR14":
            df["atr14"] = compute_atr(df, period=ATR_PERIOD)
        elif name == "MACD":
            macd_df, _, _ = compute_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
            for col in macd_df.columns:
                df[col] = macd_df[col]
        else:
            raise ValueError(f"Unknown indicator: {name}")

    all_cols = []
    for name in indicators:
        all_cols.extend(INDICATOR_COLUMN_MAP[name])

    existing_cols = [c for c in all_cols if c in df.columns]
    if existing_cols:
        df[existing_cols] = df[existing_cols].bfill()

    return df


def resolve_feature_columns(indicators):
    """
    Fallback only -- used if an old checkpoint has no "features" list
    saved. The checkpoint's own "features" is always authoritative
    when present (see TCNPredictor.__init__), since it also carries
    the 102 options columns which cannot be reconstructed here.
    """
    columns = list(BASE_FEATURES)
    for name in indicators:
        columns.extend(INDICATOR_COLUMN_MAP[name])
    return columns


# =========================================================
# TCN MODEL -- architecture must match TCN_Training.py exactly
# =========================================================

class TemporalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        kernel_size = 5
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                                padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                                padding=padding, dilation=dilation)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(DROPOUT_RATE)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else None
        )

    def forward(self, x):
        output = self.conv1(x)
        output = output[:, :, :-self.conv1.padding[0]]

        output = self.conv2(output)
        output = output[:, :, :-self.conv2.padding[0]]

        output = self.gelu(output)
        output = self.dropout(output)

        residual = x
        if self.downsample:
            residual = self.downsample(x)

        min_length = min(output.shape[2], residual.shape[2])
        output = output[:, :, -min_length:]
        residual = residual[:, :, -min_length:]

        return output + residual


class TCNDualHead(nn.Module):
    """
    Exact architectural mirror of TCN_Training.py's TCNDualHead:
    4 TemporalBlocks (channels 32/64/64/64, dilations 1/2/4/8,
    kernel_size=5), shared fc1/fc2, then two heads:
      - return_head:    Linear(64, 1)  -- predicted target_next_return
      - direction_head: Linear(64, 3)  -- SELL/HOLD/BUY logits
    """

    def __init__(self, n_features):
        super().__init__()

        self.block1 = TemporalBlock(n_features, 32, dilation=1)
        self.block2 = TemporalBlock(32, 64, dilation=2)
        self.block3 = TemporalBlock(64, 64, dilation=4)
        self.block4 = TemporalBlock(64, 64, dilation=8)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.recent_timesteps = 4

        self.fc1 = nn.Linear(64 * self.recent_timesteps, 128)
        self.fc2 = nn.Linear(128, 64)

        self.return_head = nn.Linear(64, 1)
        self.direction_head = nn.Linear(64, 3)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x[:, :, -self.recent_timesteps:]
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout(x)

        return_prediction = self.return_head(x)
        direction_logits = self.direction_head(x)

        return return_prediction, direction_logits


class TCNPredictor:

    def __init__(self, symbol="SPY"):

        symbol = symbol.upper()

        MODEL_PATH = (
            Path(__file__).parent.parent
            / "models"
            / f"best_tcn_{symbol.lower()}.pth"
        )

        print(f"Loading TCN Model: {MODEL_PATH}")

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

        self.indicators = normalize_indicators(checkpoint.get("indicators", []))

        if "features" not in checkpoint:
            raise RuntimeError(
                "Checkpoint is missing 'features' -- cannot determine the "
                "authoritative feature ordering (including the options "
                "columns). Retrain with the current TCN_Training.py."
            )

        # The checkpoint's own feature list/order is authoritative --
        # it is what the model was actually trained on, including the
        # 102 options columns. It is never independently rebuilt.
        self.features = list(checkpoint["features"])

        self.target_columns = checkpoint.get(
            "target_columns", DEFAULT_TARGET_COLUMNS
        )

        self.options_columns = list(checkpoint.get("options_columns", []))

        self.direction_threshold = checkpoint.get(
            "direction_threshold", DEFAULT_DIRECTION_THRESHOLD
        )

        checkpoint_direction_labels = checkpoint.get("direction_labels")
        if checkpoint_direction_labels:
            self.direction_labels = {
                int(key): value
                for key, value in checkpoint_direction_labels.items()
            }
        else:
            self.direction_labels = dict(DEFAULT_DIRECTION_LABELS)

        self.min_signal_confidence = checkpoint.get(
            "min_signal_confidence", DEFAULT_MIN_SIGNAL_CONFIDENCE
        )

        if "sequence_length" not in checkpoint:
            raise RuntimeError(
                "Checkpoint is missing 'sequence_length'."
            )

        self.sequence_length = checkpoint["sequence_length"]

        num_features_checkpoint = checkpoint.get(
            "num_features", len(self.features)
        )

        if num_features_checkpoint != len(self.features):
            raise RuntimeError(
                "Checkpoint feature-count mismatch: "
                f"num_features={num_features_checkpoint} but "
                f"len(features)={len(self.features)}. This predictor is "
                "not compatible with this checkpoint."
            )

        self.model = TCNDualHead(
            n_features=len(self.features)
        ).to(DEVICE)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.feature_scaler.mean_ = checkpoint["feature_scaler_mean"]
        self.feature_scaler.scale_ = checkpoint["feature_scaler_scale"]

        # target_scaler was fit ONLY on target_next_return during
        # training (target_direction is never scaled), so index 0 of
        # mean_/scale_ always corresponds to the return target.
        self.target_scaler.mean_ = checkpoint["target_scaler_mean"]
        self.target_scaler.scale_ = checkpoint["target_scaler_scale"]

    # =====================================================
    # FEATURE VALIDATION
    # =====================================================

    def _validate_features_present(self, df):
        """
        The incoming dataframe must contain every feature the
        checkpoint was trained on, including all 102 options columns.
        Missing options data is never filled with zeros or fabricated
        -- it's a hard error.
        """
        missing = [
            column for column in self.features
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required model features: {missing}"
            )

    # =====================================================
    # DECODING HELPERS
    # =====================================================

    def _decode_return(self, raw_return):
        """
        Inverse-transforms the model's scaled target_next_return
        output back into a real return value using target_scaler
        (fit only on target_next_return -- index 0).
        """
        raw_return = np.atleast_1d(
            np.asarray(raw_return, dtype=np.float64)
        ).reshape(-1)

        scaled_return = raw_return[0]

        mean = self.target_scaler.mean_[0]
        scale = self.target_scaler.scale_[0]

        return float((scaled_return * scale) + mean)

    def _decode_direction(self, direction_logits):
        """
        Converts raw direction_logits into (signal, confidence,
        probabilities). Confidence is the highest softmax probability,
        exactly matching TCN_Training.py's trading-confidence logic.
        """
        probabilities_tensor = torch.softmax(direction_logits, dim=1)
        predicted_class = int(torch.argmax(probabilities_tensor, dim=1).item())

        probs = probabilities_tensor.squeeze(0).detach().cpu().numpy()

        probabilities = {
            self.direction_labels.get(0, "SELL"): float(probs[0]),
            self.direction_labels.get(1, "HOLD"): float(probs[1]),
            self.direction_labels.get(2, "BUY"): float(probs[2]),
        }

        confidence = float(np.max(probs))
        signal = self.direction_labels.get(predicted_class, str(predicted_class))

        return signal, confidence, probabilities

    def _ensure_finite(self, value, name):
        if not np.isfinite(value):
            raise ValueError(
                f"Prediction produced a non-finite value for '{name}': {value}"
            )
        return float(value)

    def _sanitize_for_json(self, obj):
        """
        Recursively strips numpy/torch types and non-finite floats out
        of a result before it's returned, so it's always safely JSON
        serializable.
        """
        if isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(value) for value in obj]

        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()

        if isinstance(obj, np.generic):
            obj = obj.item()

        if isinstance(obj, float):
            if not np.isfinite(obj):
                raise ValueError(f"Non-finite value in prediction result: {obj}")
            return float(obj)

        if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
            return int(obj)

        return obj

    # =====================================================
    # HISTORICAL PREDICTIONS
    # =====================================================

    def historical_predictions(self, df):

        df = df.copy()
        df = add_missing_features(df, self.indicators)

        self._validate_features_present(df)

        # Raw (unscaled) close values are needed to convert predicted
        # returns back into predicted prices -- kept separate from the
        # scaled feature frame used for model input.
        raw_close = df["close"].to_numpy(dtype=np.float64)

        scaled_df = df.copy()
        scaled_df[self.features] = self.feature_scaler.transform(df[self.features])

        historical_times = []
        historical_predictions = []

        with torch.no_grad():
            for i in range(self.sequence_length, len(scaled_df)):

                sequence = scaled_df[self.features].iloc[i - self.sequence_length:i].values

                seq_tensor = (
                    torch.tensor(sequence, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                predicted_return_tensor, _direction_logits = self.model(seq_tensor)

                predicted_return = self._decode_return(
                    predicted_return_tensor.cpu().numpy()[0]
                )

                # The window's last real bar sits at index i - 1; that
                # bar's close is what target_next_return was measured
                # from during training (see add_targets() in
                # TCN_Training.py).
                current_close = float(raw_close[i - 1])

                predicted_price = current_close * (1.0 + predicted_return)
                predicted_price = self._ensure_finite(predicted_price, "predicted_price")

                historical_predictions.append(predicted_price)
                historical_times.append(df["datetime"].iloc[i])

        return historical_times, historical_predictions

    # =====================================================
    # STATE SEEDING
    #
    # Seeds the recursive state each indicator needs so that,
    # during forecasting, appending one synthetic bar advances
    # each indicator by exactly ONE true recursive step -- the
    # same math training used -- instead of recomputing over a
    # shifted window (which silently resets EMA/RSI/ATR/MACD's
    # implicit "starting point" every step and is what caused
    # forecasts to drift/detach late in the horizon).
    # =====================================================

    def _seed_indicator_state(self, original_df):
        state = {}

        if "EMA9" in self.indicators:
            state["ema9"] = float(original_df["ema9"].iloc[-1])

        if "EMA20" in self.indicators:
            state["ema20"] = float(original_df["ema20"].iloc[-1])

        if "EMA50" in self.indicators:
            state["ema50"] = float(original_df["ema50"].iloc[-1])

        if "RSI14" in self.indicators:
            _, avg_gain, avg_loss = compute_rsi(original_df["close"], period=RSI_PERIOD)
            state["rsi_avg_gain"] = float(avg_gain.iloc[-1])
            state["rsi_avg_loss"] = float(avg_loss.iloc[-1])
            state["rsi_prev_close"] = float(original_df["close"].iloc[-1])

        if "ATR14" in self.indicators:
            state["atr_prev"] = float(original_df["atr14"].iloc[-1])
            state["atr_prev_close"] = float(original_df["close"].iloc[-1])

        if "MACD" in self.indicators:
            _, ema_fast, ema_slow = compute_macd(
                original_df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL
            )
            state["macd_ema_fast"] = float(ema_fast.iloc[-1])
            state["macd_ema_slow"] = float(ema_slow.iloc[-1])
            state["macd_signal"] = float(original_df["macd_signal"].iloc[-1])

        if "VWAP" in self.indicators:
            typical_price_hist = (
                original_df["high"] + original_df["low"] + original_df["close"]
            ) / 3
            cumulative_pv_hist = (typical_price_hist * original_df["volume"]).cumsum()
            cumulative_volume_hist = original_df["volume"].cumsum()

            state["vwap_cum_pv"] = float(cumulative_pv_hist.iloc[-1])
            state["vwap_cum_vol"] = float(cumulative_volume_hist.iloc[-1])

        return state

    def _advance_indicator_state(self, raw_window, state):
        """
        Advances every selected indicator by exactly one recursive
        step, using the newly appended synthetic bar (raw_window's
        last row), and writes that step's value into raw_window.
        Mirrors the exact formulas in compute_rsi / compute_atr /
        compute_macd / compute_vwap above.
        """
        idx = raw_window.index[-1]
        last_row = raw_window.iloc[-1]

        if "EMA9" in self.indicators:
            alpha = 2 / (9 + 1)
            state["ema9"] = alpha * last_row["close"] + (1 - alpha) * state["ema9"]
            raw_window.loc[idx, "ema9"] = state["ema9"]

        if "EMA20" in self.indicators:
            alpha = 2 / (20 + 1)
            state["ema20"] = alpha * last_row["close"] + (1 - alpha) * state["ema20"]
            raw_window.loc[idx, "ema20"] = state["ema20"]

        if "EMA50" in self.indicators:
            alpha = 2 / (50 + 1)
            state["ema50"] = alpha * last_row["close"] + (1 - alpha) * state["ema50"]
            raw_window.loc[idx, "ema50"] = state["ema50"]

        if "RSI14" in self.indicators:
            delta = last_row["close"] - state["rsi_prev_close"]
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)

            alpha = 1 / RSI_PERIOD
            state["rsi_avg_gain"] = alpha * gain + (1 - alpha) * state["rsi_avg_gain"]
            state["rsi_avg_loss"] = alpha * loss + (1 - alpha) * state["rsi_avg_loss"]
            state["rsi_prev_close"] = float(last_row["close"])

            if state["rsi_avg_loss"] == 0:
                rsi_val = 100.0
            else:
                rs = state["rsi_avg_gain"] / state["rsi_avg_loss"]
                rsi_val = 100 - (100 / (1 + rs))

            raw_window.loc[idx, "rsi14"] = rsi_val

        if "ATR14" in self.indicators:
            prev_close = state["atr_prev_close"]
            true_range = max(
                last_row["high"] - last_row["low"],
                abs(last_row["high"] - prev_close),
                abs(last_row["low"] - prev_close),
            )

            alpha = 1 / ATR_PERIOD
            state["atr_prev"] = alpha * true_range + (1 - alpha) * state["atr_prev"]
            state["atr_prev_close"] = float(last_row["close"])

            raw_window.loc[idx, "atr14"] = state["atr_prev"]

        if "MACD" in self.indicators:
            af = 2 / (MACD_FAST + 1)
            as_ = 2 / (MACD_SLOW + 1)
            asig = 2 / (MACD_SIGNAL + 1)

            state["macd_ema_fast"] = af * last_row["close"] + (1 - af) * state["macd_ema_fast"]
            state["macd_ema_slow"] = as_ * last_row["close"] + (1 - as_) * state["macd_ema_slow"]

            macd_line = state["macd_ema_fast"] - state["macd_ema_slow"]
            state["macd_signal"] = asig * macd_line + (1 - asig) * state["macd_signal"]

            raw_window.loc[idx, "macd_line"] = macd_line
            raw_window.loc[idx, "macd_signal"] = state["macd_signal"]
            raw_window.loc[idx, "macd_hist"] = macd_line - state["macd_signal"]

        if "VWAP" in self.indicators:
            typical_price = (last_row["high"] + last_row["low"] + last_row["close"]) / 3

            state["vwap_cum_pv"] += typical_price * last_row["volume"]
            state["vwap_cum_vol"] += last_row["volume"]

            new_vwap = (
                state["vwap_cum_pv"] / state["vwap_cum_vol"]
                if state["vwap_cum_vol"]
                else np.nan
            )

            raw_window.loc[idx, "vwap"] = new_vwap

        return raw_window, state

    # =====================================================
    # PREDICTION
    # =====================================================

    def predict(self, df, future_steps=1):
        """
        Runs the dual-head model and returns:

            {
                "signal": "BUY" / "HOLD" / "SELL",
                "confidence": float,
                "probabilities": {"SELL": float, "HOLD": float, "BUY": float},
                "predicted_return": float,
                "predicted_price": float,
                "future_times": [...],
                "future_predictions": [...],
            }

        "signal"/"confidence"/"probabilities"/"predicted_return"/
        "predicted_price" always describe the FIRST forecast step.
        "future_times"/"future_predictions" carry the full recursive
        path (length == future_steps) for backward compatibility with
        callers that consumed the old (future_times, future_predictions)
        tuple shape.

        Multi-step (future_steps > 1) recursive forecasting is disabled
        whenever the checkpoint depends on options features, because
        future options-chain values cannot be fabricated -- see item 12.
        """

        if future_steps > 1 and self.options_columns:
            raise RuntimeError(
                "Multi-step recursive forecasting is disabled for this "
                f"checkpoint: it depends on {len(self.options_columns)} "
                "options features, and future options-chain values are "
                "not available to synthesize. Call predict() with "
                "future_steps=1 for a single-step forecast."
            )

        original_df = df.copy()
        original_df = add_missing_features(original_df, self.indicators)

        self._validate_features_present(original_df)

        scaled_df = original_df.copy()
        scaled_df[self.features] = self.feature_scaler.transform(
            original_df[self.features]
        )

        current_seq = (
            scaled_df[self.features]
            .iloc[-self.sequence_length:]
            .values
            .astype(np.float32)
        )

        raw_cols = list(dict.fromkeys(
            ["open", "high", "low", "close", "volume"] + self.features
        ))

        raw_window = (
            original_df[raw_cols]
            .iloc[-self.sequence_length:]
            .copy()
        )

        # Seed recursive state for every selected indicator from the
        # FULL real history (original_df), before it's truncated into
        # raw_window.
        indicator_state = self._seed_indicator_state(original_df)

        future_predictions = []
        future_times = []

        last_time = original_df["datetime"].iloc[-1]

        recent_raw_highs = original_df["high"].iloc[-14:].values
        recent_raw_lows = original_df["low"].iloc[-14:].values
        forecast_atr = np.mean(recent_raw_highs - recent_raw_lows)

        current_close = float(original_df["close"].iloc[-1])
        max_step_move = forecast_atr * MAX_STEP_MOVE_ATR_MULTIPLE

        last_volume = original_df["volume"].iloc[-1]

        first_signal = None
        first_confidence = None
        first_probabilities = None
        first_predicted_return = None
        first_predicted_price = None

        with torch.no_grad():

            for step in range(future_steps):

                seq_tensor = (
                    torch.tensor(current_seq, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                predicted_return_tensor, direction_logits = self.model(seq_tensor)

                predicted_return = self._decode_return(
                    predicted_return_tensor.cpu().numpy()[0]
                )

                signal, confidence, probabilities = self._decode_direction(
                    direction_logits
                )

                predicted_price = current_close * (1.0 + predicted_return)

                if max_step_move > 0:
                    delta = predicted_price - current_close
                    if abs(delta) > max_step_move:
                        predicted_price = (
                            current_close + np.sign(delta) * max_step_move
                        )
                        # Keep predicted_return consistent with the
                        # ATR-capped price rather than reporting an
                        # uncapped return alongside a capped price.
                        if current_close:
                            predicted_return = (
                                predicted_price - current_close
                            ) / current_close

                predicted_price = self._ensure_finite(predicted_price, "predicted_price")
                predicted_return = self._ensure_finite(predicted_return, "predicted_return")

                current_close = predicted_price

                next_time = last_time + pd.Timedelta(minutes=1)
                last_time = next_time

                future_predictions.append(predicted_price)
                future_times.append(next_time)

                if step == 0:
                    first_signal = signal
                    first_confidence = confidence
                    first_probabilities = probabilities
                    first_predicted_return = predicted_return
                    first_predicted_price = predicted_price

                if step < future_steps - 1:
                    # Only reachable when self.options_columns is empty
                    # (see the guard above) -- safe to synthesize the
                    # next OHLC bar and roll price-based indicators
                    # forward by one real recursive step.
                    synth_high = predicted_price + forecast_atr * 0.5
                    synth_low = predicted_price - forecast_atr * 0.5

                    new_row = pd.DataFrame([{
                        "open": predicted_price,
                        "high": synth_high,
                        "low": synth_low,
                        "close": predicted_price,
                        "volume": last_volume,
                    }])

                    raw_window = pd.concat([raw_window, new_row], ignore_index=True)
                    raw_window = raw_window.iloc[-self.sequence_length:].reset_index(drop=True)

                    raw_window, indicator_state = self._advance_indicator_state(
                        raw_window, indicator_state
                    )

                    current_seq = (
                        self.feature_scaler
                        .transform(raw_window[self.features])
                        .astype(np.float32)
                    )

        result = {
            "signal": first_signal,
            "confidence": first_confidence,
            "probabilities": first_probabilities,
            "predicted_return": first_predicted_return,
            "predicted_price": first_predicted_price,
            "future_times": future_times,
            "future_predictions": future_predictions,
        }

        return self._sanitize_for_json(result)