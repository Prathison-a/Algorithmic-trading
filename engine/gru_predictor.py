import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from train.GRU_Training import (
    GRURegressor,
    add_indicators,
    resolve_feature_columns,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUPredictor:

    # ============================================================
    # CONFIGURATION
    # ============================================================

    # Must match GRU training
    DEFAULT_SEQUENCE_LENGTH = 60

    # Light prediction-line smoothing
    # 0.25 = light smoothing
    # Higher = more responsive / less smoothing
    # Lower = smoother / more lag
    SMOOTHING_ALPHA = 0.25

    def __init__(self, symbol="SPY"):

        symbol = symbol.upper()

        model_path = (
            Path(__file__).parent.parent
            / "models"
            / f"best_gru_{symbol.lower()}.pth"
        )

        print(f"Loading GRU Model: {model_path}")

        if not model_path.exists():
            raise FileNotFoundError(
                f"GRU model not found: {model_path}"
            )

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=False,
        )

        # ========================================================
        # CHECKPOINT CONFIG
        # ========================================================

        self.features = checkpoint["features"]

        self.indicators = checkpoint.get(
            "indicators",
            [],
        )

        self.indicator_columns = checkpoint.get(
            "indicator_columns",
            [],
        )

        self.sequence_length = checkpoint.get(
            "sequence_length",
            self.DEFAULT_SEQUENCE_LENGTH,
        )

        self.target_columns = checkpoint.get(
            "target_columns",
            ["target_next_return"],
        )

        # --------------------------------------------------------
        # TARGET TYPE
        #
        # Current GRU model:
        #
        # target_next_return =
        # (next_open - current_close) / current_close
        #
        # Therefore:
        #
        # predicted_price =
        # current_close * (1 + predicted_return)
        #
        # Older checkpoints may contain "price".
        # --------------------------------------------------------

        self.target_type = checkpoint.get(
            "target_type",
            "price",
        )

        # ========================================================
        # VALIDATE FEATURE CONFIGURATION
        # ========================================================

        expected_features = resolve_feature_columns(
            self.indicators
        )

        if expected_features != self.features:

            raise RuntimeError(
                "Checkpoint feature configuration does not match "
                "current GRU_Training.py.\n"
                f"Checkpoint features: {self.features}\n"
                f"Expected features:   {expected_features}\n"
                "Retrain the GRU model."
            )

        # ========================================================
        # MODEL
        # ========================================================

        self.model = GRURegressor(
            n_features=len(self.features)
        ).to(DEVICE)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        # ========================================================
        # FEATURE SCALER
        # ========================================================

        self.feature_scaler = StandardScaler()

        self.feature_scaler.mean_ = np.asarray(
            checkpoint["feature_scaler_mean"],
            dtype=np.float64,
        )

        self.feature_scaler.scale_ = np.asarray(
            checkpoint["feature_scaler_scale"],
            dtype=np.float64,
        )

        self.feature_scaler.n_features_in_ = len(
            self.features
        )

        self.feature_scaler.var_ = (
            self.feature_scaler.scale_ ** 2
        )

        # ========================================================
        # TARGET SCALER
        # ========================================================

        self.target_scaler = StandardScaler()

        self.target_scaler.mean_ = np.asarray(
            checkpoint["target_scaler_mean"],
            dtype=np.float64,
        )

        self.target_scaler.scale_ = np.asarray(
            checkpoint["target_scaler_scale"],
            dtype=np.float64,
        )

        self.target_scaler.n_features_in_ = len(
            self.target_columns
        )

        self.target_scaler.var_ = (
            self.target_scaler.scale_ ** 2
        )

        # ========================================================
        # INFORMATION
        # ========================================================

        print("========================================")
        print("GRU Predictor Loaded")
        print("========================================")
        print(f"Features       : {self.features}")
        print(f"Indicators     : {self.indicators}")
        print(f"Sequence Length: {self.sequence_length}")
        print(f"Target Columns : {self.target_columns}")
        print(f"Target Type    : {self.target_type}")
        print(
            f"Smoothing Alpha: {self.SMOOTHING_ALPHA}"
        )
        print("========================================")

    # ============================================================
    # FEATURE FRAME
    # ============================================================

    def _build_feature_frame(self, raw_df):

        df = raw_df.copy()

        if "datetime" not in df.columns:
            raise ValueError(
                "Input dataframe must contain 'datetime'."
            )

        df["datetime"] = pd.to_datetime(
            df["datetime"]
        )

        df = (
            df
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        required = [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        missing = [
            c
            for c in required
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        # Generate exactly the same indicators used during training
        df = add_indicators(
            df,
            self.indicators,
        )

        df = (
            df
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .reset_index(drop=True)
        )

        return df

    # ============================================================
    # TIMEFRAME
    # ============================================================

    def _infer_bar_interval(self, df):

        diffs = (
            pd.to_datetime(df["datetime"])
            .diff()
            .dropna()
        )

        if len(diffs) == 0:
            return pd.Timedelta(minutes=1)

        interval = diffs.median()

        if not isinstance(interval, pd.Timedelta):
            interval = pd.Timedelta(minutes=1)

        return interval

    # ============================================================
    # MODEL PREDICTION
    #
    # EXACT PIPELINE:
    #
    # 60 candles
    #      ↓
    # GRU
    #      ↓
    # scaled next-open return
    #      ↓
    # inverse scale
    #      ↓
    # real next-open return
    #      ↓
    # current close × (1 + return)
    #      ↓
    # predicted price
    # ============================================================

    def _predict_from_feature_window(
        self,
        feature_window,
    ):

        if len(feature_window) < self.sequence_length:
            raise RuntimeError(
                f"Need {self.sequence_length} candles, "
                f"but only {len(feature_window)} are available."
            )

        # --------------------------------------------------------
        # Take exactly the last 60 candles
        # --------------------------------------------------------

        window_df = (
            feature_window[
                self.features
            ]
            .iloc[-self.sequence_length:]
        )

        values = window_df.values.astype(
            np.float32
        )

        if len(values) != self.sequence_length:
            raise RuntimeError(
                f"Invalid sequence length: "
                f"{len(values)} != "
                f"{self.sequence_length}"
            )

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # Capture the REAL unscaled close BEFORE scaling.
        #
        # The return is:
        #
        # (next_open - current_close) / current_close
        #
        # Therefore current_close is the price anchor.
        # --------------------------------------------------------

        reference_close = float(
            feature_window["close"].iloc[-1]
        )

        if not np.isfinite(reference_close):
            raise RuntimeError(
                "Reference close is not finite."
            )

        if reference_close <= 0:
            raise RuntimeError(
                f"Invalid reference close: {reference_close}"
            )

        # --------------------------------------------------------
        # Scale the 60-candle input
        # --------------------------------------------------------

        scaled = self.feature_scaler.transform(
            values
        )

        tensor = (
            torch.tensor(
                scaled,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .to(DEVICE)
        )

        # --------------------------------------------------------
        # GRU PREDICTION
        # --------------------------------------------------------

        with torch.no_grad():

            output = self.model(tensor)

        pred_scaled = float(
            output
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)[0]
        )

        if not np.isfinite(pred_scaled):
            raise RuntimeError(
                "GRU returned a non-finite prediction."
            )

        # ========================================================
        # DECODE TARGET
        # ========================================================

        pred_value = (
            self.target_scaler
            .inverse_transform(
                np.array(
                    [[pred_scaled]],
                    dtype=np.float64,
                )
            )[0, 0]
        )

        if not np.isfinite(pred_value):
            raise RuntimeError(
                "Decoded GRU prediction is not finite."
            )

        # ========================================================
        # RETURN → PRICE
        # ========================================================

        if self.target_type == "return":

            predicted_return = float(
                pred_value
            )

            # ----------------------------------------------------
            # CORE CONVERSION:
            #
            # predicted price =
            # current close × (1 + predicted return)
            # ----------------------------------------------------

            predicted_price = (
                reference_close
                * (1.0 + predicted_return)
            )

        else:

            # Legacy absolute-price model
            predicted_return = (
                predicted_price / reference_close
            ) - 1.0

            predicted_price = float(
                pred_value
            )

        if not np.isfinite(predicted_price):
            raise RuntimeError(
                "Predicted price is not finite."
            )

        if predicted_price <= 0:
            raise RuntimeError(
                f"Invalid predicted price: "
                f"{predicted_price}"
            )

        return {
            "predicted_price": float(
                predicted_price
            ),
            "predicted_return": float(
                predicted_return
            ),
            "reference_close": float(
                reference_close
            ),
        }

    # ============================================================
    # LIGHT SMOOTHING
    #
    # EMA-style smoothing.
    #
    # alpha = 0.25
    #
    # new_smoothed =
    # alpha × prediction
    # +
    # (1-alpha) × previous_smoothed
    #
    # This is intentionally LIGHT.
    # ============================================================

    @classmethod
    def _smooth_prediction(
        cls,
        values,
        alpha=None,
    ):

        if values is None:
            return []

        if len(values) == 0:
            return []

        if alpha is None:
            alpha = cls.SMOOTHING_ALPHA

        alpha = float(alpha)

        if not 0.0 < alpha <= 1.0:
            raise ValueError(
                "Smoothing alpha must be between "
                "0 and 1."
            )

        cleaned = []

        for value in values:

            value = float(value)

            if not np.isfinite(value):
                if cleaned:
                    value = cleaned[-1]
                else:
                    continue

            cleaned.append(value)

        if len(cleaned) <= 1:
            return cleaned

        smoothed = [
            cleaned[0]
        ]

        previous = cleaned[0]

        for value in cleaned[1:]:

            current = (
                alpha * value
                +
                (1.0 - alpha) * previous
            )

            smoothed.append(
                float(current)
            )

            previous = current

        return smoothed

    # ============================================================
    # HISTORICAL PREDICTIONS
    #
    # For every historical candle:
    #
    # previous 60 candles
    #        ↓
    # GRU
    #        ↓
    # next-open return
    #        ↓
    # predicted price
    #
    # The prediction is attached to the NEXT candle timestamp.
    # ============================================================

    def historical_predictions(
        self,
        raw_df,
        smooth=False,
    ):

        df = self._build_feature_frame(
            raw_df
        )

        if len(df) <= self.sequence_length:
            return [], []

        times = []
        predictions = []

        for i in range(
            self.sequence_length,
            len(df),
        ):

            # Exactly 60 real candles
            window_df = (
                df.iloc[
                    i - self.sequence_length:i
                ]
            )

            result = (
                self._predict_from_feature_window(
                    window_df
                )
            )

            pred_price = result[
                "predicted_price"
            ]

            times.append(
                df["datetime"].iloc[i]
            )

            predictions.append(
                pred_price
            )

        # Optional standalone smoothing.
        #
        # get_prediction_line() deliberately calls
        # historical_predictions(smooth=False) so the historical
        # and future segments are smoothed together continuously.
        if smooth:

            predictions = (
                self._smooth_prediction(
                    predictions
                )
            )

        return times, predictions

    # ============================================================
    # FUTURE FORECAST
    #
    # Recursive:
    #
    # 60 candles
    #      ↓
    # GRU
    #      ↓
    # predicted return
    #      ↓
    # predicted price
    #      ↓
    # synthetic candle
    #      ↓
    # repeat
    #
    # This generates the future prediction line.
    # ============================================================

    def predict(
        self,
        raw_df,
        future_steps,
        smooth=False,
    ):

        if future_steps <= 0:
            return [], []

        raw = raw_df.copy()

        raw["datetime"] = pd.to_datetime(
            raw["datetime"]
        )

        raw = (
            raw
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        interval = (
            self._infer_bar_interval(raw)
        )

        # --------------------------------------------------------
        # Indicator warmup
        # --------------------------------------------------------

        indicator_warmup = 50

        history_needed = (
            self.sequence_length
            + indicator_warmup
            + 10
        )

        raw_window = (
            raw[
                [
                    "datetime",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            ]
            .iloc[-history_needed:]
            .copy()
            .reset_index(drop=True)
        )

        if len(raw_window) < self.sequence_length:
            raise ValueError(
                "Not enough historical data "
                "for GRU forecast."
            )

        # --------------------------------------------------------
        # REAL MARKET ANCHOR
        # --------------------------------------------------------

        last_real_close = float(
            raw_window["close"].iloc[-1]
        )

        last_volume = float(
            raw_window["volume"].iloc[-1]
        )

        # --------------------------------------------------------
        # ATR PROXY
        # Used only to prevent unstable recursive predictions.
        # --------------------------------------------------------

        recent_tr = (
            raw_window["high"]
            - raw_window["low"]
        )

        atr_proxy = float(
            recent_tr
            .rolling(14)
            .mean()
            .iloc[-1]
        )

        if (
            not np.isfinite(atr_proxy)
            or atr_proxy <= 0
        ):

            atr_proxy = max(
                last_real_close * 0.001,
                0.01,
            )

        last_time = (
            raw_window["datetime"].iloc[-1]
        )

        future_times = []
        future_predictions = []

        previous_prediction = (
            last_real_close
        )

        # ========================================================
        # RECURSIVE FORECAST
        # ========================================================

        for step in range(
            future_steps
        ):

            # ----------------------------------------------------
            # Recalculate indicators on the current raw window.
            #
            # This is important because the GRU was trained using
            # indicators. Every recursive step must rebuild them.
            # ----------------------------------------------------

            feature_window = add_indicators(
                raw_window.copy(),
                self.indicators,
            )

            feature_window = (
                feature_window
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
                .reset_index(drop=True)
            )

            if len(feature_window) < self.sequence_length:

                raise RuntimeError(
                    "Not enough valid feature rows "
                    "for future prediction. "
                    "Increase history_needed."
                )

            # ----------------------------------------------------
            # GRU:
            #
            # 60 candles
            #      ↓
            # predicted return
            #      ↓
            # predicted price
            # ----------------------------------------------------

            result = (
                self._predict_from_feature_window(
                    feature_window
                )
            )

            raw_predicted_price = result[
                "predicted_price"
            ]

            # ----------------------------------------------------
            # RECURSIVE SANITY LIMIT
            #
            # This is NOT the prediction-line smoothing.
            #
            # It prevents a single bad recursive prediction from
            # exploding the entire multi-step forecast.
            # ----------------------------------------------------

            max_change = max(
                atr_proxy * 2.0,
                last_real_close * 0.002,
            )

            lower_bound = (
                previous_prediction
                - max_change
            )

            upper_bound = (
                previous_prediction
                + max_change
            )

            pred_price = float(
                np.clip(
                    raw_predicted_price,
                    lower_bound,
                    upper_bound,
                )
            )

            # ----------------------------------------------------
            # FUTURE TIMESTAMP
            # ----------------------------------------------------

            next_time = (
                last_time + interval
            )

            last_time = next_time

            future_times.append(
                next_time
            )

            future_predictions.append(
                pred_price
            )

            # ----------------------------------------------------
            # SYNTHETIC FUTURE CANDLE
            #
            # We need a synthetic candle so that the next recursive
            # GRU prediction has OHLCV data available.
            # ----------------------------------------------------

            synthetic_range = max(
                atr_proxy * 0.50,
                last_real_close * 0.0005,
            )

            synth_open = (
                previous_prediction
            )

            synth_close = (
                pred_price
            )

            synth_high = (
                max(
                    synth_open,
                    synth_close,
                )
                + synthetic_range * 0.5
            )

            synth_low = (
                min(
                    synth_open,
                    synth_close,
                )
                - synthetic_range * 0.5
            )

            new_row = pd.DataFrame(
                [
                    {
                        "datetime": next_time,
                        "open": synth_open,
                        "high": synth_high,
                        "low": synth_low,
                        "close": synth_close,
                        "volume": last_volume,
                    }
                ]
            )

            raw_window = pd.concat(
                [
                    raw_window,
                    new_row,
                ],
                ignore_index=True,
            )

            raw_window = (
                raw_window
                .iloc[-history_needed:]
                .reset_index(drop=True)
            )

            previous_prediction = (
                pred_price
            )

        # --------------------------------------------------------
        # Optional standalone future smoothing
        #
        # get_prediction_line() uses smooth=False here and then
        # smooths historical + future together.
        # --------------------------------------------------------

        if smooth:

            future_predictions = (
                self._smooth_prediction(
                    future_predictions
                )
            )

        return (
            future_times,
            future_predictions,
        )

    # ============================================================
    # FULL PREDICTION LINE
    #
    # FINAL PIPELINE:
    #
    # Historical:
    #
    # 60 candles
    #      ↓
    # GRU
    #      ↓
    # next-open % return
    #      ↓
    # predicted price
    #
    # Future:
    #
    # 60 candles
    #      ↓
    # GRU
    #      ↓
    # next-open % return
    #      ↓
    # predicted price
    #      ↓
    # recursive forecast
    #
    # Then:
    #
    # historical prices + future prices
    #      ↓
    # ONE light EMA smoothing pass
    #      ↓
    # prediction line
    # ============================================================

    def get_prediction_line(
        self,
        raw_df,
        future_steps,
        smooth=True,
    ):

        # --------------------------------------------------------
        # HISTORICAL
        #
        # IMPORTANT:
        # smooth=False here.
        # --------------------------------------------------------

        historical_times, historical_predictions = (
            self.historical_predictions(
                raw_df,
                smooth=False,
            )
        )

        # --------------------------------------------------------
        # FUTURE
        #
        # IMPORTANT:
        # smooth=False here.
        # --------------------------------------------------------

        future_times, future_predictions = (
            self.predict(
                raw_df,
                future_steps,
                smooth=False,
            )
        )

        # --------------------------------------------------------
        # COMBINE BEFORE SMOOTHING
        #
        # This avoids a visible smoothing reset at the boundary.
        # --------------------------------------------------------

        all_times = (
            list(historical_times)
            +
            list(future_times)
        )

        all_predictions = (
            list(historical_predictions)
            +
            list(future_predictions)
        )

        # --------------------------------------------------------
        # ONE LIGHT SMOOTHING PASS
        # --------------------------------------------------------

        if smooth:

            all_predictions = (
                self._smooth_prediction(
                    all_predictions,
                    alpha=self.SMOOTHING_ALPHA,
                )
            )

        # --------------------------------------------------------
        # HISTORICAL/FUTURE BOUNDARY
        # --------------------------------------------------------

        boundary_index = (
            len(historical_times)
        )

        return (
            all_times,
            all_predictions,
            boundary_index,
        )