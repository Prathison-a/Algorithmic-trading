import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import random
from ftplib import FTP
from io import BytesIO

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# REPRODUCIBILITY
# =========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing Device: {DEVICE}")


# =========================================================
# FTP CONFIG
# =========================================================

FTP_HOST = "ftp.cavedspace.com"
FTP_USER = "ftpuser"
FTP_PASS = "TestUser01!!"
FTP_ROOT = "/Backup"

DEFAULT_SYMBOL = "SPY"


# =========================================================
# TRAINING CONFIG
# =========================================================

SEQUENCE_LENGTH = 60
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-6
PATIENCE = 10
DROPOUT_RATE = 0.2

RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# GRU architecture config (pipeline: 60 candles -> GRU -> % return + direction)
GRU_HIDDEN_SIZE = 128
GRU_NUM_LAYERS = 2

MULTI_STEP_VALIDATION_STEPS = 10
FORECAST_STEPS = MULTI_STEP_VALIDATION_STEPS

DEFAULT_INDICATORS = [
    "EMA9",
    "EMA20",
    "EMA50",
    "VWAP",
    "RSI14",
    "ATR14",
    "MACD",
]

BASE_FEATURES = [
    "high",
    "low",
    "close",
    "volume",
    "open",
]

# Fixed positions of close/open within BASE_FEATURES -- used both here
# and by the predictor to unscale/rescale price when converting a
# predicted % return back into a price, and back into a synthetic bar.
CLOSE_FEATURE_INDEX = 2
OPEN_FEATURE_INDEX = 4

INDICATOR_COLUMN_MAP = {
    "EMA9":  ["ema9"],
    "EMA20": ["ema20"],
    "EMA50": ["ema50"],
    "VWAP":  ["vwap"],
    "RSI14": ["rsi14"],
    "ATR14": ["atr14"],
    "MACD":  ["macd_line", "macd_signal", "macd_hist"],
}

# ---------------------------------------------------------
# TARGETS
#
# Two targets are produced from the same underlying quantity:
#
#   1) target_next_return (regression)
#        (next_open - current_close) / current_close
#        Kept centered near zero and scale-independent from the
#        instrument's price level -- important for the recursive
#        multi-step forecaster below.
#
#   2) target_direction (classification)
#        A discretized BUY/HOLD/SELL label derived from
#        target_next_return using DIRECTION_THRESHOLD:
#            SELL = 0   (return < -DIRECTION_THRESHOLD)
#            HOLD = 1   (|return| <= DIRECTION_THRESHOLD)
#            BUY  = 2   (return >  DIRECTION_THRESHOLD)
#
# The regression target continues to be StandardScaler-transformed
# via target_scaler. The direction target is NEVER scaled -- it stays
# an integer class label so it can be fed directly into
# CrossEntropyLoss.
# ---------------------------------------------------------
RETURN_TARGET_COLUMN = "target_next_return"
DIRECTION_TARGET_COLUMN = "target_direction"

DIRECTION_THRESHOLD = 0.0015

# SELL = 0, HOLD = 1, BUY = 2
DIRECTION_LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}

# Minimum softmax probability required before a raw BUY/SELL signal is
# trusted -- anything under this confidence is reported as HOLD.
MIN_SIGNAL_CONFIDENCE = 0.70

TARGET_COLUMNS = [
    RETURN_TARGET_COLUMN,
    DIRECTION_TARGET_COLUMN,
]
TARGET_TYPE = "return"   # decoded by the predictor: price = close * (1 + return)

feature_scaler = StandardScaler()
target_scaler = StandardScaler()   # fit ONLY on target_next_return (the direction
                                    # target is never passed through this scaler)


# =========================================================
# INDICATOR NAME NORMALIZATION
# =========================================================

def normalize_indicator_name(name):
    cleaned = str(name).strip().upper()
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = " ".join(cleaned.split())
    no_space = cleaned.replace(" ", "")

    for canonical in INDICATOR_COLUMN_MAP.keys():
        if canonical.upper().replace(" ", "") == no_space:
            return canonical

    raise ValueError(
        f"Unknown indicator '{name}'. This pipeline only supports: "
        f"{list(INDICATOR_COLUMN_MAP.keys())}"
    )


def normalize_indicator_list(indicators):
    normalized = []
    for raw_name in indicators:
        canonical = normalize_indicator_name(raw_name)
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


# =========================================================
# INDICATORS (unchanged from prior pipeline)
# =========================================================

def compute_rsi(close, period=RSI_PERIOD):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def compute_atr(dataframe, period=ATR_PERIOD):
    high_low = dataframe["high"] - dataframe["low"]
    high_close = (dataframe["high"] - dataframe["close"].shift()).abs()
    low_close = (dataframe["low"] - dataframe["close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def compute_vwap(dataframe):
    typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
    cumulative_pv = (typical_price * dataframe["volume"]).cumsum()
    cumulative_volume = dataframe["volume"].cumsum().replace(0, np.nan)

    return cumulative_pv / cumulative_volume


def compute_macd(dataframe, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    ema_fast = dataframe["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = dataframe["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    return pd.DataFrame({
        "macd_line":   macd_line,
        "macd_signal": signal_line,
        "macd_hist":   macd_line - signal_line,
    })


INDICATOR_REGISTRY = {
    "EMA9":  lambda df: df["close"].ewm(span=9, adjust=False).mean(),
    "EMA20": lambda df: df["close"].ewm(span=20, adjust=False).mean(),
    "EMA50": lambda df: df["close"].ewm(span=50, adjust=False).mean(),
    "VWAP":  lambda df: compute_vwap(df),
    "RSI14": lambda df: compute_rsi(df["close"], period=RSI_PERIOD),
    "ATR14": lambda df: compute_atr(df, period=ATR_PERIOD),
    "MACD":  lambda df: compute_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL),
}


def add_indicators(dataframe, indicators):
    dataframe = dataframe.copy()

    for name in indicators:
        if name not in INDICATOR_REGISTRY:
            raise ValueError(f"Unknown indicator: {name}")

        result = INDICATOR_REGISTRY[name](dataframe)

        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                dataframe[col] = result[col]
        else:
            dataframe[INDICATOR_COLUMN_MAP[name][0]] = result

    return dataframe


def resolve_indicator_columns(indicators):
    columns = []
    for name in indicators:
        columns.extend(INDICATOR_COLUMN_MAP[name])
    return columns


def resolve_feature_columns(indicators):
    """
    FINAL_FEATURES = BASE_FEATURES + Selected_Indicator_Columns

    NOTE: this does NOT include options columns. Options columns are
    data-dependent (they come from whatever expiry1_/expiry2_/expiry3_
    columns are actually present in the loaded dataframe), so they are
    resolved separately by resolve_options_columns() and appended on
    top of this list inside prepare_training_data().
    """
    return list(BASE_FEATURES) + resolve_indicator_columns(indicators)


def resolve_options_columns(dataframe):
    """
    Returns every numeric column in `dataframe` whose name starts with
    expiry1_, expiry2_, expiry3_ -- grouped in that order (all
    expiry1_* columns first, then expiry2_*, then expiry3_*), with
    each group's original column order preserved.

    This keeps the final feature order exactly:
        BASE_FEATURES + indicator_columns + expiry1_* + expiry2_* + expiry3_*
    as required, and never invents or predicts options values -- it
    only reads columns that are already present in the source data.
    """
    prefixes = ("expiry1_", "expiry2_", "expiry3_")
    options_columns = []

    for prefix in prefixes:
        group = [
            column for column in dataframe.columns
            if column.startswith(prefix)
            and pd.api.types.is_numeric_dtype(dataframe[column])
        ]
        options_columns.extend(group)

    return options_columns


# =========================================================
# PREDICTION TARGETS -- % return, plus a discretized BUY/HOLD/SELL
# direction label derived from that same % return.
# =========================================================

def add_targets(dataframe):
    """
    target_next_return = (next_open - current_close) / current_close

    This is the % move from "where we are now" (this bar's close) to
    "where the next bar opens" -- the quantity the GRU's return head
    is trained to predict. Converting back to price at inference time
    is just:
        predicted_price = current_close * (1 + predicted_return)

    target_direction is derived from target_next_return using
    DIRECTION_THRESHOLD:
        SELL = 0   if target_next_return < -DIRECTION_THRESHOLD
        BUY  = 2   if target_next_return >  DIRECTION_THRESHOLD
        HOLD = 1   otherwise

    Rows where target_next_return is NaN (the final row, from the
    shift(-1)) default to HOLD here and are subsequently dropped by
    the dropna() step in prepare_training_data(), so they never reach
    training.
    """
    dataframe = dataframe.copy()

    next_open = dataframe["open"].shift(-1)
    dataframe["target_next_return"] = (next_open - dataframe["close"]) / dataframe["close"]

    dataframe["target_direction"] = 1
    dataframe.loc[dataframe["target_next_return"] > DIRECTION_THRESHOLD, "target_direction"] = 2
    dataframe.loc[dataframe["target_next_return"] < -DIRECTION_THRESHOLD, "target_direction"] = 0
    dataframe["target_direction"] = dataframe["target_direction"].astype(np.int64)

    return dataframe


# =========================================================
# FTP DATA LOADING (unchanged)
# =========================================================

def get_file_name(symbol):
    return f"OHLCV{symbol}.txt"


def connect_ftp():
    ftp = FTP(FTP_HOST, timeout=60)
    ftp.login(user=FTP_USER, passwd=FTP_PASS)
    ftp.set_pasv(True)
    return ftp


def load_symbol_data(symbol, data_source="ftp", data_path=""):
    if data_source == "local":
        print(f"Training on local options data: {data_path}")
        import sys
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from data.options_loader import load_local_options
        df = load_local_options(data_path)
        return df

    first_ftp = connect_ftp()
    first_ftp.cwd(FTP_ROOT)

    folders = first_ftp.nlst()

    try:
        first_ftp.quit()
    except:
        first_ftp.close()

    date_folders = sorted([f for f in folders if f.startswith("202")])
    usable_folders = date_folders[:-1]
    all_dataframes = []

    for folder in usable_folders:
        ftp = None

        try:
            print(f"\nLoading {folder}")

            ftp = connect_ftp()

            folder_path = f"{FTP_ROOT}/{folder}/{symbol}"
            file_name = get_file_name(symbol)
            file_path = f"{folder_path}/{file_name}"

            buffer = BytesIO()
            ftp.retrbinary(f"RETR {file_path}", buffer.write)
            buffer.seek(0)

            dataframe = pd.read_csv(buffer)
            all_dataframes.append(dataframe)

            print("Loaded Successfully")

        except Exception as error:
            print(f"Skipping {folder}: {error}")

        finally:
            if ftp is not None:
                try:
                    ftp.quit()
                except:
                    try:
                        ftp.close()
                    except:
                        pass

    if not all_dataframes:
        raise RuntimeError(f"No data found for symbol {symbol}")

    return pd.concat(all_dataframes, ignore_index=True)


# =========================================================
# SEQUENCE CREATION (unchanged mechanics -- targets now include both
# the return column and the direction column, in that order)
# =========================================================

def create_sequences(dataframe, features, targets=TARGET_COLUMNS):
    feature_array = dataframe[features].to_numpy(dtype=np.float32)
    target_array = dataframe[targets].to_numpy(dtype=np.float32)

    X = []
    y = []

    for index in range(SEQUENCE_LENGTH, len(dataframe)):
        X.append(feature_array[index - SEQUENCE_LENGTH:index])
        y.append(target_array[index - 1])

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32)
    )


class MarketDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# =========================================================
# TRAINING CONFIGURATION BANNER
# =========================================================

def print_training_summary(
    symbol,
    indicators,
    indicator_columns,
    features,
    data_source="ftp",
    options_columns=None
):
    options_columns = options_columns or []

    print("=========================================")
    print("Training Configuration")
    print("=========================================")

    print("Base Features:")
    print(BASE_FEATURES)

    print("\nSelected Training Indicators:")
    print(indicators if indicators else [])

    print("\nGenerated Indicator Columns:")
    print(indicator_columns if indicator_columns else [])

    if data_source == "local":
        print("\nData Source:")
        print("Local Combined OHLCV + Options")

    # ---- Explicit feature-count breakdown. Counts are always the
    # REAL counts actually found/used (never hardcoded to 5/9/102/116),
    # so this stays correct regardless of which indicators are
    # selected or how many options columns the source data contains. ----
    print("\nOHLCV Feature Count:")
    print(len(BASE_FEATURES))

    print("\nIndicator Feature Count:")
    print(len(indicator_columns) if indicator_columns else 0)

    print("\nOptions Feature Count:")
    print(len(options_columns))

    print("\nTotal Model Feature Count:")
    print(len(features))

    print("\nFinal Training Features:")
    print(features)

    print("\nTotal Features:")
    print(len(features))

    print("\nTargets:")
    print(f"Return Target    : {RETURN_TARGET_COLUMN} (type={TARGET_TYPE}, scaled=True)")
    print(f"Direction Target : {DIRECTION_TARGET_COLUMN} "
          f"(threshold={DIRECTION_THRESHOLD}, scaled=False, labels={DIRECTION_LABELS})")

    print("\nSequence Length:")
    print(SEQUENCE_LENGTH)

    print("\nForecast Steps:")
    print(FORECAST_STEPS)

    print("=========================================")
    print(f"Model  : GRU (dual-head: return + direction)")
    print(f"Symbol : {symbol}")
    print("=========================================")
    print("Training Started...")
    print("=========================================\n")


# =========================================================
# PREPARE TRAINING DATA
# =========================================================

def prepare_training_data(symbol, indicators, data_source="ftp", data_path=""):
    dataframe = load_symbol_data(symbol, data_source, data_path)

    print(f"\nMerged Shape: {dataframe.shape}")

    dataframe.columns = [c.lower().strip() for c in dataframe.columns]

    try:
        dataframe["datetime"] = pd.to_datetime(dataframe["datetime"], unit="ms")
    except:
        dataframe["datetime"] = pd.to_datetime(dataframe["datetime"])

    dataframe = dataframe.sort_values("datetime")
    dataframe.reset_index(drop=True, inplace=True)

    dataframe = add_indicators(dataframe, indicators)
    indicator_columns = resolve_indicator_columns(indicators)

    # ---- Options features: read-only, never predicted/invented.
    # Only expected (and required) for local combined OHLCV + options
    # training; for FTP OHLCV-only training this will naturally be
    # an empty list since no expiry1_/expiry2_/expiry3_ columns exist. ----
    options_columns = resolve_options_columns(dataframe)

    # Final feature order: BASE_FEATURES, then indicator columns, then
    # expiry1_* / expiry2_* / expiry3_* columns (never reordered).
    features = resolve_feature_columns(indicators) + options_columns

    if data_source == "local":
        if len(options_columns) == 0:
            raise RuntimeError(
                "Local combined training requested, but no expiry1_/expiry2_/expiry3_ "
                "options features were found."
            )

        if len(options_columns) != 102:
            raise RuntimeError(
                f"Expected 102 options features, found {len(options_columns)}"
            )

        if len(features) != 116:
            raise RuntimeError(
                f"Expected 116 total model features, found {len(features)}"
            )

    dataframe = add_targets(dataframe)

    dataframe.dropna(inplace=True)
    dataframe.reset_index(drop=True, inplace=True)

    print("\n========== DATASET DIAGNOSTICS ==========")
    print("Rows after dropna:", len(dataframe))
    print("Total features:", len(features))
    print("Options features:", len(options_columns))
    print("\nTarget Statistics (target_next_return):")
    print(dataframe["target_next_return"].describe())
    print("\nDirection Label Distribution (target_direction):")
    print(
        dataframe["target_direction"]
        .map(DIRECTION_LABELS)
        .value_counts()
    )
    print("\nOptions Feature Variation:")
    constant_options = [
        col for col in options_columns
        if dataframe[col].nunique() <= 1
    ]
    print("Constant options columns:", len(constant_options))
    if constant_options:
        print("Constant columns:")
        print(constant_options)
    print("\nOptions NaN count:")
    print(dataframe[options_columns].isna().sum().sum())
    print("==========================================")

    split_index = int(len(dataframe) * 0.80)

    train_dataframe = dataframe.iloc[:split_index].copy()
    test_dataframe = dataframe.iloc[split_index:].copy()

    train_dataframe[features] = feature_scaler.fit_transform(train_dataframe[features])
    test_dataframe[features] = feature_scaler.transform(test_dataframe[features])

    # ---- Only the return target is ever passed through target_scaler.
    # target_direction stays as an untouched integer class label. ----
    train_dataframe[[RETURN_TARGET_COLUMN]] = target_scaler.fit_transform(
        train_dataframe[[RETURN_TARGET_COLUMN]]
    )
    test_dataframe[[RETURN_TARGET_COLUMN]] = target_scaler.transform(
        test_dataframe[[RETURN_TARGET_COLUMN]]
    )

    X_train, y_train = create_sequences(train_dataframe, features)
    X_test, y_test = create_sequences(test_dataframe, features)

    if len(X_train) == 0:
        raise RuntimeError("Training dataset is empty")

    if len(X_test) == 0:
        raise RuntimeError("Validation dataset is empty")

    print(f"\nTrain Shape: {X_train.shape}")
    print(f"Test Shape : {X_test.shape}")

    train_dataset = MarketDataset(X_train, y_train)
    test_dataset = MarketDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return (
        train_loader,
        test_loader,
        feature_scaler,
        target_scaler,
        features,
        indicator_columns,
        options_columns
    )


# =========================================================
# GRU MODEL
#
# Pipeline: 60 candles -> GRU -> shared dense layer -> two heads:
#   - return_head:    1 value  (scaled % return of next-bar open
#                      relative to current close)
#   - direction_head: 3 logits (SELL / HOLD / BUY)
#
# n_features scales with however many indicators are selected (5 base
# OHLCV + whichever indicator columns are on = 14 with the full
# DEFAULT_INDICATORS set, since MACD alone expands to 3 columns),
# plus any expiry1_/expiry2_/expiry3_ options columns found in the
# source data when training on local combined OHLCV + options data
# (14 + 102 = 116). The GRU backbone itself (60-step sequence length,
# hidden size 128, 2 layers, dropout 0.2, single dense layer) is
# unchanged from the original single-head regressor -- only the final
# output layer is split into the two heads above.
# =========================================================

class GRURegressor(nn.Module):

    def __init__(self, n_features, hidden_size=GRU_HIDDEN_SIZE, num_layers=GRU_NUM_LAYERS):
        super().__init__()

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=DROPOUT_RATE if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.gelu = nn.GELU()

        # Dual heads, both fed from the same 64-unit dense layer.
        self.return_head = nn.Linear(64, 1)
        self.direction_head = nn.Linear(64, 3)

    def forward(self, x):
        output, _ = self.gru(x)

        # Last timestep's hidden state summarizes the full 60-candle window.
        output = output[:, -1, :]

        output = self.dropout(output)
        output = self.fc1(output)
        output = self.gelu(output)

        predicted_return = self.return_head(output)      # shape: (batch, 1)
        direction_logits = self.direction_head(output)   # shape: (batch, 3)

        return predicted_return, direction_logits


# =========================================================
# PREDICTION DECODING -- turns raw model outputs into the standard
# BUY/HOLD/SELL signal payload, applying the confidence floor.
# =========================================================

def decode_prediction(predicted_return, direction_logits, target_scaler):
    """
    predicted_return : tensor/array of shape (1, 1) -- scaled return
                        output from GRURegressor's return_head.
    direction_logits : tensor/array of shape (1, 3) -- raw logits
                        from GRURegressor's direction_head.
    target_scaler     : the StandardScaler fit on target_next_return.

    Returns:
        {
            "signal": "BUY" | "HOLD" | "SELL",
            "confidence": float,
            "probabilities": {"SELL": .., "HOLD": .., "BUY": ..},
            "predicted_return": float
        }

    The raw argmax class is only used as the signal when its softmax
    probability is >= MIN_SIGNAL_CONFIDENCE; otherwise the signal is
    forced to HOLD.
    """
    if isinstance(direction_logits, torch.Tensor):
        direction_logits_tensor = direction_logits.detach().cpu()
    else:
        direction_logits_tensor = torch.tensor(direction_logits, dtype=torch.float32)

    probabilities = torch.softmax(direction_logits_tensor, dim=-1).numpy().reshape(-1)

    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])

    raw_signal = DIRECTION_LABELS[predicted_class]
    signal = raw_signal if confidence >= MIN_SIGNAL_CONFIDENCE else "HOLD"

    if isinstance(predicted_return, torch.Tensor):
        scaled_return = predicted_return.detach().cpu().numpy().reshape(-1)[0]
    else:
        scaled_return = np.asarray(predicted_return).reshape(-1)[0]

    real_return = (scaled_return * target_scaler.scale_[0]) + target_scaler.mean_[0]

    return {
        "signal": signal,
        "confidence": round(confidence, 4),
        "probabilities": {
            "SELL": round(float(probabilities[0]), 4),
            "HOLD": round(float(probabilities[1]), 4),
            "BUY": round(float(probabilities[2]), 4),
        },
        "predicted_return": round(float(real_return), 6),
    }


# =========================================================
# METRICS
# =========================================================

train_losses = []
val_losses = []
mae_list = []
rmse_list = []
r2_list = []
corr_list = []
confidence_score_list = []
multi_step_mae_list = []

direction_accuracy_list = []
buy_precision_list = []
sell_precision_list = []
hold_precision_list = []
buy_recall_list = []
sell_recall_list = []
hold_recall_list = []
confusion_matrix_list = []


def calculate_confidence_score(residual_std, target_mean):
    if target_mean == 0:
        return 0.0

    confidence = 100 - (residual_std / abs(target_mean)) * 100
    confidence = max(0, min(100, confidence))

    return float(confidence)


def multi_step_validation(
    model,
    X_raw,
    y_raw,
    feature_scaler,
    target_scaler,
    steps=MULTI_STEP_VALIDATION_STEPS,
    options_columns=None
):
    """
    Rolls the model forward `steps` bars. Unlike the old price-target
    version, the model's return head output here is a % RETURN, not a
    price -- so each step has to:
      1. Decode the scaled return prediction -> real return
      2. Unscale the current window's last real close
      3. predicted_price = real_close * (1 + real_return)
      4. Rescale predicted_price back into feature space to write into
         the rolled-forward window's open/close columns

    Only the return head is used for this recursive rollout; the
    direction head has nothing to contribute here since there's no
    "future direction" feature to roll forward.

    IMPORTANT: this recursive rollout only ever overwrites the
    open/close slots of BASE_FEATURES. It has no way to produce future
    expiry1_/expiry2_/expiry3_ options values -- doing so would mean
    inventing fake future options-chain data. So whenever options
    features are part of the input (options_columns is non-empty),
    this function is skipped entirely and reports np.nan instead of
    silently reusing stale, copied-forward options values across the
    recursive horizon.
    """
    if options_columns:
        print(
            "Multi-step recursive validation skipped because future "
            "options features are unavailable."
        )
        return float(np.nan)

    model.eval()

    predictions = []
    targets = []

    sample_count = min(len(X_raw) - steps, 300)

    if sample_count <= 0:
        return 0.0

    return_mean = target_scaler.mean_[0]
    return_scale = target_scaler.scale_[0]

    close_mean = feature_scaler.mean_[CLOSE_FEATURE_INDEX]
    close_scale = feature_scaler.scale_[CLOSE_FEATURE_INDEX]

    open_mean = feature_scaler.mean_[OPEN_FEATURE_INDEX]
    open_scale = feature_scaler.scale_[OPEN_FEATURE_INDEX]

    with torch.no_grad():
        for i in range(sample_count):
            current_sequence = X_raw[i].copy()

            step_predictions = []

            for step in range(steps):
                seq_tensor = (
                    torch.tensor(current_sequence, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                predicted_return_tensor, _ = model(seq_tensor)
                pred_return_scaled = predicted_return_tensor.cpu().numpy()[0][0]

                step_predictions.append(pred_return_scaled)

                # ---- decode: scaled return -> real return -> real price ----
                real_return = (pred_return_scaled * return_scale) + return_mean

                last_close_scaled = current_sequence[-1][CLOSE_FEATURE_INDEX]
                real_close = (last_close_scaled * close_scale) + close_mean

                real_predicted_price = real_close * (1 + real_return)

                # ---- re-encode predicted price for open/close columns ----
                pred_open_scaled = (real_predicted_price - open_mean) / open_scale
                pred_close_scaled = (real_predicted_price - close_mean) / close_scale

                next_sequence = current_sequence.copy()
                next_sequence[:-1] = next_sequence[1:]

                next_sequence[-1] = next_sequence[-2]
                next_sequence[-1][OPEN_FEATURE_INDEX] = pred_open_scaled
                next_sequence[-1][CLOSE_FEATURE_INDEX] = pred_close_scaled

                current_sequence = next_sequence

            predictions.extend(step_predictions)
            targets.extend(y_raw[i:i + steps, 0])

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Compare in RETURN space -- this metric measures return-prediction
    # accuracy, matching what the return head actually optimizes.
    predictions_real = (predictions * return_scale) + return_mean
    targets_real = (targets * return_scale) + return_mean

    multi_mae = mean_absolute_error(targets_real, predictions_real)

    return float(multi_mae)


# =========================================================
# TRAINING FUNCTION
# =========================================================

def train_model(
    symbol=DEFAULT_SYMBOL,
    indicators=None,
    progress_callback=None,
    stop_callback=None,
    data_source="ftp",
    data_path=""
):
    if indicators is None:
        indicators = list(DEFAULT_INDICATORS)

    indicators = normalize_indicator_list(indicators)

    best_val_loss = np.inf
    patience_counter = 0

    train_losses.clear()
    val_losses.clear()
    mae_list.clear()
    rmse_list.clear()
    r2_list.clear()
    corr_list.clear()
    confidence_score_list.clear()
    multi_step_mae_list.clear()

    direction_accuracy_list.clear()
    buy_precision_list.clear()
    sell_precision_list.clear()
    hold_precision_list.clear()
    buy_recall_list.clear()
    sell_recall_list.clear()
    hold_recall_list.clear()
    confusion_matrix_list.clear()

    symbol = symbol.upper()

    MODEL_PATH = (
        Path(__file__).parent.parent
        / "models"
        / f"best_gru_{symbol.lower()}.pth"
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    (
        train_loader,
        test_loader,
        feature_scaler,
        target_scaler,
        features,
        indicator_columns,
        options_columns
    ) = prepare_training_data(symbol, indicators, data_source, data_path)

    options_present = len(options_columns) > 0

    print_training_summary(
        symbol,
        indicators,
        indicator_columns,
        features,
        data_source=data_source,
        options_columns=options_columns
    )

    return_loss_fn = nn.HuberLoss()
    direction_loss_fn = nn.CrossEntropyLoss()

    model = GRURegressor(n_features=len(features)).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}] Started...")

        model.train()
        running_train_loss = 0

        for X_batch, y_batch in train_loader:
            if stop_callback and stop_callback():
                print("\nTraining Stopped By User")
                return False

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            y_return = y_batch[:, 0:1]
            y_direction = y_batch[:, 1].long()

            optimizer.zero_grad()

            predicted_return, direction_logits = model(X_batch)

            return_loss_value = return_loss_fn(predicted_return, y_return)
            direction_loss_value = direction_loss_fn(direction_logits, y_direction)
            loss = return_loss_value + direction_loss_value

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()

        running_val_loss = 0
        all_return_predictions = []
        all_return_targets = []
        all_direction_logits = []
        all_direction_targets = []
        X_validation_raw = []
        y_validation_raw = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                y_return = y_batch[:, 0:1]
                y_direction = y_batch[:, 1].long()

                predicted_return, direction_logits = model(X_batch)

                return_loss_value = return_loss_fn(predicted_return, y_return)
                direction_loss_value = direction_loss_fn(direction_logits, y_direction)
                loss = return_loss_value + direction_loss_value

                running_val_loss += loss.item()

                all_return_predictions.extend(predicted_return.cpu().numpy())
                all_return_targets.extend(y_return.cpu().numpy())
                all_direction_logits.extend(direction_logits.cpu().numpy())
                all_direction_targets.extend(y_direction.cpu().numpy())

                X_validation_raw.extend(X_batch.cpu().numpy())
                y_validation_raw.extend(y_batch.cpu().numpy())

        avg_val_loss = running_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        all_return_predictions = np.array(all_return_predictions)
        all_return_targets = np.array(all_return_targets)

        # ---- Unscale: RETURN space, not price ----
        return_pred_real = target_scaler.inverse_transform(all_return_predictions)[:, 0]
        return_true_real = target_scaler.inverse_transform(all_return_targets)[:, 0]

        prediction_std = np.std(return_pred_real)
        target_std = np.std(return_true_real)

        mae = mean_absolute_error(return_true_real, return_pred_real)
        rmse = np.sqrt(mean_squared_error(return_true_real, return_pred_real))
        r2 = r2_score(return_true_real, return_pred_real)
        corr = float(np.corrcoef(return_true_real, return_pred_real)[0, 1])

        residual_std = np.std(return_true_real - return_pred_real)

        confidence_score = calculate_confidence_score(
            residual_std, np.mean(return_true_real)
        )

        # ---- Direction classification metrics ----
        all_direction_logits = np.array(all_direction_logits)
        all_direction_targets = np.array(all_direction_targets)
        direction_pred_classes = np.argmax(all_direction_logits, axis=1)

        direction_accuracy = accuracy_score(all_direction_targets, direction_pred_classes)

        precision = precision_score(
            all_direction_targets, direction_pred_classes,
            labels=[0, 1, 2], average=None, zero_division=0
        )
        recall = recall_score(
            all_direction_targets, direction_pred_classes,
            labels=[0, 1, 2], average=None, zero_division=0
        )
        cm = confusion_matrix(
            all_direction_targets, direction_pred_classes, labels=[0, 1, 2]
        )

        sell_precision, hold_precision, buy_precision = precision[0], precision[1], precision[2]
        sell_recall, hold_recall, buy_recall = recall[0], recall[1], recall[2]

        # ---- One-step validation is always run, unchanged, above.
        # Multi-step (recursive) validation is skipped internally by
        # multi_step_validation() whenever options_columns is
        # non-empty, and returns np.nan in that case instead of
        # recursively fabricating future options features. ----
        multi_step_mae = multi_step_validation(
            model,
            np.array(X_validation_raw),
            np.array(y_validation_raw),
            feature_scaler,
            target_scaler,
            steps=MULTI_STEP_VALIDATION_STEPS,
            options_columns=options_columns
        )

        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        corr_list.append(corr)
        confidence_score_list.append(confidence_score)
        multi_step_mae_list.append(multi_step_mae)

        direction_accuracy_list.append(direction_accuracy)
        buy_precision_list.append(buy_precision)
        sell_precision_list.append(sell_precision)
        hold_precision_list.append(hold_precision)
        buy_recall_list.append(buy_recall)
        sell_recall_list.append(sell_recall)
        hold_recall_list.append(hold_recall)
        confusion_matrix_list.append(cm)

        print("\n========== RETURN DIAGNOSTICS (target_next_return) ==========")
        print("Prediction STD:", prediction_std)
        print("Target STD    :", target_std)
        print("Prediction Mean:", np.mean(return_pred_real))
        print("Target Mean    :", np.mean(return_true_real))
        print("Prediction Min :", np.min(return_pred_real))
        print("Prediction Max :", np.max(return_pred_real))
        print("Target Min     :", np.min(return_true_real))
        print("Target Max     :", np.max(return_true_real))
        print("======================================")

        print("\n========== DIRECTION DIAGNOSTICS (target_direction) ==========")
        print(f"Direction Accuracy  : {direction_accuracy:.4f}")
        print(f"BUY Precision       : {buy_precision:.4f}")
        print(f"SELL Precision      : {sell_precision:.4f}")
        print(f"HOLD Precision      : {hold_precision:.4f}")
        print(f"BUY Recall          : {buy_recall:.4f}")
        print(f"SELL Recall         : {sell_recall:.4f}")
        print(f"HOLD Recall         : {hold_recall:.4f}")
        print("Confusion Matrix (rows=true, cols=pred) [SELL, HOLD, BUY]:")
        print(cm)
        print("======================================")

        # ---- Safe formatting: multi_step_mae is np.nan whenever
        # options features are enabled, so it can't go through a
        # plain ":.6f}" format spec without raising. ----
        if np.isnan(multi_step_mae):
            multi_step_mae_display = "N/A (options features enabled)"
        else:
            multi_step_mae_display = f"{multi_step_mae:.6f}"

        print(f"Epoch [{epoch + 1}] Completed")
        print(f"Train Loss          : {avg_train_loss:.6f}")
        print(f"Val Loss            : {avg_val_loss:.6f}")
        print(f"MAE                 : {mae:.6f}")
        print(f"RMSE                : {rmse:.6f}")
        print(f"R\u00b2                  : {r2:.6f}")
        print(f"Correlation         : {corr:.6f}")
        print(f"Confidence Score    : {confidence_score:.2f}%")
        print(f"Multi-Step MAE      : {multi_step_mae_display}")
        print(f"Direction Accuracy  : {direction_accuracy:.4f}")
        print(f"BUY Precision       : {buy_precision:.4f}")
        print(f"SELL Precision      : {sell_precision:.4f}")
        print(f"HOLD Precision      : {hold_precision:.4f}")

        if progress_callback:
            progress_callback(epoch + 1, EPOCHS, avg_train_loss, avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": "GRU",
                    "features": features,
                    "indicators": indicators,
                    "indicator_columns": indicator_columns,
                    "sequence_length": SEQUENCE_LENGTH,
                    "forecast_steps": FORECAST_STEPS,

                    "target_columns": TARGET_COLUMNS,
                    "return_target_column": RETURN_TARGET_COLUMN,
                    "direction_target_column": DIRECTION_TARGET_COLUMN,
                    "direction_threshold": DIRECTION_THRESHOLD,
                    "direction_labels": DIRECTION_LABELS,
                    "min_signal_confidence": MIN_SIGNAL_CONFIDENCE,
                    "target_type": TARGET_TYPE,

                    "target_scaler_mean": target_scaler.mean_,
                    "target_scaler_scale": target_scaler.scale_,

                    "feature_scaler_mean": feature_scaler.mean_,
                    "feature_scaler_scale": feature_scaler.scale_,

                    "close_feature_index": CLOSE_FEATURE_INDEX,
                    "open_feature_index": OPEN_FEATURE_INDEX,

                    "residual_std": residual_std,
                    "confidence_score": confidence_score,
                    "multi_step_validation_mae": multi_step_mae,

                    "direction_accuracy": direction_accuracy,
                    "buy_precision": buy_precision,
                    "sell_precision": sell_precision,
                    "hold_precision": hold_precision,
                    "buy_recall": buy_recall,
                    "sell_recall": sell_recall,
                    "hold_recall": hold_recall,
                    "confusion_matrix": cm.tolist(),

                    "dropout_rate": DROPOUT_RATE,
                    "gru_hidden_size": GRU_HIDDEN_SIZE,
                    "gru_num_layers": GRU_NUM_LAYERS,
                    "multi_step_validation_steps": MULTI_STEP_VALIDATION_STEPS,

                    # ---- Options-feature metadata ----
                    "options_features_enabled": options_present,
                    "options_columns": options_columns,
                },
                MODEL_PATH
            )

            print("\nBest GRU Model Saved")

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly Stopping Triggered")
            return True

    print("\nGRU Training Complete.")
    return True


if __name__ == "__main__":
    train_model(
        symbol=DEFAULT_SYMBOL,
        indicators=DEFAULT_INDICATORS,
        data_source="local",
        data_path="data/OptionsDataAllDteSPY.jsonlines"
    )