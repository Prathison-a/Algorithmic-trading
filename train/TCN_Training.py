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

# How many bars ahead the multi-step rolling validation walks the
# model forward. Also reported in the training banner as
# "Forecast Steps".
MULTI_STEP_VALIDATION_STEPS = 10
FORECAST_STEPS = MULTI_STEP_VALIDATION_STEPS

# Expanded indicator set for this pipeline, per the architecture:
#   OHLCV + EMA9 + EMA20 + EMA50 + VWAP + RSI14 + ATR14 + MACD
#     -> TCN -> predicted next return + direction -> signal logic
#
# "Volume" is not listed as a separate indicator here because it is
# already part of BASE_FEATURES below (every bar's raw volume is
# always included as an input column, indicator selection or not).
DEFAULT_INDICATORS = [
    "EMA9",
    "EMA20",
    "EMA50",
    "VWAP",
    "RSI14",
    "ATR14",
    "MACD",
]

# Mandatory base OHLCV columns. Always included first, regardless of
# which indicators are selected. Keeping them fixed at indices [0..4]
# is what lets multi_step_validation() below safely assume "close" is
# index 2 and "open" is index 4 no matter which indicators are on.
BASE_FEATURES = [
    "high",
    "low",
    "close",
    "volume",
    "open",
]

# Maps each selectable indicator name -> the dataframe column(s) it
# produces.
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
# The model now has two heads:
#   1) a regression head predicting the next-bar return
#   2) a classification head predicting SELL / HOLD / BUY
# ---------------------------------------------------------

DIRECTION_THRESHOLD = 0.0015

DIRECTION_LABELS = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}

MIN_SIGNAL_CONFIDENCE = 0.70

TARGET_COLUMNS = [
    "target_next_return",
    "target_direction",
]

feature_scaler = StandardScaler()
target_scaler = StandardScaler()   # fit ONLY on target_next_return


# =========================================================
# INDICATOR NAME NORMALIZATION
# =========================================================

def normalize_indicator_name(name):
    """
    Turns whatever is passed in ("ema9", "EMA 9", etc.) into the
    exact canonical key used in INDICATOR_COLUMN_MAP / INDICATOR_REGISTRY.
    """
    cleaned = str(name).strip().upper()
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = " ".join(cleaned.split())          # collapse repeated spaces
    no_space = cleaned.replace(" ", "")

    for canonical in INDICATOR_COLUMN_MAP.keys():
        if canonical.upper().replace(" ", "") == no_space:
            return canonical

    raise ValueError(
        f"Unknown indicator '{name}'. This pipeline only supports: "
        f"{list(INDICATOR_COLUMN_MAP.keys())}"
    )


def normalize_indicator_list(indicators):
    """
    Normalizes a whole list of indicator names and removes duplicates
    while preserving the order they were given in.
    """
    normalized = []
    for raw_name in indicators:
        canonical = normalize_indicator_name(raw_name)
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


# =========================================================
# INDICATORS (EMA9 / EMA20 / EMA50 / VWAP / RSI14 / ATR14 / MACD)
# =========================================================

def compute_rsi(close, period=RSI_PERIOD):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def compute_atr(dataframe, period=ATR_PERIOD):
    """
    Average True Range: volatility measure derived from the True Range
    (max of high-low, |high-prev_close|, |low-prev_close|), smoothed
    with Wilder's EWM (alpha = 1/period).
    """
    high_low = dataframe["high"] - dataframe["low"]
    high_close = (dataframe["high"] - dataframe["close"].shift()).abs()
    low_close = (dataframe["low"] - dataframe["close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def compute_vwap(dataframe):
    """
    Volume-Weighted Average Price, cumulative over the full series:
    cumsum(typical_price * volume) / cumsum(volume), where
    typical_price = (high + low + close) / 3.
    """
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


# Registry: indicator name -> function that computes it from raw OHLCV.
# add_indicators() below only ever calls entries that were selected,
# so no unused indicator ever gets computed.
INDICATOR_REGISTRY = {

    "EMA9": lambda df: (
        df["close"].ewm(span=9, adjust=False).mean()
    ),

    "EMA20": lambda df: (
        df["close"].ewm(span=20, adjust=False).mean()
    ),

    "EMA50": lambda df: (
        df["close"].ewm(span=50, adjust=False).mean()
    ),

    "VWAP": lambda df: (
        compute_vwap(df)
    ),

    "RSI14": lambda df: (
        compute_rsi(df["close"], period=RSI_PERIOD)
    ),

    "ATR14": lambda df: (
        compute_atr(df, period=ATR_PERIOD)
    ),

    "MACD": lambda df: (
        compute_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    ),
}


def add_indicators(dataframe, indicators):
    """
    Computes ONLY the indicators requested (already normalized to
    their canonical names) and appends their column(s) to the
    dataframe. Nothing else is ever calculated here.
    """
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
    """
    Returns the flat list of generated indicator column names for the
    selected indicators, in selection order. E.g. for
    ["EMA9", "EMA20", "EMA50", "VWAP", "RSI14", "ATR14", "MACD"] ->
    ["ema9", "ema20", "ema50", "vwap", "rsi14", "atr14",
     "macd_line", "macd_signal", "macd_hist"]
    """
    columns = []
    for name in indicators:
        columns.extend(INDICATOR_COLUMN_MAP[name])
    return columns


def resolve_feature_columns(indicators):
    """
    FINAL_FEATURES = BASE_FEATURES + Selected_Indicator_Columns
    """
    return list(BASE_FEATURES) + resolve_indicator_columns(indicators)


def resolve_options_columns(dataframe):
    """
    The dataframe already contains options features using the
    expiry1_ / expiry2_ / expiry3_ prefixes. This simply picks out the
    numeric ones, preserving the dataframe's original column order.
    Nothing is sorted or reordered.
    """
    options_columns = [
        column
        for column in dataframe.columns
        if (
            column.startswith("expiry1_")
            or column.startswith("expiry2_")
            or column.startswith("expiry3_")
        )
        and pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    return options_columns


# =========================================================
# PREDICTION TARGETS
# =========================================================

def add_targets(dataframe):
    """
    Adds the two prediction targets:
      - target_next_return: forward return from close[t] to open[t+1]
      - target_direction:   SELL (0) / HOLD (1) / BUY (2), derived
        from target_next_return via DIRECTION_THRESHOLD
    """
    dataframe = dataframe.copy()

    next_open = dataframe["open"].shift(-1)

    dataframe["target_next_return"] = (
        next_open - dataframe["close"]
    ) / dataframe["close"]

    dataframe["target_direction"] = 1

    dataframe.loc[
        dataframe["target_next_return"] > DIRECTION_THRESHOLD,
        "target_direction"
    ] = 2

    dataframe.loc[
        dataframe["target_next_return"] < -DIRECTION_THRESHOLD,
        "target_direction"
    ] = 0

    dataframe["target_direction"] = (
        dataframe["target_direction"].astype(np.int64)
    )

    return dataframe


# =========================================================
# FTP DATA LOADING
# =========================================================

def get_file_name(symbol):
    return f"OHLCV{symbol}.txt"


def connect_ftp():
    ftp = FTP(
        FTP_HOST,
        timeout=60
    )

    ftp.login(
        user=FTP_USER,
        passwd=FTP_PASS
    )

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

    date_folders = sorted([
        folder for folder in folders
        if folder.startswith("202")
    ])

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

            ftp.retrbinary(
                f"RETR {file_path}",
                buffer.write
            )

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
        raise RuntimeError(
            f"No data found for symbol {symbol}"
        )

    return pd.concat(
        all_dataframes,
        ignore_index=True
    )


# =========================================================
# SEQUENCE CREATION
# =========================================================

def create_sequences(dataframe, features, targets=TARGET_COLUMNS):
    """
    Builds sliding windows of `features` as model input (X) and, for
    each window, the [target_next_return, target_direction] pair (y)
    that lines up with the bar right after the window ends.
    """
    feature_array = dataframe[features].to_numpy(dtype=np.float32)
    target_array = dataframe[targets].to_numpy(dtype=np.float32)

    X = []
    y = []

    for index in range(SEQUENCE_LENGTH, len(dataframe)):
        X.append(
            feature_array[index - SEQUENCE_LENGTH:index]
        )

        # The last bar the model actually sees is at position
        # `index - 1`, so the label for "the bar right after the
        # window" is target_array[index - 1].
        y.append(
            target_array[index - 1]
        )

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32)
    )


# =========================================================
# DATASET
# =========================================================

class MarketDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (
            self.X[index],
            self.y[index]
        )


# =========================================================
# TRAINING CONFIGURATION BANNER
# =========================================================

def print_training_summary(symbol, indicators, indicator_columns, features, options_columns):
    print("=========================================")
    print("Training Configuration")
    print("=========================================")

    print("Base Features:")
    print(BASE_FEATURES)

    print("\nSelected Training Indicators:")
    print(indicators if indicators else [])

    print("\nGenerated Indicator Columns:")
    print(indicator_columns if indicator_columns else [])

    print("\nFinal Training Features:")
    print(features)

    print("\nTotal Features:")
    print(len(features))

    print("\nTargets:")
    print(TARGET_COLUMNS)

    print("\nSequence Length:")
    print(SEQUENCE_LENGTH)

    print("\nForecast Steps:")
    print(FORECAST_STEPS)

    print("=========================================")
    print(f"Model              : TCN")
    print(f"Symbol             : {symbol}")
    print()
    print(f"Base Features      : {len(BASE_FEATURES)}")
    print(f"Indicator Features : {len(resolve_indicator_columns(indicators))}")
    print(f"Options Features   : {len(options_columns)}")
    print(f"Total Features     : {len(features)}")
    print()
    print(f"Sequence Length    : {SEQUENCE_LENGTH}")
    print()
    print("Targets:")
    print("- target_next_return")
    print("- target_direction")
    print()
    print("Direction Mapping:")
    print("SELL = 0")
    print("HOLD = 1")
    print("BUY  = 2")
    print()
    print(f"Direction Threshold: {DIRECTION_THRESHOLD}")
    print(f"Minimum Confidence : {MIN_SIGNAL_CONFIDENCE}")
    print("=========================================")
    print("Training Started...")
    print("=========================================\n")


# =========================================================
# PREPARE TRAINING DATA
# =========================================================

def prepare_training_data(symbol, indicators, data_source="ftp", data_path=""):
    dataframe = load_symbol_data(symbol, data_source, data_path)

    print(f"\nMerged Shape: {dataframe.shape}")

    dataframe.columns = [
        column.lower().strip()
        for column in dataframe.columns
    ]

    try:
        dataframe["datetime"] = pd.to_datetime(
            dataframe["datetime"],
            unit="ms"
        )
    except:
        dataframe["datetime"] = pd.to_datetime(
            dataframe["datetime"]
        )

    dataframe = dataframe.sort_values("datetime")
    dataframe.reset_index(drop=True, inplace=True)

    # ---- Feature engineering: selected indicators only ----
    dataframe = add_indicators(dataframe, indicators)
    indicator_columns = resolve_indicator_columns(indicators)

    # ---- Options features (expiry1_ / expiry2_ / expiry3_) ----
    options_columns = resolve_options_columns(dataframe)

    features = resolve_feature_columns(indicators) + options_columns

    if len(options_columns) != 102:
        raise RuntimeError(
            f"Expected 102 options features, found {len(options_columns)}"
        )

    if len(features) != 116:
        raise RuntimeError(
            f"Expected 116 total features, found {len(features)}"
        )

    # ---- Targets: next-bar return + SELL/HOLD/BUY direction ----
    dataframe = add_targets(dataframe)

    dataframe.dropna(inplace=True)
    dataframe.reset_index(drop=True, inplace=True)

    split_index = int(len(dataframe) * 0.80)

    train_dataframe = dataframe.iloc[:split_index].copy()
    test_dataframe = dataframe.iloc[split_index:].copy()

    # ---- Scale input features ----
    train_dataframe[features] = feature_scaler.fit_transform(
        train_dataframe[features]
    )

    test_dataframe[features] = feature_scaler.transform(
        test_dataframe[features]
    )

    # ---- Scale ONLY the regression target ----
    # target_direction stays as raw integer class labels (0/1/2).
    train_dataframe[["target_next_return"]] = target_scaler.fit_transform(
        train_dataframe[["target_next_return"]]
    )

    test_dataframe[["target_next_return"]] = target_scaler.transform(
        test_dataframe[["target_next_return"]]
    )

    X_train, y_train = create_sequences(train_dataframe, features)
    X_test, y_test = create_sequences(test_dataframe, features)

    if len(X_train) == 0:
        raise RuntimeError("Training dataset is empty")

    if len(X_test) == 0:
        raise RuntimeError("Validation dataset is empty")

    print(f"\nTrain Shape: {X_train.shape}")
    print(f"Test Shape : {X_test.shape}")

    train_dataset = MarketDataset(
        X_train,
        y_train
    )

    test_dataset = MarketDataset(
        X_test,
        y_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

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
# TCN MODEL
# The input layer size (n_features) is passed in dynamically from
# len(features). With the full DEFAULT_INDICATORS set plus the 102
# options features that's:
#   5 base OHLCV + 9 indicator columns + 102 options columns = 116.
#
# The output layer now has two heads:
#   - return_head:    1 value  -> predicted (scaled) target_next_return
#   - direction_head: 3 logits -> SELL / HOLD / BUY
# =========================================================

class TemporalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        kernel_size = 5
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(DROPOUT_RATE)

        self.downsample = (
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            )
            if in_channels != out_channels
            else None
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

        min_length = min(
            output.shape[2],
            residual.shape[2]
        )

        output = output[:, :, -min_length:]
        residual = residual[:, :, -min_length:]

        return output + residual


class TCNDualHead(nn.Module):

    def __init__(self, n_features):
        super().__init__()

        self.block1 = TemporalBlock(
            n_features,
            32,
            dilation=1
        )

        self.block2 = TemporalBlock(
            32,
            64,
            dilation=2
        )

        self.block3 = TemporalBlock(
            64,
            64,
            dilation=4
        )

        self.block4 = TemporalBlock(
            64,
            64,
            dilation=8
        )

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.recent_timesteps = 4

        self.fc1 = nn.Linear(
            64 * self.recent_timesteps,
            128
        )

        self.fc2 = nn.Linear(
            128,
            64
        )

        # Two heads sharing the same TCN representation.
        self.return_head = nn.Linear(64, 1)
        self.direction_head = nn.Linear(64, 3)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x[:, :, -self.recent_timesteps:]

        x = x.reshape(
            x.size(0),
            -1
        )

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout(x)

        return_prediction = self.return_head(x)
        direction_logits = self.direction_head(x)

        # No softmax here -- CrossEntropyLoss expects raw logits, and
        # trading confidence is derived from softmax(direction_logits)
        # outside the model (see calculate_trading_confidence()).
        return return_prediction, direction_logits


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
direction_precision_list = []
direction_recall_list = []
trading_confidence_list = []


def calculate_confidence_score(residual_std, target_mean):
    """
    Legacy regression/statistical diagnostic only. This is NOT used as
    trading confidence anymore -- see calculate_trading_confidence().
    """
    if target_mean == 0:
        return 0.0

    confidence = 100 - (
        residual_std / abs(target_mean)
    ) * 100

    confidence = max(
        0,
        min(100, confidence)
    )

    return float(confidence)


def calculate_trading_confidence(direction_logits):
    """
    Trading confidence comes from the direction head: softmax over
    SELL/HOLD/BUY logits, then the highest probability is the
    confidence for that predicted class.
    """
    probabilities = torch.softmax(
        direction_logits,
        dim=1
    )

    confidence = torch.max(
        probabilities,
        dim=1
    ).values

    return probabilities, confidence


def multi_step_validation(
    model,
    X_raw,
    y_raw,
    target_scaler,
    steps=MULTI_STEP_VALIDATION_STEPS
):
    """
    Rolls the model forward `steps` bars, feeding its own predicted
    return back in (translated into an updated open/close) and
    measures how far off the return predictions drift.

    NOTE: per the leakage rules, this path is only ever reached when
    there are NO options columns in play (see train_model()). With
    the 116-feature options pipeline this function is intentionally
    skipped, since faking future options data is not allowed.
    """
    model.eval()

    predictions = []
    targets = []

    sample_count = min(
        len(X_raw) - steps,
        300
    )

    if sample_count <= 0:
        return 0.0

    # target_next_return is the only column the target_scaler was
    # fit on, so its stats live at index 0.
    next_return_mean = target_scaler.mean_[0]
    next_return_scale = target_scaler.scale_[0]

    with torch.no_grad():
        for i in range(sample_count):
            current_sequence = X_raw[i].copy()

            step_predictions = []

            for step in range(steps):
                seq_tensor = (
                    torch.tensor(
                        current_sequence,
                        dtype=torch.float32
                    )
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                pred_return, _pred_direction_logits = model(seq_tensor)

                pred_vector = (
                    pred_return
                    .cpu()
                    .numpy()[0]
                )

                pred_next_return_scaled = pred_vector[0]

                step_predictions.append(pred_next_return_scaled)

                next_sequence = current_sequence.copy()
                next_sequence[:-1] = next_sequence[1:]

                # BASE_FEATURES = [high, low, close, volume, open], so
                # "close" is index 2 and "open" is index 4. BASE_FEATURES
                # always occupies the first 5 feature columns regardless
                # of which indicators are selected (see
                # resolve_feature_columns()). These MUST be updated
                # together with BASE_FEATURES if its order ever changes.
                next_sequence[-1] = next_sequence[-2]
                next_sequence[-1][4] = pred_next_return_scaled
                next_sequence[-1][2] = pred_next_return_scaled

                current_sequence = next_sequence

            predictions.extend(step_predictions)

            # y_raw column 0 is target_next_return (scaled)
            targets.extend(
                y_raw[i:i + steps, 0]
            )

    predictions = np.array(predictions)
    targets = np.array(targets)

    predictions_real = (predictions * next_return_scale) + next_return_mean
    targets_real = (targets * next_return_scale) + next_return_mean

    multi_mae = mean_absolute_error(
        targets_real,
        predictions_real
    )

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
    # Default to the full EMA9 + EMA20 + EMA50 + VWAP + RSI14 + ATR14
    # + MACD indicator set for this pipeline. Callers can still pass a
    # subset, but nothing outside INDICATOR_COLUMN_MAP is accepted --
    # see INDICATOR_REGISTRY / normalize_indicator_name().
    if indicators is None:
        indicators = list(DEFAULT_INDICATORS)

    # ---- Normalize into canonical names ----
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
    direction_precision_list.clear()
    direction_recall_list.clear()
    trading_confidence_list.clear()

    symbol = symbol.upper()

    MODEL_PATH = (
        Path(__file__).parent.parent
        / "models"
        / f"best_tcn_{symbol.lower()}.pth"
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        train_loader,
        test_loader,
        feature_scaler,
        target_scaler,
        features,
        indicator_columns,
        options_columns
    ) = prepare_training_data(symbol, indicators, data_source, data_path)

    # ---- Print the full pre-training configuration banner ----
    print_training_summary(symbol, indicators, indicator_columns, features, options_columns)

    huber_loss = nn.HuberLoss()
    direction_loss_fn = nn.CrossEntropyLoss()

    model = TCNDualHead(
        n_features=len(features)
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10
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

            optimizer.zero_grad()

            predicted_return, direction_logits = model(X_batch)

            y_return = y_batch[:, 0:1]
            y_direction = y_batch[:, 1].long()

            return_loss = huber_loss(
                predicted_return,
                y_return
            )

            classification_loss = direction_loss_fn(
                direction_logits,
                y_direction
            )

            loss = return_loss + classification_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

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
        all_trading_confidences = []
        X_validation_raw = []
        y_validation_raw = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                predicted_return, direction_logits = model(X_batch)

                y_return = y_batch[:, 0:1]
                y_direction = y_batch[:, 1].long()

                return_loss = huber_loss(
                    predicted_return,
                    y_return
                )

                classification_loss = direction_loss_fn(
                    direction_logits,
                    y_direction
                )

                loss = return_loss + classification_loss

                running_val_loss += loss.item()

                all_return_predictions.extend(
                    predicted_return.cpu().numpy()
                )

                all_return_targets.extend(
                    y_return.cpu().numpy()
                )

                all_direction_logits.extend(
                    direction_logits.cpu().numpy()
                )

                all_direction_targets.extend(
                    y_direction.cpu().numpy()
                )

                _probabilities, batch_confidence = calculate_trading_confidence(
                    direction_logits
                )

                all_trading_confidences.extend(
                    batch_confidence.cpu().numpy()
                )

                X_validation_raw.extend(
                    X_batch.cpu().numpy()
                )

                y_validation_raw.extend(
                    y_batch.cpu().numpy()
                )

        avg_val_loss = running_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        all_return_predictions = np.array(all_return_predictions)   # (N, 1) scaled
        all_return_targets = np.array(all_return_targets)           # (N, 1) scaled
        all_direction_logits = np.array(all_direction_logits)       # (N, 3)
        all_direction_targets = np.array(all_direction_targets)     # (N,)

        # ---- Unscale the regression target ----
        next_return_pred_real = target_scaler.inverse_transform(
            all_return_predictions
        )[:, 0]

        next_return_true_real = target_scaler.inverse_transform(
            all_return_targets
        )[:, 0]

        prediction_std = np.std(next_return_pred_real)
        target_std = np.std(next_return_true_real)

        mae = mean_absolute_error(
            next_return_true_real,
            next_return_pred_real
        )

        rmse = np.sqrt(
            mean_squared_error(
                next_return_true_real,
                next_return_pred_real
            )
        )

        r2 = r2_score(
            next_return_true_real,
            next_return_pred_real
        )

        corr = float(
            np.corrcoef(
                next_return_true_real,
                next_return_pred_real
            )[0, 1]
        )

        residual_std = np.std(
            next_return_true_real - next_return_pred_real
        )

        # Legacy statistical confidence -- diagnostic only, NOT used
        # as trading confidence anymore.
        confidence_score = calculate_confidence_score(
            residual_std,
            np.mean(next_return_true_real)
        )

        # ---- Direction (SELL/HOLD/BUY) metrics ----
        direction_predictions = np.argmax(
            all_direction_logits,
            axis=1
        )

        direction_accuracy = accuracy_score(
            all_direction_targets,
            direction_predictions
        )

        direction_precision = precision_score(
            all_direction_targets,
            direction_predictions,
            average="weighted",
            zero_division=0
        )

        direction_recall = recall_score(
            all_direction_targets,
            direction_predictions,
            average="weighted",
            zero_division=0
        )

        direction_confusion_matrix = confusion_matrix(
            all_direction_targets,
            direction_predictions,
            labels=[0, 1, 2]
        )

        mean_trading_confidence = float(
            np.mean(all_trading_confidences)
        )

        # ---- Multi-step validation: skipped when options features
        # exist, since we never fabricate future options data ----
        if options_columns:
            multi_step_mae = 0.0
            print(
                "Multi-step validation skipped: "
                "future options data is unavailable."
            )
        else:
            multi_step_mae = multi_step_validation(
                model,
                np.array(X_validation_raw),
                np.array(y_validation_raw),
                target_scaler,
                steps=MULTI_STEP_VALIDATION_STEPS
            )

        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        corr_list.append(corr)
        confidence_score_list.append(confidence_score)
        multi_step_mae_list.append(multi_step_mae)
        direction_accuracy_list.append(direction_accuracy)
        direction_precision_list.append(direction_precision)
        direction_recall_list.append(direction_recall)
        trading_confidence_list.append(mean_trading_confidence)

        print("\n========== RETURN DIAGNOSTICS (target_next_return) ==========")
        print("Prediction STD:", prediction_std)
        print("Target STD    :", target_std)
        print("Prediction Mean:", np.mean(next_return_pred_real))
        print("Target Mean    :", np.mean(next_return_true_real))
        print("Prediction Min :", np.min(next_return_pred_real))
        print("Prediction Max :", np.max(next_return_pred_real))
        print("Target Min     :", np.min(next_return_true_real))
        print("Target Max     :", np.max(next_return_true_real))
        print("======================================")

        print("\n========== DIRECTION DIAGNOSTICS (SELL=0 / HOLD=1 / BUY=2) ==========")
        print(f"Direction Accuracy  : {direction_accuracy:.4f}")
        print(f"Direction Precision : {direction_precision:.4f}")
        print(f"Direction Recall    : {direction_recall:.4f}")
        print("Confusion Matrix (rows=true, cols=pred, order=[SELL, HOLD, BUY]):")
        print(direction_confusion_matrix)
        print(f"Mean Trading Confidence: {mean_trading_confidence:.4f}")
        print("======================================")

        print(f"Epoch [{epoch + 1}] Completed")
        print(f"Train Loss           : {avg_train_loss:.6f}")
        print(f"Val Loss             : {avg_val_loss:.6f}")
        print(f"Return MAE           : {mae:.6f}")
        print(f"Return RMSE          : {rmse:.6f}")
        print(f"Return R\u00b2            : {r2:.6f}")
        print(f"Correlation          : {corr:.6f}")
        print(f"Confidence Score     : {confidence_score:.2f}%  (legacy diagnostic)")
        print(f"Direction Accuracy   : {direction_accuracy:.4f}")
        print(f"Direction Precision  : {direction_precision:.4f}")
        print(f"Direction Recall     : {direction_recall:.4f}")
        print(f"Multi-Step MAE       : {multi_step_mae:.6f}")

        if progress_callback:
            progress_callback(
                epoch + 1,
                EPOCHS,
                avg_train_loss,
                avg_val_loss
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "features": features,
                    "indicators": indicators,
                    "indicator_columns": indicator_columns,
                    "sequence_length": SEQUENCE_LENGTH,
                    "forecast_steps": FORECAST_STEPS,

                    "target_columns": TARGET_COLUMNS,
                    "direction_threshold": DIRECTION_THRESHOLD,
                    "direction_labels": DIRECTION_LABELS,
                    "min_signal_confidence": MIN_SIGNAL_CONFIDENCE,

                    "options_columns": options_columns,
                    "num_features": len(features),
                    "tcn_channels": [32, 64, 64, 64],
                    "kernel_size": 5,

                    "target_scaler_mean": target_scaler.mean_,
                    "target_scaler_scale": target_scaler.scale_,

                    "feature_scaler_mean": feature_scaler.mean_,
                    "feature_scaler_scale": feature_scaler.scale_,

                    "residual_std": residual_std,
                    "confidence_score": confidence_score,
                    "direction_accuracy": direction_accuracy,
                    "direction_precision": direction_precision,
                    "direction_recall": direction_recall,
                    "mean_trading_confidence": mean_trading_confidence,
                    "multi_step_validation_mae": multi_step_mae,

                    "dropout_rate": DROPOUT_RATE,
                    "multi_step_validation_steps": MULTI_STEP_VALIDATION_STEPS,
                },
                MODEL_PATH
            )

            print("\nBest TCN Model Saved")

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly Stopping Triggered")
            return True

    print("\nTCN Training Complete.")
    return True


if __name__ == "__main__":
    # Standalone run: trains on OHLCV + EMA9 + EMA20 + EMA50 + VWAP +
    # RSI14 + ATR14 + MACD + the 102 options features (the pipeline's
    # full 116-feature set).
    train_model(symbol=DEFAULT_SYMBOL, indicators=DEFAULT_INDICATORS)