# engine/inference_engine.py

from data.ftp_loader import load_ohlcv
from data.options_loader import load_local_options


class InferenceEngine:

    def __init__(self):
        pass

    def load_data(self, ticker, data_source="ftp", data_path=""):

        print(f"ENGINE LOAD_DATA CALLED | Source: {data_source} | Path: {data_path}")

        if data_source == "local":
            df = load_local_options(data_path)
            # Options data column normalization
            df.columns = [c.lower() for c in df.columns]
            if 'datetime' in df.columns:
                df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        else:
            df = load_ohlcv(ticker)

            # Must match GRU_Training.py / TCN_Training.py's load_data()
            # normalization exactly
            df.columns = [c.lower() for c in df.columns]

            if 'datetime' in df.columns:
                df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)

            return df