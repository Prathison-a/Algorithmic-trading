from ftplib import FTP
from io import BytesIO
import pandas as pd


FTP_HOST = "ftp.cavedspace.com"
FTP_USER = "ftpuser"
FTP_PASSWORD = "TestUser01!!"
FTP_ROOT = "/Backup"


def _connect():
    ftp = FTP(FTP_HOST, timeout=60)
    ftp.login(user=FTP_USER, passwd=FTP_PASSWORD)
    ftp.set_pasv(True)
    return ftp


def _latest_date_folder(ftp):
    """
    Lists date folders under FTP_ROOT (e.g. "20260809", "20260810", ...)
    and returns the most recent COMPLETE one -- the same [:-1] exclusion
    TCN_Training.py's load_symbol_data() uses, since the very last folder
    may still be mid-upload for the current day.
    """
    ftp.cwd(FTP_ROOT)
    folders = ftp.nlst()

    date_folders = sorted([f for f in folders if f.startswith("202")])

    print(f"All date folders found ({len(date_folders)}): {date_folders}")

    if not date_folders:
        raise RuntimeError(f"No date folders found under {FTP_ROOT}")

    usable_folders = date_folders[:-1] if len(date_folders) > 1 else date_folders

    chosen = usable_folders[-1]
    print(f"Chosen folder: {chosen}")

    return chosen


def load_ohlcv(ticker="SPY"):
    """
    Fetches the most recent day's OHLCV file for `ticker`, from:
        {FTP_ROOT}/{latest_date_folder}/{ticker}/OHLCV{ticker}.txt

    This mirrors TCN_Training.py's load_symbol_data() path structure --
    the old flat "RETR OHLCV{ticker}.txt" (no folder navigation) doesn't
    match where the files actually live on the server, which is what
    was causing the 550 error.
    """
    print("FTP LOADER CALLED")

    filename = f"OHLCV{ticker}.txt"

    ftp = _connect()

    try:
        latest_folder = _latest_date_folder(ftp)

        file_path = f"{FTP_ROOT}/{latest_folder}/{ticker}/{filename}"

        print(f"Loading: {file_path}")

        buffer = BytesIO()

        ftp.retrbinary(
            f"RETR {file_path}",
            buffer.write
        )

    finally:
        try:
            ftp.quit()
        except:
            ftp.close()

    buffer.seek(0)

    df = pd.read_csv(buffer)

    df["datetime"] = pd.to_datetime(
        df["datetime"], unit='ms'
    )

    return df