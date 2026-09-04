import json
import datetime
import pandas as pd
from pprint import pprint

# Number of nearest expiries (by daysToExpiration) to extract features for.
NUM_EXPIRIES = 3

# Field names to try (in order) when pulling a value off an individual
# option record. Different data vendors use slightly different keys, so we
# fall back through a list of common aliases.
FIELD_ALIASES = {
    'open_interest': ['openInterest', 'open_interest', 'oi'],
    'volume': ['totalVolume', 'volume'],
    'last': ['last', 'lastPrice'],
    'mark': ['mark', 'markPrice'],
    'bid': ['bid', 'bidPrice'],
    'ask': ['ask', 'askPrice'],
    'delta': ['delta'],
    'gamma': ['gamma'],
    'theta': ['theta'],
    'vega': ['vega'],
    'rho': ['rho'],
    'iv': ['volatility', 'impliedVolatility', 'iv'],
    'put_call': ['putCall', 'optionType', 'type'],
    'dte': ['daysToExpiration', 'dte'],
}


def _safe_float(value, default=0.0):
    """Safely convert a value to float, handling None/invalid/missing values."""
    if value is None:
        return default
    try:
        f = float(value)
        if f != f:  # NaN check
            return default
        return f
    except (TypeError, ValueError):
        return default


def _get_field(option, field_key, default=None):
    """Pull a value off an option dict, trying each known alias for that field."""
    for alias in FIELD_ALIASES.get(field_key, []):
        if alias in option:
            return option[alias]
    return default


def _is_call(option):
    pc = _get_field(option, 'put_call', '')
    if pc is None:
        return False
    return str(pc).strip().upper().startswith('C')


def _is_put(option):
    pc = _get_field(option, 'put_call', '')
    if pc is None:
        return False
    return str(pc).strip().upper().startswith('P')


def _mean(values):
    """Safe mean that returns 0.0 for an empty list instead of crashing."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _aggregate_side(options):
    """
    Given a list of option records (already filtered to CALLs or PUTs),
    compute OI/Volume/GEX totals and averaged greeks/IV/pricing fields.
    Safely handles missing/invalid values.
    """
    oi_total = 0.0
    volume_total = 0.0
    gex_total = 0.0

    last_vals, mark_vals, bid_vals, ask_vals = [], [], [], []
    delta_vals, gamma_vals, theta_vals, vega_vals, rho_vals, iv_vals = [], [], [], [], [], []

    for opt in options:
        try:
            oi = _safe_float(_get_field(opt, 'open_interest'))
            vol = _safe_float(_get_field(opt, 'volume'))
            gamma = _safe_float(_get_field(opt, 'gamma'))

            oi_total += oi
            volume_total += vol
            # Simplified gamma exposure: gamma * open interest * 100 (per-contract multiplier)
            gex_total += gamma * oi * 100.0

            last_vals.append(_safe_float(_get_field(opt, 'last')))
            mark_vals.append(_safe_float(_get_field(opt, 'mark')))
            bid_vals.append(_safe_float(_get_field(opt, 'bid')))
            ask_vals.append(_safe_float(_get_field(opt, 'ask')))
            delta_vals.append(_safe_float(_get_field(opt, 'delta')))
            gamma_vals.append(gamma)
            theta_vals.append(_safe_float(_get_field(opt, 'theta')))
            vega_vals.append(_safe_float(_get_field(opt, 'vega')))
            rho_vals.append(_safe_float(_get_field(opt, 'rho')))
            iv_vals.append(_safe_float(_get_field(opt, 'iv')))
        except Exception:
            # Skip any malformed individual option record rather than crashing.
            continue

    return {
        'oi': oi_total,
        'volume': volume_total,
        'gex': gex_total,
        'avg_last': _mean(last_vals),
        'avg_mark': _mean(mark_vals),
        'avg_bid': _mean(bid_vals),
        'avg_ask': _mean(ask_vals),
        'avg_delta': _mean(delta_vals),
        'avg_gamma': _mean(gamma_vals),
        'avg_theta': _mean(theta_vals),
        'avg_vega': _mean(vega_vals),
        'avg_rho': _mean(rho_vals),
        'avg_iv': _mean(iv_vals),
    }


def _build_expiry_features(prefix, dte, call_agg, put_agg):
    """Build the flat feature dict for a single expiry (calls + puts + ratios)."""
    features = {f'{prefix}_dte': dte}

    for side_name, agg in (('call', call_agg), ('put', put_agg)):
        features[f'{prefix}_{side_name}_oi'] = agg['oi']
        features[f'{prefix}_{side_name}_volume'] = agg['volume']
        features[f'{prefix}_{side_name}_gex'] = agg['gex']
        features[f'{prefix}_{side_name}_avg_last'] = agg['avg_last']
        features[f'{prefix}_{side_name}_avg_mark'] = agg['avg_mark']
        features[f'{prefix}_{side_name}_avg_bid'] = agg['avg_bid']
        features[f'{prefix}_{side_name}_avg_ask'] = agg['avg_ask']
        features[f'{prefix}_{side_name}_avg_delta'] = agg['avg_delta']
        features[f'{prefix}_{side_name}_avg_gamma'] = agg['avg_gamma']
        features[f'{prefix}_{side_name}_avg_theta'] = agg['avg_theta']
        features[f'{prefix}_{side_name}_avg_vega'] = agg['avg_vega']
        features[f'{prefix}_{side_name}_avg_rho'] = agg['avg_rho']
        features[f'{prefix}_{side_name}_avg_iv'] = agg['avg_iv']

    call_oi, put_oi = call_agg['oi'], put_agg['oi']
    call_vol, put_vol = call_agg['volume'], put_agg['volume']
    call_gex, put_gex = call_agg['gex'], put_agg['gex']

    features[f'{prefix}_put_call_oi_ratio'] = (put_oi / call_oi) if call_oi else 0.0
    features[f'{prefix}_put_call_volume_ratio'] = (put_vol / call_vol) if call_vol else 0.0
    features[f'{prefix}_net_oi'] = call_oi - put_oi
    features[f'{prefix}_net_volume'] = call_vol - put_vol
    features[f'{prefix}_net_gex'] = call_gex - put_gex
    features[f'{prefix}_total_oi'] = call_oi + put_oi
    features[f'{prefix}_total_volume'] = call_vol + put_vol

    return features


def _empty_expiry_features(prefix):
    """Zeroed-out feature block for snapshots that don't have a given expiry slot."""
    empty_agg = {
        'oi': 0.0, 'volume': 0.0, 'gex': 0.0,
        'avg_last': 0.0, 'avg_mark': 0.0, 'avg_bid': 0.0, 'avg_ask': 0.0,
        'avg_delta': 0.0, 'avg_gamma': 0.0, 'avg_theta': 0.0,
        'avg_vega': 0.0, 'avg_rho': 0.0, 'avg_iv': 0.0,
    }
    return _build_expiry_features(prefix, None, empty_agg, empty_agg)


def _extract_snapshot_options_features(options_list):
    """
    Given the full optionsData list for one snapshot (all expiries, all strikes),
    select the nearest NUM_EXPIRIES expiries by daysToExpiration and build the
    flat feature dict (expiry1_*, expiry2_*, expiry3_*).
    """
    by_dte = {}
    for opt in options_list:
        try:
            dte = _get_field(opt, 'dte')
            if dte is None:
                continue
            dte = float(dte)
        except (TypeError, ValueError):
            continue
        by_dte.setdefault(dte, []).append(opt)

    nearest_dtes = sorted(by_dte.keys())[:NUM_EXPIRIES]

    row_features = {}
    for i in range(NUM_EXPIRIES):
        prefix = f'expiry{i + 1}'
        if i < len(nearest_dtes):
            dte = nearest_dtes[i]
            options_at_dte = by_dte[dte]
            calls = [o for o in options_at_dte if _is_call(o)]
            puts = [o for o in options_at_dte if _is_put(o)]
            call_agg = _aggregate_side(calls)
            put_agg = _aggregate_side(puts)
            row_features.update(_build_expiry_features(prefix, dte, call_agg, put_agg))
        else:
            # Fewer than NUM_EXPIRIES expiries available in this snapshot.
            row_features.update(_empty_expiry_features(prefix))

    return row_features


def load_local_options(data_path="data/OptionsDataAllDteSPY.jsonlines"):
    """
    Loads JSONLines options data from a local file path efficiently line-by-line
    (single pass), builds a combined OHLCV + nearest-3-expiry options feature
    DataFrame suitable for ML training, and prints a diagnostic summary.
    """
    print(f"OPTIONS LOADER CALLED: {data_path}")

    total_records = 0
    datetimes = []
    options_records_per_snapshot = []
    days_to_expiration = set()
    sample_record = None

    rows = []

    try:
        with open(data_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                parsed = json.loads(line)
                total_records += 1

                if sample_record is None:
                    sample_record = parsed

                # --- OHLCV ---
                row = {}
                if 'ohlcv' in parsed:
                    ohlcv_record = parsed['ohlcv']
                    dt = ohlcv_record.get('datetime')
                    if isinstance(dt, (int, float)):
                        dt = datetime.datetime.fromtimestamp(dt)
                    row['datetime'] = dt
                    row['open'] = _safe_float(ohlcv_record.get('open'), None)
                    row['high'] = _safe_float(ohlcv_record.get('high'), None)
                    row['low'] = _safe_float(ohlcv_record.get('low'), None)
                    row['close'] = _safe_float(ohlcv_record.get('close'), None)
                    row['volume'] = _safe_float(ohlcv_record.get('volume'), None)

                    if dt is not None:
                        datetimes.append(dt)
                else:
                    # No OHLCV on this line - skip building a row for it.
                    continue

                # --- Options ---
                options_list = parsed.get('optionsData', [])
                options_records_per_snapshot.append(len(options_list))
                for opt in options_list:
                    dte_val = _get_field(opt, 'dte')
                    if dte_val is not None:
                        try:
                            days_to_expiration.add(float(dte_val))
                        except (TypeError, ValueError):
                            pass

                try:
                    options_features = _extract_snapshot_options_features(options_list)
                except Exception:
                    # If options feature extraction fails for this snapshot,
                    # fall back to empty (zeroed) options features rather than
                    # dropping the OHLCV row or crashing the whole load.
                    options_features = {}
                    for i in range(NUM_EXPIRIES):
                        options_features.update(_empty_expiry_features(f'expiry{i + 1}'))

                row.update(options_features)
                rows.append(row)

        # Print Diagnostics
        print("\n=== DIAGNOSTIC SUMMARY ===")
        print(f"Total records (lines): {total_records}")

        if datetimes:
            sorted_dts = sorted(datetimes)
            print(f"First datetime: {sorted_dts[0]}")
            print(f"Last datetime: {sorted_dts[-1]}")
        else:
            print("No valid datetimes found.")

        if options_records_per_snapshot:
            avg_opts = sum(options_records_per_snapshot) / len(options_records_per_snapshot)
            min_opts = min(options_records_per_snapshot)
            max_opts = max(options_records_per_snapshot)
            print(f"Options records per snapshot: Avg: {avg_opts:.2f}, Min: {min_opts}, Max: {max_opts}")
        else:
            print("No options records found in snapshots.")

        print(f"Days to expiration found: {sorted(list(days_to_expiration))}")

        print("\n=== SAMPLE RECORD ===")
        pprint(sample_record)
        print("==========================\n")

        # --- Build final combined DataFrame ---
        df = pd.DataFrame(rows)
        if 'datetime' in df.columns:
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)

        print("=== FINAL DATASET ===")
        print(f"Shape: {df.shape}")
        print(f"Columns ({len(df.columns)}):")
        for col in df.columns:
            print(f"  - {col}")
        print("======================\n")

        return df

    except Exception as e:
        print(f"Error loading options data: {e}")
        raise


if __name__ == "__main__":
    load_local_options()