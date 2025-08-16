# fetch_dse_screener.py

from __future__ import annotations


from utils.requests.requests_wrapper import create_session, SafeSession

# tv_bangladesh_extended_hours.py
# Requirements: pandas, requests, tenacity, and your safe_requests.py in the same directory.

import uuid
import logging
from typing import List, Dict, Optional

import pandas as pd
# from safe_requests import create_session  # your rate-limited, retrying wrapper around requests

TV_URL = "https://scanner.tradingview.com/bangladesh/scan?label-product=screener-stock"

# EXACT column keys from your working browser payload (order matters)
TV_COLUMNS = [
    "name",               # 0 -> short ticker (e.g., "GP")
    "description",        # 1 -> full name (e.g., "GRAMEENPHONE LTD")
    "logoid",             # 2
    "update_mode",        # 3
    "type",               # 4
    "typespecs",          # 5
    "premarket_close",    # 6
    "pricescale",         # 7
    "minmov",             # 8
    "fractional",         # 9
    "minmove2",           # 10
    "currency",           # 11
    "premarket_change",   # 12
    "premarket_gap",      # 13
    "premarket_volume",   # 14
    "close",              # 15 -> Price
    "change",             # 16 -> Change %
    "gap",                # 17 -> Gap %
    "volume",             # 18 -> Volume
    "volume_change",      # 19 -> Volume Change %
    "postmarket_close",   # 20
    "postmarket_change",  # 21
    "postmarket_volume",  # 22
    "exchange",           # 23 -> e.g., "DSEBD"
]

# Map indices in the "d" array to the final DataFrame column names you want to see
IDX_TO_FINAL = {
    0:  "Symbol",
    1:  "Name",
    15: "Price",
    16: "Change %",
    17: "Gap %",
    18: "Volume",
    19: "Volume Change %",
    # 23 ("exchange") will be used as a guard, not included in output
}

FINAL_COLUMNS = [
    "Symbol", "Name", "Price", "Change %",
    "Gap %", "Volume", "Volume Change %",
]

def _build_payload(offset: int, page_size: int) -> Dict:
    """
    Build the exact payload you pasted, with pagination via a half-open range:
    [offset, offset + page_size)
    """
    return {
        "columns": TV_COLUMNS,
        "filter": [
            {"left": "is_primary", "operation": "equal", "right": True}
        ],
        "ignore_unknown_fields": False,
        "options": {"lang": "en"},
        "range": [offset, offset + page_size],
        "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
        "symbols": {},
        "markets": ["bangladesh"],
        "filter2": {
            "operator": "and",
            "operands": [
                {
                    "operation": {
                        "operator": "or",
                        "operands": [
                            {
                                "operation": {
                                    "operator": "and",
                                    "operands": [
                                        {"expression": {"left": "type", "operation": "equal", "right": "stock"}},
                                        {"expression": {"left": "typespecs", "operation": "has", "right": ["common"]}}
                                    ]
                                }
                            },
                            {
                                "operation": {
                                    "operator": "and",
                                    "operands": [
                                        {"expression": {"left": "type", "operation": "equal", "right": "stock"}},
                                        {"expression": {"left": "typespecs", "operation": "has", "right": ["preferred"]}}
                                    ]
                                }
                            },
                            {
                                "operation": {
                                    "operator": "and",
                                    "operands": [
                                        {"expression": {"left": "type", "operation": "equal", "right": "dr"}}
                                    ]
                                }
                            },
                            {
                                "operation": {
                                    "operator": "and",
                                    "operands": [
                                        {"expression": {"left": "type", "operation": "equal", "right": "fund"}},
                                        {"expression": {"left": "typespecs", "operation": "has_none_of", "right": ["etf"]}}
                                    ]
                                }
                            }
                        ]
                    }
                }
            ]
        }
    }

def _fetch_page(session, offset: int, page_size: int, debug: bool = False) -> list:
    """
    POST one page to the TradingView scanner and return the raw 'data' list.
    Uses SafeSession => shared rate limit + retries (incl. Retry-After handling).
    """
    headers = {
        "Content-Type": "application/json",
        "Origin": "https://www.tradingview.com",
        "Referer": "https://www.tradingview.com/markets/stocks-bangladesh/market-movers-all-stocks/",
        "Idempotency-Key": str(uuid.uuid4()),  # safe POST retries
    }
    payload = _build_payload(offset, page_size)

    resp = session.post(
        TV_URL,
        json=payload,
        headers=headers,
        allow_unsafe_method_retry=True,  # scan is read-like; safe to retry
        raise_on_status=False,           # don’t raise so we can debug on 4xx
        timeout=25,
    )

    if resp.status_code >= 400:
        if debug:
            print(f"[DEBUG] HTTP {resp.status_code} at offset={offset}, size={page_size}")
            try:
                print("[DEBUG] Body snippet:", resp.text[:500])
            except Exception:
                pass
        return []

    try:
        j = resp.json()
    except Exception:
        if debug:
            print("[DEBUG] Non-JSON response at offset", offset)
        return []

    return j.get("data") or []

def _rows_to_df_records(raw_items: list) -> List[dict]:
    """
    Convert each raw item ({'s': 'DSEBD:GP', 'd': [...]}) into a dict for DataFrame.
    We also keep only 'DSEBD' rows as a sanity check.
    """
    out: List[dict] = []
    for item in raw_items:
        d = item.get("d") or []
        if len(d) < len(TV_COLUMNS):
            continue

        # Ensure we're on DSEBD (position 23 in TV_COLUMNS)
        exchange = d[23] if len(d) > 23 else None
        if exchange and exchange != "DSEBD":
            continue

        row = {}
        for idx, col_name in IDX_TO_FINAL.items():
            row[col_name] = d[idx]
        out.append(row)
    return out

def fetch_all_bangladesh_stocks_extended_hours(page_size: int = 300,
                                               max_total: Optional[int] = None,
                                               csv_path: str = "bangladesh_extended_hours.csv",
                                               debug: bool = False) -> pd.DataFrame:
    """
    Fetch all Bangladesh stocks (extended-hours payload), paginate through every page,
    and save the resulting DataFrame to CSV.

    Returns the pandas.DataFrame.
    """
    # Shared session => shared token bucket & retries across pages
    s = create_session(
        max_calls=4, per_seconds=1.0, burst=6,      # polite shared rate limit
        max_attempts=6, backoff_min=0.5, backoff_max=10.0,
        user_agent="bd-screener-extended/1.0 (+https://example.com)",
    )

    # Clamp page size (TradingView typically supports 150–300 here)
    page_size = max(1, min(int(page_size), 300))

    all_records: List[dict] = []
    offset = 0

    while True:
        if max_total is not None and offset >= max_total:
            break

        curr_size = page_size if max_total is None else min(page_size, max_total - offset)
        raw = _fetch_page(s, offset=offset, page_size=curr_size, debug=debug)
        if not raw:
            break  # either no more rows or request was rejected

        records = _rows_to_df_records(raw)
        if not records:
            # Received items but none mapped (or exchange mismatch) -> finish
            break

        all_records.extend(records)

        # If we received fewer than requested, we've reached the end
        if len(raw) < curr_size:
            break

        offset += curr_size

    # Build the DataFrame
    if not all_records:
        df = pd.DataFrame(columns=FINAL_COLUMNS)
    else:
        df = pd.DataFrame(all_records, columns=FINAL_COLUMNS)

        # Coerce numeric columns
        for col in ["Price", "Change %", "Gap %", "Volume", "Volume Change %"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort for determinism
        df = df.sort_values(["Symbol"]).reset_index(drop=True)

    # Save to CSV (no index column)
    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV -> {csv_path}  (rows: {len(df)})")

    return df

# ------------- Example run -------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    df = fetch_all_bangladesh_stocks_extended_hours(page_size=300,
                                                    csv_path="bangladesh_extended_hours.csv",
                                                    debug=True)
    print(df.head(10).to_string(index=False))
    print("\nRows fetched:", len(df))