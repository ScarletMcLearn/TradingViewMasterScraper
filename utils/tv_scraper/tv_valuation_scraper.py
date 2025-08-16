# fetch_dse_screener.py

from __future__ import annotations


from utils.requests.requests_wrapper import create_session, SafeSession

# tv_bangladesh_valuation.py
# Requirements: pandas, requests, tenacity, and your safe_requests.py in the same directory.

import uuid
import logging
from typing import List, Dict, Optional

import pandas as pd
# from safe_requests import create_session  # your rate-limited, retrying wrapper around requests

TV_URL = "https://scanner.tradingview.com/bangladesh/scan?label-product=screener-stock"

# --- EXACT column keys from your browser payload (ORDER MATTERS) ---
TV_COLUMNS = [
    "name",                              # 0 -> short symbol (e.g., "GP")
    "description",                       # 1 -> full name (e.g., "GRAMEENPHONE LTD")
    "logoid",                            # 2
    "update_mode",                       # 3
    "type",                              # 4
    "typespecs",                         # 5
    "market_cap_basic",                  # 6 -> Market Cap
    "fundamental_currency_code",         # 7
    "Perf.1Y.MarketCap",                 # 8 -> Market Cap Performance % 1Y
    "price_earnings_ttm",                # 9 -> P/E
    "price_earnings_growth_ttm",         # 10 -> PEG (P/E/Growth, Trading 12M)
    "price_sales_current",               # 11 -> P/S
    "price_book_fq",                     # 12 -> P/B
    "price_to_cash_f_operating_activities_ttm",  # 13 -> P/CF (operating cash flow)
    "price_free_cash_flow_ttm",          # 14 -> P/FCF
    "price_to_cash_ratio",               # 15 -> P/Cash
    "enterprise_value_current",          # 16 -> EV
    "enterprise_value_to_revenue_ttm",   # 17 -> EV/Revenue TTM
    "enterprise_value_to_ebit_ttm",      # 18 -> EV/EBIT TTM
    "enterprise_value_ebitda_ttm",       # 19 -> EV/EBITDA TTM
    "exchange",                          # 20 -> exchange (e.g., "DSEBD")
]

# Map indices in the returned "d" array to your final DataFrame column names
IDX_TO_FINAL = {
    0:  "Symbol",
    1:  "Name",
    6:  "Market Capitalization Performance",
    8:  "Market Capitalization Performance % 1Y",
    9:  "P/E",
    10: "P/E/Growth, Trading 12M",
    11: "P/S",
    12: "P/B",
    13: "P/CF",
    14: "P/FCF",
    15: "P/Cash",
    16: "EV",
    17: "EV/Revenue TTM",
    18: "EV/EBIT TTM",
    19: "EV/EBITDA TTM",
    # 20 is exchange → used as a guard, not included in output
}

FINAL_COLUMNS = [
    "Symbol", "Name",
    "Market Capitalization Performance",
    "Market Capitalization Performance % 1Y",
    "P/E", "P/E/Growth, Trading 12M", "P/S", "P/B", "P/CF", "P/FCF", "P/Cash",
    "EV", "EV/Revenue TTM", "EV/EBIT TTM", "EV/EBITDA TTM",
]

# Common headers that the scanner expects
COMMON_HEADERS = {
    "Content-Type": "application/json",
    "Origin": "https://www.tradingview.com",
    "Referer": "https://www.tradingview.com/markets/stocks-bangladesh/market-movers-all-stocks/",
}

def _build_payload(offset: int, page_size: int) -> Dict:
    """
    Build the exact payload you shared, with pagination via a half-open range:
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
    Uses your SafeSession => shared rate limit + retries (incl. Retry-After).
    """
    headers = {**COMMON_HEADERS, "Idempotency-Key": str(uuid.uuid4())}

    resp = session.post(
        TV_URL,
        json=_build_payload(offset, page_size),
        headers=headers,
        allow_unsafe_method_retry=True,  # scan is read-like; safe to retry
        raise_on_status=False,           # don't raise so we can keep paginating even if a page 4xx's
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

def _rows_to_records(raw_items: list) -> List[dict]:
    """
    Convert each raw item ({'s': 'DSEBD:GP', 'd': [...]}) into a dict for DataFrame.
    Keep only DSEBD (index 20) as a sanity check.
    """
    out: List[dict] = []
    for item in raw_items:
        d = item.get("d") or []
        if len(d) < len(TV_COLUMNS):
            continue

        # Ensure we're on DSEBD
        exchange = d[20] if len(d) > 20 else None
        if exchange and exchange != "DSEBD":
            continue

        row = {}
        for idx, col_name in IDX_TO_FINAL.items():
            row[col_name] = d[idx]
        out.append(row)
    return out

def fetch_all_bangladesh_stocks_valuation(page_size: int = 400,
                                          max_total: Optional[int] = None,
                                          csv_path: Optional[str] = None,
                                          debug: bool = False) -> pd.DataFrame:
    """
    Fetch all Bangladesh stocks (valuation view), paginate across all pages,
    and return a DataFrame with:
      Symbol, Name, Market Cap, Market Cap %1Y, P/E, PEG (TTM), P/S, P/B, P/CF, P/FCF, P/Cash,
      EV, EV/Revenue TTM, EV/EBIT TTM, EV/EBITDA TTM.

    If `csv_path` is provided, the DataFrame is saved to that CSV (no index).
    """
    # Shared session => shared token bucket & retries across pages
    s = create_session(
        max_calls=4, per_seconds=1.0, burst=6,         # polite rate limit
        max_attempts=6, backoff_min=0.5, backoff_max=10.0,
        user_agent="bd-screener-valuation/1.0 (+https://example.com)",
    )

    # Clamp page size (TV typically supports up to ~400 here)
    page_size = max(1, min(int(page_size), 400))

    all_records: List[dict] = []
    offset = 0

    while True:
        if max_total is not None and offset >= max_total:
            break

        curr_size = page_size if max_total is None else min(page_size, max_total - offset)
        raw = _fetch_page(s, offset=offset, page_size=curr_size, debug=debug)
        if not raw:
            break  # either end or request issue

        records = _rows_to_records(raw)
        if not records:
            # Received items but none mapped (or exchange mismatch) -> finish
            break

        all_records.extend(records)

        # If we received fewer than requested, we've reached the end
        if len(raw) < curr_size:
            break

        offset += curr_size

    # Build DataFrame
    if not all_records:
        df = pd.DataFrame(columns=FINAL_COLUMNS)
    else:
        df = pd.DataFrame(all_records, columns=FINAL_COLUMNS)

        # Coerce numeric columns
        for col in [
            "Market Capitalization Performance",
            "Market Capitalization Performance % 1Y",
            "P/E", "P/E/Growth, Trading 12M", "P/S", "P/B", "P/CF", "P/FCF", "P/Cash",
            "EV", "EV/Revenue TTM", "EV/EBIT TTM", "EV/EBITDA TTM",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort for determinism
        df = df.sort_values(["Symbol"]).reset_index(drop=True)

    # Save only if a filename/path was provided
    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV -> {csv_path}  (rows: {len(df)})")

    return df

# ------------- Example run -------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    df = fetch_all_bangladesh_stocks_valuation(
        page_size=400,
        # csv_path="bangladesh_valuation.csv",  # pass a filename to save; or omit to skip saving
        debug=True
    )
    print(df.head(10).to_string(index=False))
    print("\nRows fetched:", len(df))