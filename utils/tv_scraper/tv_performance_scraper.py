# fetch_dse_screener.py

from __future__ import annotations

import math
from typing import List, Optional

import pandas as pd
from utils.requests.requests_wrapper import create_session, SafeSession

# tv_bangladesh_screener_exact.py
import uuid
import logging
from typing import List, Dict, Optional

# from safe_requests import create_session  # your rate-limited, retrying session

TV_URL = "https://scanner.tradingview.com/bangladesh/scan?label-product=screener-stock"

# EXACT column keys from your working browser payload (order matters)
TV_COLUMNS = [
    "name",           # 0 -> short ticker (e.g., "GP")
    "description",    # 1 -> full name (e.g., "GRAMEENPHONE LTD")
    "logoid",         # 2
    "update_mode",    # 3
    "type",           # 4
    "typespecs",      # 5
    "close",          # 6 -> last price
    "pricescale",     # 7
    "minmov",         # 8
    "fractional",     # 9
    "minmove2",       #10
    "currency",       #11
    "change",         #12 -> daily change %
    "Perf.W",         #13 -> 1W %
    "Perf.1M",        #14 -> 1M %
    "Perf.3M",        #15 -> 3M %
    "Perf.6M",        #16 -> 6M %
    "Perf.YTD",       #17 -> YTD %
    "Perf.Y",         #18 -> 1Y %
    "Perf.5Y",        #19 -> 5Y %
    "Perf.10Y",       #20 -> 10Y %
    "Perf.All",       #21 -> All time %
    "Volatility.W",   #22 -> 1W volatility
    "Volatility.M",   #23 -> 1M volatility
    "exchange",       #24 -> exchange code (e.g., "DSEBD")
]

# Map **indices** in the "d" array to your final DataFrame columns.
IDX_TO_FINAL = {
    0:  "Symbol",
    1:  "Name",
    6:  "Price",
    12: "Change %",
    13: "Perf % 1W",
    14: "Perf % 1M",
    15: "Perf % 3M",
    16: "Perf % 6M",
    17: "Perf % YTD",
    18: "Perf % 1Y",
    19: "Perf % 5Y",
    20: "Perf % 10Y",
    21: "Perf % All Time",
    22: "Volatility 1W",
    23: "Volatility 1M",
    # 24 is "exchange" (we'll use it to filter to DSEBD but not keep it in the final DF)
}

FINAL_COLUMNS = [
    "Symbol", "Name", "Price", "Change %",
    "Perf % 1W", "Perf % 1M", "Perf % 3M", "Perf % 6M",
    "Perf % YTD", "Perf % 1Y", "Perf % 5Y", "Perf % 10Y",
    "Perf % All Time", "Volatility 1W", "Volatility 1M",
]

def _build_payload(offset: int, page_size: int) -> Dict:
    """
    Build the exact payload you posted, with pagination:
    range = [offset, offset + page_size] (half-open interval).
    """
    return {
        "columns": TV_COLUMNS,
        # From your payload: keep primary listings only
        "filter": [
            {"left": "is_primary", "operation": "equal", "right": True}
        ],
        "ignore_unknown_fields": False,
        "options": {"lang": "en"},
        "range": [offset, offset + page_size],
        # Sort by market cap desc (as in your payload)
        "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
        # Empty object in your payload
        "symbols": {},
        # Explicit market selection (you had ["bangladesh"])
        "markets": ["bangladesh"],
        # The nested filter2 tree exactly as you pasted
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
    POST one page to the TradingView scanner.
    Uses SafeSession so we have shared rate limit, retries, and Retry-After handling.
    """
    headers = {
        "Content-Type": "application/json",
        "Origin": "https://www.tradingview.com",
        "Referer": "https://www.tradingview.com/markets/stocks-bangladesh/market-movers-all-stocks/",
        "Idempotency-Key": str(uuid.uuid4()),  # safe POST retries
    }

    payload = _build_payload(offset, page_size)

    # Important: don't raise immediately; let us inspect payloads if a 400 slips through.
    resp = session.post(
        TV_URL,
        json=payload,
        headers=headers,
        allow_unsafe_method_retry=True,
        raise_on_status=False,
        timeout=25,
    )

    if resp.status_code >= 400:
        if debug:
            print(f"[DEBUG] HTTP {resp.status_code} for offset={offset}, size={page_size}")
            txt = ""
            try:
                txt = resp.text[:500]
            except Exception:
                pass
            print("[DEBUG] Body snippet:", txt)
        return []

    try:
        j = resp.json()
    except Exception:
        if debug:
            print("[DEBUG] Non-JSON reply")
        return []

    return j.get("data") or []

def _rows_to_df_rows(raw_items: list) -> list[dict]:
    """
    Convert each item from the API ({'s': 'DSEBD:GP', 'd': [...]})
    to a dict keyed by FINAL_COLUMNS.
    """
    out = []
    for item in raw_items:
        d = item.get("d") or []
        if len(d) < len(TV_COLUMNS):
            continue  # malformed
        # Optionally ensure Bangladesh exchange
        exchange = d[24] if len(d) > 24 else None
        if exchange and exchange != "DSEBD":
            continue

        row = {}
        for idx, name in IDX_TO_FINAL.items():
            row[name] = d[idx]
        out.append(row)
    return out

def fetch_all_bangladesh_stocks_performance(page_size: int = 200,
                                            max_total: Optional[int] = None,
                                            debug: bool = False,
                                            session: Optional[SafeSession] = None) -> pd.DataFrame:
    """
    Fetch *all* pages from TradingView Bangladesh screener using your exact payload,
    and return a DataFrame with the columns you listed.

    Args:
      page_size: how many rows per page (the payload you pasted used 200).
      max_total: optional cap for testing.
      debug:     print short diagnostics on 4xx/parse errors.

    Returns:
      pandas.DataFrame with FINAL_COLUMNS order.
    """
    # Shared session = shared rate limit + retries for all pages
    s = session or create_session(
        max_calls=4, per_seconds=1.0, burst=6,      # polite shared rate limit
        max_attempts=6, backoff_min=0.5, backoff_max=10.0,
        user_agent="bd-screener/1.0 (+https://example.com)",
    )

    # TradingView typically supports 150–200 per page; clamp sanely
    page_size = max(1, min(int(page_size), 200))

    all_records: List[dict] = []
    offset = 0

    while True:
        if max_total is not None and offset >= max_total:
            break

        curr_size = page_size if max_total is None else min(page_size, max_total - offset)
        raw = _fetch_page(s, offset=offset, page_size=curr_size, debug=debug)
        if not raw:
            break  # no more rows / or error

        df_rows = _rows_to_df_rows(raw)
        if not df_rows:
            # If we got items but none matched mapping/exchange, treat as done.
            break

        all_records.extend(df_rows)

        # Stop if short page
        if len(raw) < curr_size:
            break

        offset += curr_size

    if not all_records:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    df = pd.DataFrame(all_records, columns=FINAL_COLUMNS)

    # Coerce numeric fields
    for col in [
        "Price", "Change %",
        "Perf % 1W", "Perf % 1M", "Perf % 3M", "Perf % 6M",
        "Perf % YTD", "Perf % 1Y", "Perf % 5Y", "Perf % 10Y",
        "Perf % All Time", "Volatility 1W", "Volatility 1M",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort for determinism
    df = df.sort_values("Symbol").reset_index(drop=True)
    return df

# ---------- Example run ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    session = create_session(
        max_calls=4, per_seconds=1.0, burst=6,
        max_attempts=6, backoff_min=0.5, backoff_max=10.0,
        user_agent="bd-screener/1.0 (+https://example.com)",
    )
    df = fetch_all_bangladesh_stocks_performance(page_size=200, debug=True, session=session)
    print(df.head(10).to_string(index=False))
    print("\nRows fetched:", len(df))
    df.to_csv("dse_screener_performance_all.csv", index=False)