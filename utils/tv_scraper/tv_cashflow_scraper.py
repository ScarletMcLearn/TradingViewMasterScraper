# fetch_dse_screener.py

from __future__ import annotations


from utils.requests.requests_wrapper import create_session, SafeSession

# fetch_dse_cashflow.py

import math
from typing import List, Optional

import pandas as pd
# from safe_requests import create_session, SafeSession

TV_URL = "https://scanner.tradingview.com/bangladesh/scan?label-product=screener-stock"

# Request exactly these columns (indexes matter)
TV_COLUMNS = [
    "name",                              # 0
    "description",                       # 1
    "logoid",                            # 2
    "update_mode",                       # 3
    "type",                              # 4
    "typespecs",                         # 5
    "cash_f_operating_activities_ttm",   # 6  -> Operating CF TTM
    "fundamental_currency_code",         # 7
    "cash_f_investing_activities_ttm",   # 8  -> Investing CF TTM
    "cash_f_financing_activities_ttm",   # 9  -> Financing CF TTM
    "free_cash_flow_ttm",                # 10 -> FCF TTM
    "neg_capital_expenditures_ttm",      # 11 -> CAPEX TTM (positive number = magnitude of capex)
    "exchange",                          # 12
]

# Map TradingView array indexes -> desired DataFrame columns
IDX_TO_DF = {
    0:  "Symbol",
    1:  "Name",
    6:  "Operating CF TTM",
    8:  "Investing CF TTM",
    9:  "Financing CF TTM",
    10: "FCF TTM",
    11: "CAPEX TTM",
}

BASE_PAYLOAD = {
    "columns": TV_COLUMNS,
    "filter": [{"left": "is_primary", "operation": "equal", "right": True}],
    "ignore_unknown_fields": False,
    "options": {"lang": "en"},
    "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
    "symbols": {},
    "markets": ["bangladesh"],
    "filter2": {
        "operator": "and",
        "operands": [{
            "operation": {
                "operator": "or",
                "operands": [
                    {"operation": {"operator": "and", "operands": [
                        {"expression": {"left": "type", "operation": "equal", "right": "stock"}},
                        {"expression": {"left": "typespecs", "operation": "has", "right": ["common"]}},
                    ]}},
                    {"operation": {"operator": "and", "operands": [
                        {"expression": {"left": "type", "operation": "equal", "right": "stock"}},
                        {"expression": {"left": "typespecs", "operation": "has", "right": ["preferred"]}},
                    ]}},
                    {"operation": {"operator": "and", "operands": [
                        {"expression": {"left": "type", "operation": "equal", "right": "dr"}},
                    ]}},
                    {"operation": {"operator": "and", "operands": [
                        {"expression": {"left": "type", "operation": "equal", "right": "fund"}},
                        {"expression": {"left": "typespecs", "operation": "has_none_of", "right": ["etf"]}},
                    ]}},
                ],
            }
        }],
    },
}

def _payload_with_range(start: int, end: int) -> dict:
    # TradingView expects [from, to) — end-exclusive
    p = dict(BASE_PAYLOAD)
    p["range"] = [start, end]
    return p

def _row_to_record(item: dict) -> dict:
    arr = item.get("d", []) or []
    rec = {}
    for idx, out_name in IDX_TO_DF.items():
        rec[out_name] = arr[idx] if idx < len(arr) else None
    # If you want to keep extras, uncomment:
    # rec["Ticker"] = item.get("s")
    # rec["Currency"] = arr[7] if len(arr) > 7 else None
    return rec

def fetch_dse_cashflow_dataframe(
    page_size: int = 100,
    session: Optional[SafeSession] = None,
    max_pages: Optional[int] = None,
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch ALL Bangladesh stocks cash-flow fields with correct pagination.

    Args:
        page_size: rows per page (100 is a good default).
        session: pass a shared SafeSession to apply global rate limits; if None, one is created.
        max_pages: optional cap; None fetches all pages based on totalCount.
        csv_path: if provided, saves the final DataFrame to this CSV path.

    Returns:
        pandas.DataFrame with columns:
        ["Symbol","Name","Operating CF TTM","Investing CF TTM","Financing CF TTM","FCF TTM","CAPEX TTM"]
    """
    s = session or create_session(
        max_calls=2, per_seconds=1.0, burst=2,            # global token-bucket
        max_attempts=6, backoff_min=0.5, backoff_max=30,  # retries with backoff
        user_agent="dse-cashflow-fetcher/1.0 (+https://example.org)",
    )

    # First page to learn totalCount
    start, end = 0, page_size
    r = s.post(TV_URL, json=_payload_with_range(start, end), timeout=15)
    js = r.json()
    total = int(js.get("totalCount", 0))
    data = js.get("data", []) or []

    if total <= 0 or not data:
        return pd.DataFrame(columns=list(IDX_TO_DF.values()))

    chunks: List[pd.DataFrame] = []
    chunks.append(pd.DataFrame.from_records([_row_to_record(it) for it in data]))

    # Remaining pages
    pages_total = math.ceil(total / page_size)
    if max_pages is not None:
        pages_total = min(pages_total, max_pages)

    for page_idx in range(1, pages_total):
        start = page_idx * page_size
        end = min(start + page_size, total)
        r = s.post(TV_URL, json=_payload_with_range(start, end), timeout=15)
        js = r.json()
        page_data = js.get("data", []) or []
        if not page_data:
            break
        chunks.append(pd.DataFrame.from_records([_row_to_record(it) for it in page_data]))

    df = pd.concat(chunks, ignore_index=True)

    # Order exactly as requested
    ordered = [
        "Symbol",
        "Name",
        "Operating CF TTM",
        "Investing CF TTM",
        "Financing CF TTM",
        "FCF TTM",
        "CAPEX TTM",
    ]
    for c in ordered:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[ordered]

    if csv_path:
        df.to_csv(csv_path, index=False, encoding="utf-8")

    return df




# run_fetch_dse_cashflow.py

# from safe_requests import create_session
# from fetch_dse_cashflow import fetch_dse_cashflow_dataframe

if __name__ == "__main__":
    session = create_session(
        max_calls=2, per_seconds=1.0, burst=2,
        max_attempts=6, backoff_min=0.5, backoff_max=30.0,
        user_agent="dse-cashflow-fetcher/1.0 (+https://example.org)",
    )

    # 1) Without saving
    df = fetch_dse_cashflow_dataframe(page_size=100, session=session)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print(df.head(10).to_string(index=False))

    # 2) With CSV save
    df2 = fetch_dse_cashflow_dataframe(
        page_size=100,
        session=session,
        csv_path="dse_cashflow.csv",
    )
    print("Saved CSV: dse_cashflow.csv")

    # Basic sanity checks
    assert len(df2) > 0, "No rows returned"
    for col in [
        "Symbol","Name","Operating CF TTM","Investing CF TTM",
        "Financing CF TTM","FCF TTM","CAPEX TTM",
    ]:
        assert col in df2.columns, f"Missing column: {col}"