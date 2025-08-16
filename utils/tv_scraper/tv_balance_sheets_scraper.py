# fetch_dse_screener.py

from __future__ import annotations


from utils.requests.requests_wrapper import create_session, SafeSession

# fetch_dse_balance_sheet.py

import math
from typing import List, Optional

import pandas as pd
# from safe_requests import create_session, SafeSession


TV_URL = "https://scanner.tradingview.com/bangladesh/scan?label-product=screener-stock"

# Columns requested from the API (index positions matter!)
TV_COLUMNS = [
    "name",                              # 0
    "description",                       # 1
    "logoid",                            # 2
    "update_mode",                       # 3
    "type",                              # 4
    "typespecs",                         # 5
    "total_assets_fq",                   # 6
    "fundamental_currency_code",         # 7
    "total_current_assets_fq",           # 8
    "cash_n_short_term_invest_fq",       # 9
    "total_liabilities_fq",              # 10
    "total_debt_fq",                     # 11
    "net_debt_fq",                       # 12
    "total_equity_fq",                   # 13
    "current_ratio_fq",                  # 14
    "quick_ratio_fq",                    # 15
    "debt_to_equity_fq",                 # 16
    "cash_n_short_term_invest_to_total_debt_fq",  # 17
    "exchange",                          # 18
]

# Map index -> DataFrame column name you asked for
IDX_TO_DF = {
    0:  "Symbol",
    1:  "Name",
    6:  "Total Assets FQ",
    8:  "Current Assets FQ",
    9:  "Cash on Hand FQ",
    10: "Total Liabilities FQ",
    11: "Total Debt FQ",
    12: "Net Debt FQ",
    13: "Total equity FQ",
    14: "Current ratio FQ",
    15: "Quick ratio FQ",
    16: "Debt/Equity FQ",
    17: "Cash/Debt FQ",
}

# Static parts of the request body (filters & sort)
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
    p = dict(BASE_PAYLOAD)
    p["range"] = [start, end]  # IMPORTANT: [from, to) end-exclusive
    return p

def _row_to_record(item: dict) -> dict:
    arr = item.get("d", []) or []
    rec = {}
    for idx, out_name in IDX_TO_DF.items():
        rec[out_name] = arr[idx] if idx < len(arr) else None
    # If you ever want to keep ticker/currency:
    # rec["Ticker"] = item.get("s")
    # rec["Currency"] = arr[7] if len(arr) > 7 else None
    return rec

def fetch_dse_balance_sheet_dataframe(
    page_size: int = 100,
    session: Optional[SafeSession] = None,
    max_pages: Optional[int] = None,
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch ALL Bangladesh stocks balance-sheet fields from TradingView with proper pagination.

    Args:
        page_size: number of rows per page (100 is safe).
        session: pass a shared SafeSession to apply a global rate limit; if None, one is created.
        max_pages: optional safety cap; None pulls all pages.
        csv_path: if provided, saves the final DataFrame to this CSV path.

    Returns:
        pandas.DataFrame with requested columns in your order.
    """
    s = session or create_session(
        max_calls=2, per_seconds=1.0, burst=2,            # global rate limit
        max_attempts=6, backoff_min=0.5, backoff_max=30,  # robust retries
        user_agent="dse-balance-sheet-fetcher/1.0 (+https://example.org)",
    )

    # Initial request to learn totalCount
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

    # Column order exactly as requested
    ordered_cols = [
        "Symbol",
        "Name",
        "Total Assets FQ",
        "Current Assets FQ",
        "Cash on Hand FQ",
        "Total Liabilities FQ",
        "Total Debt FQ",
        "Net Debt FQ",
        "Total equity FQ",
        "Current ratio FQ",
        "Quick ratio FQ",
        "Debt/Equity FQ",
        "Cash/Debt FQ",
    ]
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[ordered_cols]

    if csv_path:
        df.to_csv(csv_path, index=False, encoding="utf-8")

    return df




# run_fetch_dse_balance_sheet.py

# from safe_requests import create_session
# from fetch_dse_balance_sheet import fetch_dse_balance_sheet_dataframe

if __name__ == "__main__":
    # Share one global SafeSession so rate limiting applies across your app
    session = create_session(
        max_calls=2, per_seconds=1.0, burst=2,
        max_attempts=6, backoff_min=0.5, backoff_max=30.0,
        user_agent="dse-balance-sheet-fetcher/1.0 (+https://example.org)",
    )

    # Without saving
    df = fetch_dse_balance_sheet_dataframe(page_size=100, session=session)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print(df.head(10).to_string(index=False))

    # With CSV save
    df2 = fetch_dse_balance_sheet_dataframe(
        page_size=100,
        session=session,
        csv_path="dse_balance_sheet.csv",
    )
    print("Saved CSV:", "dse_balance_sheet.csv")
    # Sanity checks
    assert len(df2) > 0, "No rows returned"
    for col in [
        "Symbol","Name","Total Assets FQ","Current Assets FQ","Cash on Hand FQ",
        "Total Liabilities FQ","Total Debt FQ","Net Debt FQ","Total equity FQ",
        "Current ratio FQ","Quick ratio FQ","Debt/Equity FQ","Cash/Debt FQ",
    ]:
        assert col in df2.columns, f"Missing column: {col}"