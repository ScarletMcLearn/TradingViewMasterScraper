# fetch_dse_screener.py

from __future__ import annotations


from utils.requests.requests_wrapper import create_session, SafeSession

# fetch_dse_per_share.py


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
    "revenue_per_share_ttm",             # 6  -> Revenue per share TTM
    "fundamental_currency_code",         # 7
    "earnings_per_share_basic_ttm",      # 8  -> EPS basic TTM
    "earnings_per_share_diluted_ttm",    # 9  -> EPS dil TTM
    "operating_cash_flow_per_share_ttm", # 10 -> Operating cash flow per share TTM
    "free_cash_flow_per_share_ttm",      # 11 -> FCF per share TTM
    "ebit_per_share_ttm",                # 12 -> EBIT per share TTM
    "ebitda_per_share_ttm",              # 13 -> EBITDA per share TTM
    "book_value_per_share_fq",           # 14 -> Book per share FQ
    "total_debt_per_share_fq",           # 15 -> Total debt per share FQ
    "cash_per_share_fq",                 # 16 -> Cash per share FQ
    "exchange",                          # 17
]

# Map index -> DataFrame column names you requested
IDX_TO_DF = {
    0:  "Symbol",
    1:  "Name",
    6:  "Revenue per share TTM",
    8:  "EPS basic TTM",
    9:  "EPS dil TTM",
    10: "Operating cash flow per share TTM",
    11: "FCF per share TTM",
    12: "EBIT per share TTM",
    13: "EBITDA per share TTM",
    14: "Book per share FQ",
    15: "Total debt per share FQ",
    16: "Cash per share FQ",
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
    # If you also want ticker/currency, uncomment:
    # rec["Ticker"] = item.get("s")
    # rec["Currency"] = arr[7] if len(arr) > 7 else None
    return rec

def fetch_dse_per_share_dataframe(
    page_size: int = 100,
    session: Optional[SafeSession] = None,
    max_pages: Optional[int] = None,
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch ALL Bangladesh stocks per-share metrics with correct pagination.
    Optionally save to CSV when csv_path is provided.
    """
    s = session or create_session(
        max_calls=2, per_seconds=1.0, burst=2,            # global token-bucket
        max_attempts=6, backoff_min=0.5, backoff_max=30,  # robust retries
        user_agent="dse-per-share-fetcher/1.0 (+https://example.org)",
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

    # Exact column order you asked for
    ordered_cols = [
        "Symbol",
        "Name",
        "Revenue per share TTM",
        "EPS basic TTM",
        "EPS dil TTM",
        "Operating cash flow per share TTM",
        "FCF per share TTM",
        "EBIT per share TTM",
        "EBITDA per share TTM",
        "Book per share FQ",
        "Total debt per share FQ",
        "Cash per share FQ",
    ]
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[ordered_cols]

    if csv_path:
        df.to_csv(csv_path, index=False, encoding="utf-8")

    return df



# run_fetch_dse_per_share.py

# from safe_requests import create_session
# from fetch_dse_per_share import fetch_dse_per_share_dataframe

if __name__ == "__main__":
    session = create_session(
        max_calls=2, per_seconds=1.0, burst=2,
        max_attempts=6, backoff_min=0.5, backoff_max=30.0,
        user_agent="dse-per-share-fetcher/1.0 (+https://example.org)",
    )

    # 1) Without saving
    df = fetch_dse_per_share_dataframe(page_size=100, session=session)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print(df.head(10).to_string(index=False))

    # 2) With CSV save
    df2 = fetch_dse_per_share_dataframe(
        page_size=100,
        session=session,
        csv_path="dse_per_share.csv",
    )
    print("Saved CSV: dse_per_share.csv")

    # Quick sanity checks
    assert len(df2) > 0, "No rows returned"
    needed = [
        "Symbol","Name","Revenue per share TTM","EPS basic TTM","EPS dil TTM",
        "Operating cash flow per share TTM","FCF per share TTM","EBIT per share TTM",
        "EBITDA per share TTM","Book per share FQ","Total debt per share FQ","Cash per share FQ",
    ]
    for col in needed:
        assert col in df2.columns, f"Missing column: {col}"
