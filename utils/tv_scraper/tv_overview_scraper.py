# fetch_dse_screener.py

from __future__ import annotations

import math
from typing import List, Optional

import pandas as pd
from utils.requests.requests_wrapper import create_session, SafeSession

TV_URL = "https://scanner.tradingview.com/bangladesh/scan?label-product=screener-stock"

TV_COLUMNS = [
    "name","description","logoid","update_mode","type","typespecs",
    "close","pricescale","minmov","fractional","minmove2","currency",
    "change","volume","relative_volume_10d_calc","market_cap_basic",
    "fundamental_currency_code","price_earnings_ttm","earnings_per_share_diluted_ttm",
    "earnings_per_share_diluted_yoy_growth_ttm","dividends_yield_current",
    "sector.tr","market","sector","AnalystRating","AnalystRating.tr","exchange",
]

INDEX_TO_DF_FIELD = {
    0:  "Symbol",
    1:  "Name",
    6:  "Price",
    12: "Change %",
    13: "Volume",
    14: "Relative Volume",
    15: "Market Capital",
    17: "P/E",
    18: "EPS dil TTM",
    19: "EPS dil growth TTM YoY",
    20: "Div Yield % TTM",
    21: "Sector",
    24: "Analyst Rating",
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
                        {"expression": {"left": "typespecs", "operation": "has", "right": ["common"]}}]}},
                    {"operation": {"operator": "and", "operands": [
                        {"expression": {"left": "type", "operation": "equal", "right": "stock"}},
                        {"expression": {"left": "typespecs", "operation": "has", "right": ["preferred"]}}]}},
                    {"operation": {"operator": "and", "operands": [
                        {"expression": {"left": "type", "operation": "equal", "right": "dr"}}]}},
                    {"operation": {"operator": "and", "operands": [
                        {"expression": {"left": "type", "operation": "equal", "right": "fund"}},
                        {"expression": {"left": "typespecs", "operation": "has_none_of", "right": ["etf"]}}]}}
                ]}}]
    },
}

def _payload_with_range(start: int, end: int) -> dict:
    p = dict(BASE_PAYLOAD)
    p["range"] = [start, end]  # IMPORTANT: [from, to] (end-exclusive)
    return p

def _row_to_record(item: dict) -> dict:
    arr = item.get("d", []) or []
    rec = {}
    for idx, name in INDEX_TO_DF_FIELD.items():
        rec[name] = arr[idx] if idx < len(arr) else None
    rec["Ticker"] = item.get("s")                   # e.g. "DSEBD:GP"
    rec["Currency"] = arr[11] if len(arr) > 11 else None
    return rec

def fetch_dse_screener_dataframe(
    page_size: int = 100,
    session: Optional[SafeSession] = None,
    max_pages: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch all DSE screener data with proper pagination (range=[from, to]),
    using a global rate-limited SafeSession.
    """
    s = session or create_session(
        max_calls=2, per_seconds=1.0, burst=2,
        max_attempts=6, backoff_min=0.5, backoff_max=30.0,
        user_agent="dse-screener-fetcher/1.0 (+https://example.org)",
    )

    # First call to learn totalCount
    start = 0
    end = page_size
    r = s.post(TV_URL, json=_payload_with_range(start, end), timeout=15)
    js = r.json()
    total = int(js.get("totalCount", 0))
    data = js.get("data", []) or []

    if total <= 0 or not data:
        cols = list(INDEX_TO_DF_FIELD.values()) + ["Ticker", "Currency"]
        return pd.DataFrame(columns=cols)

    chunks: List[pd.DataFrame] = []
    chunks.append(pd.DataFrame.from_records([_row_to_record(it) for it in data]))

    # How many more pages?
    pages_total = math.ceil(total / page_size)
    if max_pages is not None:
        pages_total = min(pages_total, max_pages)

    # Pull remaining pages
    for page_index in range(1, pages_total):
        start = page_index * page_size
        end = min(start + page_size, total)
        r = s.post(TV_URL, json=_payload_with_range(start, end), timeout=15)
        js = r.json()
        page_data = js.get("data", []) or []
        if not page_data:
            break
        chunks.append(pd.DataFrame.from_records([_row_to_record(it) for it in page_data]))

    df = pd.concat(chunks, ignore_index=True)

    preferred_cols = [
        "Symbol", "Name", "Price", "Change %", "Volume", "Relative Volume",
        "Market Capital", "P/E", "EPS dil TTM", "EPS dil growth TTM YoY",
        "Div Yield % TTM", "Sector", "Analyst Rating", "Ticker", "Currency",
    ]
    for c in preferred_cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[preferred_cols]



# run_fetch_dse.py

# from safe_requests import create_session
# from fetch_dse_screener import fetch_dse_screener_dataframe

if __name__ == "__main__":
    session = create_session(
        max_calls=2, per_seconds=1.0, burst=2,
        max_attempts=6, backoff_min=0.5, backoff_max=30.0,
        user_agent="dse-screener-fetcher/1.0 (+https://example.org)",
    )

    df = fetch_dse_screener_dataframe(page_size=100, session=session)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print(df.head(10).to_string(index=False))

    # Optional: sanity assertions
    assert len(df) > 0, "No rows returned"
    # If totalCount is known/suspected, you can assert lower bound
    # assert len(df) >= 300

    df.to_csv("dse_screener_all.csv", index=False)