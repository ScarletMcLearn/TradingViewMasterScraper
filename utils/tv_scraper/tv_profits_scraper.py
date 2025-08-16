# fetch_dse_screener.py

from __future__ import annotations


from utils.requests.requests_wrapper import create_session, SafeSession

# tv_bangladesh_profits.py
# Requirements: pandas, requests, tenacity, and your safe_requests.py in the same directory.

import uuid
import logging
from typing import List, Dict, Optional

import pandas as pd
# from safe_requests import create_session  # your rate-limited, retrying wrapper

TV_URL = "https://scanner.tradingview.com/bangladesh/scan?label-product=screener-stock"

# EXACT column keys from your payload (order matters)
TV_COLUMNS = [
    "name",                               # 0 -> short symbol (e.g., "GP")
    "description",                        # 1 -> company name
    "logoid",                             # 2
    "update_mode",                        # 3
    "type",                               # 4
    "typespecs",                          # 5
    "gross_margin_ttm",                   # 6
    "operating_margin_ttm",               # 7
    "pre_tax_margin_ttm",                 # 8
    "net_margin_ttm",                     # 9
    "free_cash_flow_margin_ttm",          # 10
    "return_on_assets_fq",                # 11 (you labeled as “ROA TTM” in the DF)
    "return_on_equity_fq",                # 12 (you labeled as “ROE TTM”)
    "return_on_invested_capital_fq",      # 13 (you labeled as “ROIC TTM”)
    "research_and_dev_ratio_ttm",         # 14
    "sell_gen_admin_exp_other_ratio_ttm", # 15
    "exchange",                           # 16 -> e.g., "DSEBD"
]

# Map indices in the "d" array to your final DataFrame column names
IDX_TO_FINAL = {
    0:  "Symbol",
    1:  "Name",
    6:  "Gross Margin TTM",
    7:  "Operating Margin TTM",
    8:  "Pretax Margin TTM",
    9:  "Net Margin TTM",
    10: "FCF Margin TTM",
    11: "ROA TTM",
    12: "ROE TTM",
    13: "ROIC TTM",
    14: "RAD Ratio TTM",
    15: "SGAA Ratio TTM",
    # 16 is exchange → guard only
}

FINAL_COLUMNS = [
    "Symbol", "Name",
    "Gross Margin TTM", "Operating Margin TTM", "Pretax Margin TTM", "Net Margin TTM",
    "FCF Margin TTM", "ROA TTM", "ROE TTM", "ROIC TTM",
    "RAD Ratio TTM", "SGAA Ratio TTM",
]

COMMON_HEADERS = {
    "Content-Type": "application/json",
    "Origin": "https://www.tradingview.com",
    "Referer": "https://www.tradingview.com/markets/stocks-bangladesh/market-movers-all-stocks/",
}

def _build_payload(offset: int, page_size: int) -> Dict:
    """Exact payload, paginated via half-open range [offset, offset + page_size)."""
    return {
        "columns": TV_COLUMNS,
        "filter": [{"left": "is_primary", "operation": "equal", "right": True}],
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
    """POST one page; return raw 'data' list."""
    headers = {**COMMON_HEADERS, "Idempotency-Key": str(uuid.uuid4())}
    resp = session.post(
        TV_URL,
        json=_build_payload(offset, page_size),
        headers=headers,
        allow_unsafe_method_retry=True,   # safe: read-like scan
        raise_on_status=False,            # don’t raise; keep paginating even if a page 4xx's
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
    """Map each raw item to our final schema; keep only DSEBD rows."""
    out: List[dict] = []
    for item in raw_items:
        d = item.get("d") or []
        if len(d) < len(TV_COLUMNS):
            continue
        # Ensure DSEBD (index 16)
        if d[16] and d[16] != "DSEBD":
            continue
        row = {final: d[idx] for idx, final in IDX_TO_FINAL.items()}
        out.append(row)
    return out

def fetch_all_bangladesh_stocks_profits(page_size: int = 400,
                                        max_total: Optional[int] = None,
                                        csv_path: Optional[str] = None,
                                        debug: bool = False) -> pd.DataFrame:
    """
    Fetch all Bangladesh stocks (profits view), paginate across all pages, and return a DataFrame:
      Symbol, Name, Gross/Operating/Pretax/Net Margin TTM, FCF Margin TTM,
      ROA/ROE/ROIC, RAD Ratio TTM, SGAA Ratio TTM.
    If `csv_path` is provided, also saves CSV (no index).
    """
    # One shared session => shared token bucket & retries across pages
    s = create_session(
        max_calls=4, per_seconds=1.0, burst=6,         # polite shared rate limit
        max_attempts=6, backoff_min=0.5, backoff_max=10.0,
        user_agent="bd-screener-profits/1.0 (+https://example.com)",
    )

    page_size = max(1, min(int(page_size), 400))
    all_records: List[dict] = []
    offset = 0

    while True:
        if max_total is not None and offset >= max_total:
            break

        curr_size = page_size if max_total is None else min(page_size, max_total - offset)
        raw = _fetch_page(s, offset=offset, page_size=curr_size, debug=debug)
        if not raw:
            break

        recs = _rows_to_records(raw)
        if not recs:
            break

        all_records.extend(recs)
        if len(raw) < curr_size:  # short page => end
            break
        offset += curr_size

    if not all_records:
        df = pd.DataFrame(columns=FINAL_COLUMNS)
    else:
        df = pd.DataFrame(all_records, columns=FINAL_COLUMNS)
        # Coerce numeric columns
        for col in [
            "Gross Margin TTM", "Operating Margin TTM", "Pretax Margin TTM", "Net Margin TTM",
            "FCF Margin TTM", "ROA TTM", "ROE TTM", "ROIC TTM",
            "RAD Ratio TTM", "SGAA Ratio TTM",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values(["Symbol"]).reset_index(drop=True)

    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV -> {csv_path}  (rows: {len(df)})")

    return df

# ---------- Example run ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    df = fetch_all_bangladesh_stocks_profits(
        page_size=400,
        # csv_path="bangladesh_profits.csv",  # pass a filename to save; omit to skip saving
        debug=True
    )
    print(df.head(10).to_string(index=False))
    print("\nRows fetched:", len(df))