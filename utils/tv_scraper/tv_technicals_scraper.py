# fetch_dse_screener.py

from __future__ import annotations


from utils.requests.requests_wrapper import create_session, SafeSession

# tv_bangladesh_technical.py
# Requirements: pandas, requests, tenacity, and your safe_requests.py in the same directory.

from __future__ import annotations
import uuid
import logging
from typing import List, Dict, Optional

import pandas as pd
# from safe_requests import create_session  # your rate-limited, retrying wrapper

TV_URL = "https://scanner.tradingview.com/bangladesh/scan?label-product=screener-stock"

# EXACT column keys from your payload (order matters)
TV_COLUMNS = [
    "name", "description", "logoid", "update_mode", "type", "typespecs",
    "TechRating_1D",           # 6  (numeric)
    "TechRating_1D.tr",        # 7  (human text -> use this)
    "MARating_1D",             # 8  (numeric)
    "MARating_1D.tr",          # 9  (human text -> use this)
    "OsRating_1D",             # 10 (numeric)
    "OsRating_1D.tr",          # 11 (human text -> use this)
    "RSI",                     # 12
    "Mom",                     # 13
    "pricescale",              # 14
    "minmov",                  # 15
    "fractional",              # 16
    "minmove2",                # 17
    "AO",                      # 18
    "CCI20",                   # 19
    "Stoch.K",                 # 20
    "Stoch.D",                 # 21
    "Candle.3BlackCrows",      # 22
    "Candle.3WhiteSoldiers",   # 23
    "Candle.AbandonedBaby.Bearish",  # 24
    "Candle.AbandonedBaby.Bullish",  # 25
    "Candle.Doji",             # 26
    "Candle.Doji.Dragonfly",   # 27
    "Candle.Doji.Gravestone",  # 28
    "Candle.Engulfing.Bearish",# 29
    "Candle.Engulfing.Bullish",# 30
    "Candle.EveningStar",      # 31
    "Candle.Hammer",           # 32
    "Candle.HangingMan",       # 33
    "Candle.Harami.Bearish",   # 34
    "Candle.Harami.Bullish",   # 35
    "Candle.InvertedHammer",   # 36
    "Candle.Kicking.Bearish",  # 37
    "Candle.Kicking.Bullish",  # 38
    "Candle.LongShadow.Lower", # 39
    "Candle.LongShadow.Upper", # 40
    "Candle.Marubozu.Black",   # 41
    "Candle.Marubozu.White",   # 42
    "Candle.MorningStar",      # 43
    "Candle.ShootingStar",     # 44
    "Candle.SpinningTop.Black",# 45
    "Candle.SpinningTop.White",# 46
    "Candle.TriStar.Bearish",  # 47
    "Candle.TriStar.Bullish",  # 48
    "exchange",                # 49
]

# Human labels for candle flags (index -> label)
CANDLE_MAP = {
    22: "3 Black Crows",
    23: "3 White Soldiers",
    24: "Abandoned Baby (Bearish)",
    25: "Abandoned Baby (Bullish)",
    26: "Doji",
    27: "Dragonfly Doji",
    28: "Gravestone Doji",
    29: "Bearish Engulfing",
    30: "Bullish Engulfing",
    31: "Evening Star",
    32: "Hammer",
    33: "Hanging Man",
    34: "Bearish Harami",
    35: "Bullish Harami",
    36: "Inverted Hammer",
    37: "Bearish Kicking",
    38: "Bullish Kicking",
    39: "Long Lower Shadow",
    40: "Long Upper Shadow",
    41: "Black Marubozu",
    42: "White Marubozu",
    43: "Morning Star",
    44: "Shooting Star",
    45: "Black Spinning Top",
    46: "White Spinning Top",
    47: "Bearish Tri-Star",
    48: "Bullish Tri-Star",
}

# Final columns to present (compact view + computed “Patterns”)
FINAL_COLUMNS = [
    "Symbol", "Name",
    "Tech Rating", "MA Rating", "Os Rating",
    "RSI (14)", "Mom (10)", "AO", "CCI (20)",
    "Stoch (14,3,3) %K", "Stoch (14,3,3) %D",
    "Patterns",
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
                            {"operation": {"operator": "and", "operands": [
                                {"expression": {"left": "type", "operation": "equal", "right": "dr"}}
                            ]}},
                            {"operation": {"operator": "and", "operands": [
                                {"expression": {"left": "type", "operation": "equal", "right": "fund"}},
                                {"expression": {"left": "typespecs", "operation": "has_none_of", "right": ["etf"]}}
                            ]}},
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
        allow_unsafe_method_retry=True,  # safe: read-like scan
        raise_on_status=False,           # keep paginating even if a page 4xx's
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

def _compute_patterns(d_row: list) -> str:
    """
    Build a comma-separated list of candle pattern labels for which the flag == 1.
    If none are present, return an empty string.
    """
    names = []
    for idx, label in CANDLE_MAP.items():
        try:
            val = d_row[idx]
        except IndexError:
            continue
        # TV flags are typically 0/1 for absent/present
        # (some indicators use -1/1 for polarity, but these are already split into bull/bear cols)
        if val == 1:
            names.append(label)
    return ", ".join(names)

def _rows_to_records(raw_items: list, include_candle_columns: bool) -> List[dict]:
    """Map each raw item to our final schema; keep only DSEBD rows."""
    out: List[dict] = []
    for item in raw_items:
        d = item.get("d") or []
        if len(d) < len(TV_COLUMNS):
            continue
        # Ensure DSEBD at index 49
        if d[49] and d[49] != "DSEBD":
            continue

        row = {
            "Symbol": d[0],
            "Name": d[1],
            "Tech Rating": d[7],  # human-readable
            "MA Rating":   d[9],
            "Os Rating":   d[11],
            "RSI (14)":    d[12],
            "Mom (10)":    d[13],
            "AO":          d[18],
            "CCI (20)":    d[19],
            "Stoch (14,3,3) %K": d[20],
            "Stoch (14,3,3) %D": d[21],
            "Patterns": _compute_patterns(d),
        }

        if include_candle_columns:
            # Add one boolean/int column per candle flag
            for idx, label in CANDLE_MAP.items():
                row[label] = d[idx]

        out.append(row)
    return out

def fetch_all_bangladesh_stocks_technical(page_size: int = 400,
                                          max_total: Optional[int] = None,
                                          csv_path: Optional[str] = None,
                                          debug: bool = False,
                                          include_candle_columns: bool = False) -> pd.DataFrame:
    """
    Fetch all Bangladesh stocks (technical view), paginate across all pages, and return a DataFrame:
      Symbol, Name, Tech/MA/Os Ratings (human text), RSI(14), Mom(10), AO, CCI(20),
      Stoch(14,3,3) %K/%D, and a computed “Patterns” column listing all detected candles.

    If `include_candle_columns=True`, adds one column per candle flag for deeper analysis.
    If `csv_path` is provided, saves CSV (no index).
    """
    # One shared session => shared token bucket & retries across pages
    s = create_session(
        max_calls=4, per_seconds=1.0, burst=6,         # polite shared rate limit
        max_attempts=6, backoff_min=0.5, backoff_max=10.0,
        user_agent="bd-screener-technical/1.1 (+https://example.com)",
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

        recs = _rows_to_records(raw, include_candle_columns=include_candle_columns)
        if not recs:
            break

        all_records.extend(recs)
        if len(raw) < curr_size:  # short page => end
            break
        offset += curr_size

    if not all_records:
        df = pd.DataFrame(columns=FINAL_COLUMNS if not include_candle_columns
                          else FINAL_COLUMNS + list(CANDLE_MAP.values()))
    else:
        # Build DF with optional candle columns
        base_cols = FINAL_COLUMNS if not include_candle_columns \
            else FINAL_COLUMNS + [c for c in CANDLE_MAP.values()]
        df = pd.DataFrame(all_records)
        # Ensure column order (add any missing columns if some patterns never appear)
        for col in base_cols:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[base_cols]

        # Coerce numeric indicators
        for col in ["RSI (14)", "Mom (10)", "AO", "CCI (20)", "Stoch (14,3,3) %K", "Stoch (14,3,3) %D"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # If candle columns are included, make them ints where possible
        if include_candle_columns:
            for col in CANDLE_MAP.values():
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        df = df.sort_values(["Symbol"]).reset_index(drop=True)

    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV -> {csv_path}  (rows: {len(df)})")

    return df

# ---------- Example run ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    df = fetch_all_bangladesh_stocks_technical(
        page_size=400,
        # csv_path="bangladesh_technical.csv",  # pass a filename to save; omit to skip saving
        include_candle_columns=False,          # set True to include one column per candle flag
        debug=True
    )
    print(df.head(10).to_string(index=False))
    print("\nRows fetched:", len(df))