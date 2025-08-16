# TradingViewAllScreenerScraper/run_all_screener_scrapes.py
# Run with:  python -m TradingViewAllScreenerScraper.run_all_screener_scrapes

from __future__ import annotations
import os
import time
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

# ---- Your SafeSession wrapper ----
from utils.requests.requests_wrapper import create_session

# ---- Import all fetchers ----
# (Modules that accept an injected session)
from utils.tv_scraper.tv_overview_scraper import fetch_dse_screener_dataframe
from utils.tv_scraper.tv_balance_sheets_scraper import fetch_dse_balance_sheet_dataframe
from utils.tv_scraper.tv_cashflow_scraper import fetch_dse_cashflow_dataframe
from utils.tv_scraper.tv_per_share_scraper import fetch_dse_per_share_dataframe

# (Modules that manage their own session internally but accept csv_path)
from utils.tv_scraper.tv_performance_scraper import fetch_all_bangladesh_stocks_performance
from utils.tv_scraper.tv_hours_scraper import fetch_all_bangladesh_stocks_extended_hours
from utils.tv_scraper.tv_dividends_scraper import fetch_all_bangladesh_stocks_dividends
from utils.tv_scraper.tv_valuation_scraper import fetch_all_bangladesh_stocks_valuation
from utils.tv_scraper.tv_profits_scraper import fetch_all_bangladesh_stocks_profits
from utils.tv_scraper.tv_income_statements_scraper import fetch_all_bangladesh_stocks_income_statement
from utils.tv_scraper.tv_technicals_scraper import fetch_all_bangladesh_stocks_technical


# ---------------------------
# Logging setup
# ---------------------------
def _build_logger(date_str: str) -> logging.Logger:
    os.makedirs("data", exist_ok=True)
    logger = logging.getLogger("tv_runner")
    logger.setLevel(logging.INFO)
    logger.handlers[:] = []  # avoid dupes if run multiple times

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join("data", f"tv_runner_{date_str}.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------
# Helpers
# ---------------------------
def today_yyyy_mm_dd_dhaka() -> str:
    """Return today's date string in Asia/Dhaka timezone."""
    tz = ZoneInfo("Asia/Dhaka")
    return datetime.now(tz).strftime("%Y-%m-%d")


def save_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


@dataclass
class Task:
    name: str
    outfile_stem: str
    fn: Callable[..., pd.DataFrame]
    kwargs: Dict
    uses_injected_session: bool = False  # True if we pass the shared session into the fn


# ---------------------------
# Main runner
# ---------------------------
def main() -> int:
    date_str = today_yyyy_mm_dd_dhaka()
    logger = _build_logger(date_str)

    # Shared SafeSession for scrapers that accept a session (overview/balance/cashflow/per-share, etc.)
    session = create_session(
        max_calls=2, per_seconds=1.0, burst=2,
        max_attempts=6, backoff_min=0.5, backoff_max=30.0,
        user_agent="tv-dse-runner/1.0 (+https://example.org)",
    )

    # Build the task list
    tasks = [
        # ---- Accept injected session (we’ll save ourselves) ----
        Task(
            name="Overview (All)",
            outfile_stem="dse_screener_all",
            fn=fetch_dse_screener_dataframe,
            kwargs={"page_size": 100, "session": session},
            uses_injected_session=True,
        ),
        Task(
            name="Balance Sheets",
            outfile_stem="dse_balance_sheet",
            fn=fetch_dse_balance_sheet_dataframe,
            kwargs={"page_size": 100, "session": session},
            uses_injected_session=True,
        ),
        Task(
            name="Cash Flow",
            outfile_stem="dse_cashflow",
            fn=fetch_dse_cashflow_dataframe,
            kwargs={"page_size": 100, "session": session},
            uses_injected_session=True,
        ),
        Task(
            name="Per Share",
            outfile_stem="dse_per_share",
            fn=fetch_dse_per_share_dataframe,
            kwargs={"page_size": 100, "session": session},
            uses_injected_session=True,
        ),

        # ---- Manage their own session internally (we pass csv_path) ----
        Task(
            name="Performance",
            outfile_stem="bangladesh_performance",
            fn=fetch_all_bangladesh_stocks_performance,
            kwargs={"page_size": 200, "debug": False},  # csv_path added at runtime
        ),
        Task(
            name="Extended Hours",
            outfile_stem="bangladesh_extended_hours",
            fn=fetch_all_bangladesh_stocks_extended_hours,
            kwargs={"page_size": 300, "debug": False},
        ),
        Task(
            name="Dividends",
            outfile_stem="bangladesh_dividends",
            fn=fetch_all_bangladesh_stocks_dividends,
            kwargs={"page_size": 400, "debug": False},
        ),
        Task(
            name="Valuation",
            outfile_stem="bangladesh_valuation",
            fn=fetch_all_bangladesh_stocks_valuation,
            kwargs={"page_size": 400, "debug": False},
        ),
        Task(
            name="Profits",
            outfile_stem="bangladesh_profits",
            fn=fetch_all_bangladesh_stocks_profits,
            kwargs={"page_size": 400, "debug": False},
        ),
        Task(
            name="Income Statement",
            outfile_stem="bangladesh_income_statement",
            fn=fetch_all_bangladesh_stocks_income_statement,
            kwargs={"page_size": 400, "debug": False},
        ),
        Task(
            name="Technicals",
            outfile_stem="bangladesh_technical",
            fn=fetch_all_bangladesh_stocks_technical,
            kwargs={"page_size": 400, "include_candle_columns": False, "debug": False},
        ),
    ]

    failures = 0
    total_rows = 0
    t0 = time.perf_counter()

    for task in tasks:
        out_path = os.path.join("data", f"{task.outfile_stem}_{date_str}.csv")
        logger.info("▶ START: %s -> %s", task.name, out_path)
        t_start = time.perf_counter()

        try:
            # For scrapers that accept csv_path, pass it so they save internally (and still return df)
            # For scrapers that accept session, we save the returned df ourselves right after.
            if "csv_path" in task.fn.__code__.co_varnames:
                df = task.fn(**{**task.kwargs, "csv_path": out_path})
            else:
                df = task.fn(**task.kwargs)
                save_df(df, out_path)

            rows = len(df) if isinstance(df, pd.DataFrame) else 0
            total_rows += rows
            elapsed = time.perf_counter() - t_start
            logger.info("✔ DONE: %s | rows=%s | elapsed=%.2fs | saved=%s",
                        task.name, rows, elapsed, out_path)

        except Exception as e:
            failures += 1
            elapsed = time.perf_counter() - t_start
            logger.exception("✖ FAIL: %s | elapsed=%.2fs | error=%s", task.name, elapsed, e)

    total_elapsed = time.perf_counter() - t0
    logger.info("SUMMARY: tasks=%d | failures=%d | total_rows=%d | total_time=%.2fs",
                len(tasks), failures, total_rows, total_elapsed)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
