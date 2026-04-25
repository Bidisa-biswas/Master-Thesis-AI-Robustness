

# python
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.data_loader import SP500DataLoader


def inspect_df_dates(df: pd.DataFrame, loader: SP500DataLoader) -> Dict[str, Any]:
    # Get datetime index for checks (handle MultiIndex with Date level)
    idx = df.index
    if isinstance(idx, pd.MultiIndex):
        if "Date" in idx.names:
            dt = pd.to_datetime(idx.get_level_values("Date"))
        else:
            # assume second level is date-like (common: (symbol, Date))
            dt = pd.to_datetime(idx.get_level_values(-1))
    else:
        dt = pd.to_datetime(idx)

    dt = dt.sort_values()
    min_date, max_date = dt.min(), dt.max()

    # Full business-day calendar between min and max (use business days)
    full = pd.date_range(start=min_date, end=max_date, freq="B")
    missing = full.difference(dt.normalize())
    missing_pct = (len(missing) / len(full)) if len(full) > 0 else None

    return {
        "rows": len(df),
        "date_min": str(min_date.date()) if pd.notnull(min_date) else None,
        "date_max": str(max_date.date()) if pd.notnull(max_date) else None,
        "expected_start": loader.start_date,
        "expected_end": loader.end_date,
        "date_index_type": type(idx).__name__,
        "business_days_in_range": len(full),
        "missing_business_days": int(len(missing)),
        "missing_business_days_pct": missing_pct,
        "duplicate_index_rows": int(df.index.duplicated().sum()),
    }


def inspect_symbol(loader: SP500DataLoader, symbol: str, force_refresh: bool) -> Dict[str, Any]:
    report: Dict[str, Any] = {"symbol": symbol, "cache_path": str(loader.cache_path_for_symbol(symbol)), "cached": False}
    df = loader.load_cache(symbol)
    if df is None:
        report["cached"] = False
        if force_refresh:
            # will fetch remote and save to cache (may need network)
            df = loader.load_symbol(symbol, force_refresh=True)
        else:
            report["note"] = "cache_missing"
            return report
    else:
        report["cached"] = True

    # Summary stats
    report.update(inspect_df_dates(df, loader))

    # Columns, dtypes, NaN counts
    report["columns"] = list(df.columns)
    report["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    nan_counts = df.isnull().sum()
    report["nan_counts"] = {col: int(cnt) for col, cnt in nan_counts.items()}
    report["any_nan"] = bool(df.isnull().any().any())

    # few sample rows
    report["head"] = df.head(3).to_dict(orient="list")
    report["tail"] = df.tail(3).to_dict(orient="list")

    return report


def main(argv: List[str] = None) -> None:
    p = argparse.ArgumentParser(description="Inspect cached S&P dataset")
    p.add_argument("--symbols", help="Comma-separated list of symbols to inspect (default: loader.symbols)", default=None)
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--cache-dir", default=None, help="Override cache directory path")
    p.add_argument("--force-refresh", action="store_true", help="If cache missing, fetch remotely")
    p.add_argument("--output", default="./dataset_report.json", help="Output JSON report path")
    args = p.parse_args(argv)

    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    loader = SP500DataLoader(
        symbols=symbols,
        start_year=args.start_year,
        end_year=args.end_year,
        cache_dir=cache_dir,
        use_cache=True,
    )

    # If user didn't pass symbols, use the loader's configured symbols
    inspect_list = loader.symbols

    all_reports = []
    for sym in inspect_list:
        try:
            rep = inspect_symbol(loader, sym, force_refresh=args.force_refresh)
        except Exception as e:
            rep = {"symbol": sym, "error": str(e)}
        all_reports.append(rep)

    out_path = Path(args.output)
    out_path.write_text(json.dumps({"summary": all_reports}, indent=2))
    print(f"Report written to `{out_path}`")
    for r in all_reports:
        cached = r.get("cached", False)
        print(f"{r.get('symbol')}: cached={cached}, rows={r.get('rows')}, missing_days={r.get('missing_business_days')}")


if __name__ == "__main__":
    main()