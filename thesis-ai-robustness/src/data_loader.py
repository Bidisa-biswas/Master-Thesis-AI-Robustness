"""
thesis-ai-robustness/src/data_loader.py

Complete data loader for S&P 500 data between 2010 and 2024.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import yfinance as yf
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SP500DataLoader:
    """
    Data loader for S&P 500 historical data.

    Responsibilities (to implement):
    - Fetch remote data (e.g. via yfinance, Alpha Vantage, or a vendor API)
    - Cache downloaded CSV/Parquet locally
    - Load from cache when available
    - Basic preprocessing (datetime index, sorting, missing-value policy)
    - Provide utility to split into train/test

    Attributes:
        symbols: list of tickers to fetch (defaults to '^GSPC' if None)
        start_year: start year (inclusive)
        end_year: end year (inclusive)
        cache_dir: local directory to store cached files
        use_cache: whether to prefer reading cached data
    """
    symbols: Optional[List[str]] = None
    start_year: int = 2010
    end_year: int = 2024
    cache_dir: Optional[Path] = None
    use_cache: bool = True

    def __post_init__(self) -> None:
        if self.symbols is None:
            self.symbols = ["^GSPC"]
        if self.cache_dir is None:
            self.cache_dir = Path(".cache") / "sp500"
        # validate years
        self._validate_years()

    def _validate_years(self) -> None:
        """Validate start_year and end_year; raise ValueError on invalid ranges."""
        if self.start_year > self.end_year:
            raise ValueError("start_year must be <= end_year")
        if self.start_year < 1900 or self.end_year > date.today().year:
            raise ValueError("start_year/end_year out of plausible range")

    @property
    def start_date(self) -> str:
        """Return ISO start date string (YYYY-MM-DD)."""
        return f"{self.start_year}-01-01"

    @property
    def end_date(self) -> str:
        """Return ISO end date string (YYYY-MM-DD)."""
        return f"{self.end_year}-12-31"

    def cache_path_for_symbol(self, symbol: str) -> Path:
        """Return the cache file path for a given symbol (prefer parquet)."""
        filename = f"{symbol}_{self.start_year}_{self.end_year}.parquet"
        return self.cache_dir / filename

    def fetch_remote(self, symbol: str) -> pd.DataFrame:
        """
        Fetch remote data for `symbol` between start_date and end_date.

        Implementation note:
            - Use a chosen provider (yfinance, alphavantage, etc.)
            - Ensure timezone-aware datetimes are converted to naive UTC or local as needed
            - Keep columns: ['Open','High','Low','Close','Adj Close','Volume']
        """
        logger.info(f"Fetching {symbol} from {self.start_date} to {self.end_date}")

        # Download data using yfinance
        df = yf.download(
            symbol,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=False
        )

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        # yfinance returns with Date as index, which is what we want
        return df

    def load_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load cached DataFrame for `symbol` if it exists."""
        cache_path = self.cache_path_for_symbol(symbol)

        if not cache_path.exists():
            return None

        try:
            logger.info(f"Loading {symbol} from cache: {cache_path}")
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache for {symbol}: {e}")
            return None

    def save_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """Save DataFrame to cache directory (create directory if needed)."""
        cache_path = self.cache_path_for_symbol(symbol)

        # Create directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {symbol} to cache: {cache_path}")
        df.to_parquet(cache_path)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing steps to raw DataFrame:
          - Ensure datetime index named 'Date'
          - Sort by index ascending
          - Forward/backward fill or drop missing rows (document policy)
          - Keep consistent column names (e.g., 'Adj Close' -> 'adj_close')
        """
        # Make a copy to avoid modifying original
        df = df.copy()

        # Ensure index is datetime and named
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'

        # Sort by date ascending
        df = df.sort_index()

        # Handle missing values
        if df.isnull().any().any():
            logger.warning(f"Found {df.isnull().sum().sum()} missing values")
            # Forward fill then backward fill for any remaining NaNs
            df = df.ffill().bfill()

        # Rename columns to lowercase with underscores
        df.columns = [col[0].lower().replace(' ', '_') if isinstance(col, tuple) else col.lower().replace(' ', '_')
                      for col in df.columns]

        # Add returns column (will be calculated later, but add placeholder)
        df['returns'] = df['adj_close'].pct_change()

        # Add log returns
        df['log_returns'] = np.log(df['adj_close'] / df['adj_close'].shift(1))

        return df

    def load_symbol(self, symbol: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load data for a single symbol.

        Workflow:
          1. If use_cache and not force_refresh -> try load_cache
          2. If cache missing or force_refresh -> fetch_remote, preprocess, save_cache
          3. Return DataFrame
        """
        # Try cache first
        if self.use_cache and not force_refresh:
            df = self.load_cache(symbol)
            if df is not None:
                return df

        # Fall back to remote fetch
        df = self.fetch_remote(symbol)
        df = self.preprocess(df)

        # Save to cache
        if self.use_cache:
            self.save_cache(symbol, df)

        return df

    def load_all(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load and concatenate data for all configured symbols.

        If multiple symbols, include a column or multiindex to distinguish them.
        Return a single DataFrame indexed by Date (and optionally symbol).
        """
        all_dfs = []

        for symbol in self.symbols:
            df = self.load_symbol(symbol, force_refresh)

            # Keep each symbol's dataframe as-is (with Date index).
            # We'll add the symbol differentiation when concatenating.
            all_dfs.append(df)

        # Combine
        if len(all_dfs) == 1:
            return all_dfs[0]
        else:
            # For multiple symbols, concatenate with multi-index (symbol, Date)
            combined = pd.concat(all_dfs, keys=self.symbols, names=['symbol', 'Date'])
            return combined

    def get_train_test_split(
            self,
            df: pd.DataFrame,
            test_size: Optional[float] = None,
            split_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split df into (train, test).

        Two modes:
          - If split_date is provided (ISO string) split by date (train <= split_date < test)
          - Else if test_size is provided (0 < test_size < 1) split by proportion on time-ordered data

        Return (train_df, test_df)
        """
        # Support both single DatetimeIndex and MultiIndex with a Date level (e.g. ('symbol','Date')).
        # We'll operate on a copy/reset view so original index is preserved on return.

        if split_date is not None:
            split_ts = pd.to_datetime(split_date)

            if isinstance(df.index, pd.MultiIndex):
                # Reset index so Date is a column we can compare against
                df_reset = df.reset_index()

                # Expect a column named 'Date' after reset_index; if not present, try to find datetime column
                if 'Date' not in df_reset.columns:
                    # try find first datetime-like column
                    date_cols = [c for c in df_reset.columns if pd.api.types.is_datetime64_any_dtype(df_reset[c])]
                    if not date_cols:
                        raise ValueError("Could not find a Date column in MultiIndex DataFrame")
                    date_col = date_cols[0]
                else:
                    date_col = 'Date'

                train_reset = df_reset[df_reset[date_col] < split_ts].copy()
                test_reset = df_reset[df_reset[date_col] >= split_ts].copy()

                # Reconstruct the original indexed DataFrames
                train = train_reset.set_index([c for c in df_reset.columns if c not in df.columns]).sort_index()
                test = test_reset.set_index([c for c in df_reset.columns if c not in df.columns]).sort_index()

            else:
                # Single-level DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    raise ValueError("DataFrame index must be a DatetimeIndex or a MultiIndex with a Date level")

                train = df[df.index < split_ts].copy()
                test = df[df.index >= split_ts].copy()

            if train.empty or test.empty:
                raise ValueError("Train or test set is empty after the split_date operation")

            logger.info(f"Split by date: train {train.index.min()} to {train.index.max() if not train.empty else 'EMPTY'}, "
                        f"test {test.index.min() if not test.empty else 'EMPTY'} to {test.index.max()}")

            return train, test

        # Method 2: Split by proportion
        if test_size is not None and 0 < test_size < 1:
            if isinstance(df.index, pd.MultiIndex):
                # Reset, sort by Date column, split, then restore index
                df_reset = df.reset_index()

                # find Date column
                if 'Date' in df_reset.columns:
                    date_col = 'Date'
                else:
                    date_cols = [c for c in df_reset.columns if pd.api.types.is_datetime64_any_dtype(df_reset[c])]
                    if not date_cols:
                        raise ValueError("Could not find a Date column in MultiIndex DataFrame")
                    date_col = date_cols[0]

                df_reset = df_reset.sort_values(by=date_col).reset_index(drop=True)
                n = len(df_reset)
                split_idx = int(n * (1 - test_size))

                train_reset = df_reset.iloc[:split_idx].copy()
                test_reset = df_reset.iloc[split_idx:].copy()

                train = train_reset.set_index([c for c in df_reset.columns if c not in df.columns]).sort_index()
                test = test_reset.set_index([c for c in df_reset.columns if c not in df.columns]).sort_index()

            else:
                n = len(df)
                split_idx = int(n * (1 - test_size))
                train = df.iloc[:split_idx].copy()
                test = df.iloc[split_idx:].copy()

            if train.empty or test.empty:
                raise ValueError("Train or test set is empty after the proportional split")

            logger.info(f"Split by proportion: {len(train)} train, {len(test)} test")
            return train, test

        raise ValueError("Either split_date or test_size must be provided")


def load_sp500(
        symbols: Optional[List[str]] = None,
        start_year: int = 2010,
        end_year: int = 2024,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Convenience function to create loader and load combined S&P data.

    Example:
        loader = SP500DataLoader(symbols=['^GSPC'], start_year=2010, start_year=2024)
        df = loader.load_all()
    """
    loader = SP500DataLoader(
        symbols=symbols, start_year=start_year, end_year=end_year, cache_dir=cache_dir, use_cache=use_cache
    )
    return loader.load_all(force_refresh=force_refresh)


# Example usage:
if __name__ == "__main__":
    import numpy as np

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Test the loader
    loader = SP500DataLoader(symbols=['^GSPC'], start_year=2010, end_year=2024)
    df = loader.load_all(force_refresh=False)

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nColumns: {df.columns.tolist()}")

    # Test split
    train, test = loader.get_train_test_split(df, test_size=0.2)
    print(f"\nTrain: {len(train)} days ({train.index[0]} to {train.index[-1]})")
    print(f"Test: {len(test)} days ({test.index[0]} to {test.index[-1]})")

