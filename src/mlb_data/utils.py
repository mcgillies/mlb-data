"""
Utility functions for date handling, caching, and data processing.
"""

from datetime import datetime, timedelta
from typing import Generator, Tuple

import pandas as pd


def parse_date(date_val) -> datetime:
    """
    Parse date from various formats.

    Args:
        date_val: Date string, datetime, or pandas Timestamp

    Returns:
        datetime object
    """
    if isinstance(date_val, datetime):
        return date_val

    if isinstance(date_val, pd.Timestamp):
        return date_val.to_pydatetime()

    date_str = str(date_val)

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%b %d, %Y",
        "%m/%d/%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.split()[0], fmt)
        except ValueError:
            continue

    # Try pandas as fallback
    try:
        return pd.to_datetime(date_val).to_pydatetime()
    except Exception:
        pass

    raise ValueError(f"Could not parse date: {date_val}")


def format_date(dt: datetime) -> str:
    """Format datetime as YYYY-MM-DD string."""
    return dt.strftime("%Y-%m-%d")


def chunk_date_range(
    start_date: str,
    end_date: str,
    days: int = 30,
) -> Generator[Tuple[str, str], None, None]:
    """
    Split a date range into smaller chunks.

    Useful for avoiding API timeouts on large date ranges.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        days: Maximum days per chunk

    Yields:
        Tuples of (chunk_start, chunk_end) date strings
    """
    start = parse_date(start_date)
    end = parse_date(end_date)

    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=days - 1), end)
        yield format_date(current), format_date(chunk_end)
        current = chunk_end + timedelta(days=1)


def season_date_range(season: int) -> Tuple[str, str]:
    """
    Get approximate date range for an MLB season.

    Args:
        season: MLB season year

    Returns:
        Tuple of (start_date, end_date)
    """
    # Approximate season dates
    start = f"{season}-03-28"  # Opening day is usually late March
    end = f"{season}-10-01"    # Regular season ends late September

    return start, end


def compute_rolling(
    df: pd.DataFrame,
    group_col: str,
    stat_cols: list[str],
    windows: list[int] = [3, 5, 10],
    date_col: str = 'game_date',
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Compute rolling averages for stats, avoiding data leakage.

    Uses shift(1) to exclude current game from the calculation.

    Args:
        df: DataFrame with game-level data
        group_col: Column to group by (e.g., 'pitcher_id')
        stat_cols: Columns to compute rolling stats for
        windows: Window sizes
        date_col: Date column for sorting
        min_periods: Minimum periods for valid calculation

    Returns:
        DataFrame with rolling stat columns added
    """
    df = df.copy()

    # Parse and sort by date
    df['_date_parsed'] = df[date_col].apply(parse_date)
    df = df.sort_values([group_col, '_date_parsed'])

    available = [c for c in stat_cols if c in df.columns]

    for stat in available:
        for window in windows:
            col_name = f"{stat}_roll{window}"
            df[col_name] = (
                df.groupby(group_col)[stat]
                .shift(1)
                .rolling(window=window, min_periods=min_periods)
                .mean()
                .reset_index(level=0, drop=True)
            )

    df = df.drop(columns=['_date_parsed'])
    return df


def compute_rest_days(
    df: pd.DataFrame,
    group_col: str,
    date_col: str = 'game_date',
    default: int = 5,
    max_days: int = 30,
) -> pd.DataFrame:
    """
    Compute days of rest since last appearance.

    Args:
        df: DataFrame with game data
        group_col: Column to group by (e.g., 'pitcher_id')
        date_col: Date column
        default: Default rest days for first appearance
        max_days: Maximum rest days (cap extreme values)

    Returns:
        DataFrame with 'rest_days' column added
    """
    df = df.copy()

    df['_date_parsed'] = df[date_col].apply(parse_date)
    df = df.sort_values([group_col, '_date_parsed'])

    df['_prev_date'] = df.groupby(group_col)['_date_parsed'].shift(1)
    df['rest_days'] = (df['_date_parsed'] - df['_prev_date']).dt.days
    df['rest_days'] = df['rest_days'].fillna(default).clip(upper=max_days)

    df = df.drop(columns=['_date_parsed', '_prev_date'])
    return df
