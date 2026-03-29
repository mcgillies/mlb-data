"""
Batter data collection functions.

Provides functions to fetch batter statistics with accurate team identification.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pybaseball as pb
from pybaseball import cache

from .teams import ALL_TEAMS
from .utils import chunk_date_range

cache.enable()


def get_batter_game_logs(
    start_date: str,
    end_date: str | None = None,
    batter_id: int | None = None,
    team: str | None = None,
) -> pd.DataFrame:
    """
    Fetch batter game-by-game statistics using Statcast.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to yesterday
        batter_id: Optional MLBAM player ID to filter
        team: Optional team abbreviation to filter

    Returns:
        DataFrame with game-level batting stats
    """
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"Fetching batter game logs from {start_date} to {end_date}...")

    if batter_id is not None:
        df = pb.statcast_batter(start_date, end_date, batter_id)
    else:
        dfs = []
        for chunk_start, chunk_end in chunk_date_range(start_date, end_date, days=30):
            print(f"  Fetching {chunk_start} to {chunk_end}...")
            chunk_df = pb.statcast(start_dt=chunk_start, end_dt=chunk_end)
            if not chunk_df.empty:
                dfs.append(chunk_df)

        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)

    if df.empty:
        return pd.DataFrame()

    game_logs = _aggregate_batter_game_logs(df)

    if team is not None:
        team = team.upper()
        game_logs = game_logs[game_logs['team_abbrev'] == team]

    print(f"  Found {len(game_logs)} batter game logs")
    return game_logs


def _aggregate_batter_game_logs(statcast_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-level data to batter game logs."""
    df = statcast_df.copy()

    df = df[df['batter'].notna()]

    # Determine batter's team
    df['is_home_batter'] = df['inning_topbot'] == 'Bot'
    df['team_abbrev'] = np.where(df['is_home_batter'], df['home_team'], df['away_team'])
    df['opp_abbrev'] = np.where(df['is_home_batter'], df['away_team'], df['home_team'])

    group_cols = ['game_date', 'game_pk', 'batter', 'team_abbrev', 'opp_abbrev']

    game_logs = df.groupby(group_cols, as_index=False).agg(
        PA=('events', lambda x: x.notna().sum()),
        AB=('events', lambda x: (~x.isin(['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt', 'catcher_interf']) & x.notna()).sum()),
        H=('events', lambda x: x.isin(['single', 'double', 'triple', 'home_run']).sum()),
        singles=('events', lambda x: (x == 'single').sum()),
        doubles=('events', lambda x: (x == 'double').sum()),
        triples=('events', lambda x: (x == 'triple').sum()),
        HR=('events', lambda x: (x == 'home_run').sum()),
        RBI=('events', 'count'),  # Placeholder - RBI not in statcast
        BB=('events', lambda x: (x == 'walk').sum()),
        SO=('events', lambda x: x.isin(['strikeout', 'strikeout_double_play']).sum()),
        HBP=('events', lambda x: (x == 'hit_by_pitch').sum()),
    )

    game_logs = game_logs.rename(columns={'batter': 'batter_id'})

    return game_logs.reset_index(drop=True)


def get_batter_season_stats(
    season: int,
    qual: int = 0,
) -> pd.DataFrame:
    """
    Fetch batter season statistics from FanGraphs.

    Args:
        season: MLB season year
        qual: Minimum PA qualifier (0 = all batters)

    Returns:
        DataFrame with season batting statistics
    """
    print(f"Fetching batter season stats for {season}...")
    df = pb.batting_stats(season, qual=qual)

    cols = [
        'Name', 'Team', 'G', 'PA', 'AB', 'H', '1B', '2B', '3B', 'HR',
        'R', 'RBI', 'BB', 'SO', 'HBP', 'SB', 'CS',
        'AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP',
        'wOBA', 'wRC+', 'WAR',
        'K%', 'BB%', 'BB/K',
        'O-Swing%', 'Z-Swing%', 'Swing%',
        'O-Contact%', 'Z-Contact%', 'Contact%',
        'Zone%', 'SwStr%',
        'GB%', 'FB%', 'LD%',
        'Pull%', 'Cent%', 'Oppo%',
        'Soft%', 'Med%', 'Hard%',
        'EV', 'LA', 'Barrel%', 'HardHit%',
    ]

    available = [c for c in cols if c in df.columns]
    return df[available].copy()


def get_batter_statcast(
    start_date: str,
    end_date: str | None = None,
    batter_id: int | None = None,
) -> pd.DataFrame:
    """
    Fetch raw pitch-level Statcast data for batters.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        batter_id: Optional MLBAM player ID

    Returns:
        DataFrame with pitch-level statcast data
    """
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"Fetching statcast data from {start_date} to {end_date}...")

    if batter_id is not None:
        return pb.statcast_batter(start_date, end_date, batter_id)
    else:
        return pb.statcast(start_dt=start_date, end_dt=end_date)
