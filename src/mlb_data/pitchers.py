"""
Pitcher data collection functions.

Provides functions to fetch pitcher statistics with accurate team identification.
Uses Statcast data by default for 100% accurate team abbreviations.
"""

from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import pybaseball as pb
from pybaseball import cache

from .teams import ALL_TEAMS
from .utils import parse_date, chunk_date_range

# Enable caching
cache.enable()


def get_pitcher_game_logs(
    start_date: str,
    end_date: str | None = None,
    pitcher_id: int | None = None,
    team: str | None = None,
    starters_only: bool = True,
) -> pd.DataFrame:
    """
    Fetch pitcher game-by-game statistics using Statcast.

    This is the recommended method - it provides accurate team abbreviations
    (no city name ambiguity for NY, Chicago, LA).

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to yesterday
        pitcher_id: Optional MLBAM player ID to filter to one pitcher
        team: Optional team abbreviation to filter
        starters_only: If True, return only starting pitchers

    Returns:
        DataFrame with columns:
        - pitcher_id: MLBAM player ID
        - Name: Player name
        - game_date: Date of game
        - team_abbrev: Pitcher's team (e.g., "NYY", "NYM")
        - opp_abbrev: Opponent team
        - is_home: 1 if home game
        - SO: Strikeouts
        - pitches: Pitch count
        - H, BB, HR, etc.
    """
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"Fetching pitcher game logs from {start_date} to {end_date}...")

    # Use statcast for accurate team IDs
    if pitcher_id is not None:
        df = pb.statcast_pitcher(start_date, end_date, pitcher_id)
    else:
        # Fetch in chunks to avoid timeouts on large ranges
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

    # Aggregate to game level
    game_logs = _aggregate_to_game_logs(df, starters_only)

    # Filter by team if specified
    if team is not None:
        team = team.upper()
        if team not in ALL_TEAMS:
            raise ValueError(f"Unknown team: {team}. Use standard abbreviation.")
        game_logs = game_logs[game_logs['team_abbrev'] == team]

    print(f"  Found {len(game_logs)} pitcher game logs")
    return game_logs


def _aggregate_to_game_logs(
    statcast_df: pd.DataFrame,
    starters_only: bool = True,
) -> pd.DataFrame:
    """Aggregate pitch-level statcast data to game-level stats."""
    df = statcast_df.copy()

    # Filter to pitchers only
    df = df[df['pitcher'].notna()]

    # Determine pitcher's team
    df['is_home_pitcher'] = df['inning_topbot'] == 'Top'
    df['team_abbrev'] = np.where(df['is_home_pitcher'], df['home_team'], df['away_team'])
    df['opp_abbrev'] = np.where(df['is_home_pitcher'], df['away_team'], df['home_team'])

    # Group by game and pitcher
    group_cols = ['game_date', 'game_pk', 'pitcher', 'player_name',
                  'team_abbrev', 'opp_abbrev', 'home_team', 'away_team']

    # Aggregate stats
    game_logs = df.groupby(group_cols, as_index=False).agg(
        pitches=('release_speed', 'count'),
        SO=('events', lambda x: x.isin(['strikeout', 'strikeout_double_play']).sum()),
        BB=('events', lambda x: x.isin(['walk']).sum()),
        H=('events', lambda x: x.isin(['single', 'double', 'triple', 'home_run']).sum()),
        HR=('events', lambda x: x.isin(['home_run']).sum()),
        HBP=('events', lambda x: x.isin(['hit_by_pitch']).sum()),
        swinging_strikes=('description', lambda x: x.isin([
            'swinging_strike', 'swinging_strike_blocked'
        ]).sum()),
        called_strikes=('description', lambda x: x.isin(['called_strike']).sum()),
    )

    # Rename columns
    game_logs = game_logs.rename(columns={
        'pitcher': 'pitcher_id',
        'player_name': 'Name',
    })

    # Filter to starters if requested
    if starters_only:
        game_logs['pitch_rank'] = game_logs.groupby(
            ['game_date', 'team_abbrev']
        )['pitches'].rank(ascending=False, method='first')
        game_logs = game_logs[game_logs['pitch_rank'] == 1]
        game_logs = game_logs.drop(columns=['pitch_rank'])

    # Add is_home indicator
    game_logs['is_home'] = (game_logs['team_abbrev'] == game_logs['home_team']).astype(int)

    # Calculate rates
    game_logs['swstr_pct'] = game_logs['swinging_strikes'] / game_logs['pitches']

    return game_logs.reset_index(drop=True)


def get_pitcher_season_stats(
    season: int,
    qual: int = 0,
) -> pd.DataFrame:
    """
    Fetch pitcher season statistics from FanGraphs.

    Args:
        season: MLB season year
        qual: Minimum IP qualifier (0 = all pitchers)

    Returns:
        DataFrame with season pitching statistics
    """
    print(f"Fetching pitcher season stats for {season}...")
    df = pb.pitching_stats(season, qual=qual)

    # Select useful columns
    cols = [
        'Name', 'Team', 'W', 'L', 'ERA', 'G', 'GS', 'IP', 'TBF',
        'H', 'R', 'ER', 'HR', 'BB', 'SO',
        'K/9', 'BB/9', 'K/BB', 'H/9', 'HR/9',
        'AVG', 'WHIP', 'BABIP', 'LOB%', 'FIP', 'xFIP', 'SIERA',
        'GB%', 'FB%', 'LD%', 'HR/FB',
        'K%', 'BB%', 'K-BB%',
        'O-Swing%', 'Z-Swing%', 'Swing%',
        'O-Contact%', 'Z-Contact%', 'Contact%',
        'Zone%', 'F-Strike%', 'SwStr%',
        'Soft%', 'Med%', 'Hard%',
        'FBv', 'SL%', 'SLv', 'CB%', 'CBv', 'CH%', 'CHv',
        'wFB', 'wSL', 'wCB', 'wCH',
        'wFB/C', 'wSL/C', 'wCB/C', 'wCH/C',
        'CStr%', 'CSW%', 'xERA',
        'EV', 'Barrel%', 'HardHit%',
    ]

    available = [c for c in cols if c in df.columns]
    return df[available].copy()


def get_pitcher_statcast(
    start_date: str,
    end_date: str | None = None,
    pitcher_id: int | None = None,
) -> pd.DataFrame:
    """
    Fetch raw pitch-level Statcast data for pitchers.

    Use this when you need pitch-level detail (pitch types, velocities, etc.)

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        pitcher_id: Optional MLBAM player ID

    Returns:
        DataFrame with pitch-level statcast data
    """
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"Fetching statcast pitch data from {start_date} to {end_date}...")

    if pitcher_id is not None:
        return pb.statcast_pitcher(start_date, end_date, pitcher_id)
    else:
        return pb.statcast(start_dt=start_date, end_dt=end_date)
