"""
Team-level statistics collection.

Provides team batting and pitching aggregates.
"""

import pandas as pd
import pybaseball as pb
from pybaseball import cache

from .teams import ALL_TEAMS

cache.enable()


def get_team_batting(
    season: int,
    teams: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch team batting statistics for a season.

    Args:
        season: MLB season year
        teams: Optional list of team abbreviations to filter

    Returns:
        DataFrame with team batting stats, indexed by team abbreviation
    """
    print(f"Fetching team batting stats for {season}...")
    df = pb.team_batting(start_season=season, end_season=season)

    # Select relevant columns
    cols = [
        'Team', 'G', 'PA', 'AB', 'H', '1B', '2B', '3B', 'HR',
        'R', 'RBI', 'BB', 'SO', 'HBP',
        'AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP',
        'K%', 'BB%', 'BB/K',
        'wOBA', 'wRC+',
        'O-Swing%', 'Z-Swing%', 'Swing%',
        'O-Contact%', 'Z-Contact%', 'Contact%',
        'Zone%', 'F-Strike%', 'SwStr%',
        'GB%', 'FB%', 'LD%',
        'Soft%', 'Med%', 'Hard%',
        'EV', 'Barrel%', 'HardHit%',
        'CStr%', 'CSW%',
    ]

    available = [c for c in cols if c in df.columns]
    df = df[available].copy()

    # Validate team abbreviations
    unknown = set(df['Team']) - set(ALL_TEAMS)
    if unknown:
        print(f"  Warning: Unknown team abbreviations: {unknown}")

    # Filter to specific teams if requested
    if teams is not None:
        teams = [t.upper() for t in teams]
        df = df[df['Team'].isin(teams)]

    return df


def get_team_pitching(
    season: int,
    teams: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch team pitching statistics for a season.

    Args:
        season: MLB season year
        teams: Optional list of team abbreviations to filter

    Returns:
        DataFrame with team pitching stats
    """
    print(f"Fetching team pitching stats for {season}...")
    df = pb.team_pitching(start_season=season, end_season=season)

    cols = [
        'Team', 'G', 'W', 'L', 'ERA', 'IP', 'TBF',
        'H', 'R', 'ER', 'HR', 'BB', 'SO',
        'K/9', 'BB/9', 'K/BB', 'H/9', 'HR/9',
        'AVG', 'WHIP', 'BABIP', 'LOB%', 'FIP', 'xFIP',
        'K%', 'BB%', 'K-BB%',
        'GB%', 'FB%', 'LD%',
        'Soft%', 'Med%', 'Hard%',
        'SwStr%', 'F-Strike%',
    ]

    available = [c for c in cols if c in df.columns]
    df = df[available].copy()

    if teams is not None:
        teams = [t.upper() for t in teams]
        df = df[df['Team'].isin(teams)]

    return df


def get_team_game_logs(
    season: int,
    team: str,
    log_type: str = 'batting',
) -> pd.DataFrame:
    """
    Fetch game-by-game team statistics.

    Args:
        season: MLB season year
        team: Team abbreviation
        log_type: 'batting' or 'pitching'

    Returns:
        DataFrame with game-level team stats
    """
    team = team.upper()
    if team not in ALL_TEAMS:
        raise ValueError(f"Unknown team: {team}")

    print(f"Fetching {team} {log_type} game logs for {season}...")
    return pb.team_game_logs(season, team, log_type)
