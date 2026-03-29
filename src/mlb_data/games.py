"""
Game and schedule data collection.

Provides schedule, game information, and boxscore functions.
"""

import pandas as pd
import pybaseball as pb
from pybaseball import cache

from .teams import ALL_TEAMS

cache.enable()


def get_schedule(
    season: int,
    team: str | None = None,
) -> pd.DataFrame:
    """
    Fetch game schedule and results.

    Args:
        season: MLB season year
        team: Optional team abbreviation to filter

    Returns:
        DataFrame with game schedule and results
    """
    if team is not None:
        team = team.upper()
        if team not in ALL_TEAMS:
            raise ValueError(f"Unknown team: {team}")
        print(f"Fetching {team} schedule for {season}...")
        return pb.schedule_and_record(season, team)

    # Fetch for all teams and combine
    print(f"Fetching full {season} schedule (this may take a moment)...")
    dfs = []
    for t in ALL_TEAMS:
        try:
            df = pb.schedule_and_record(season, t)
            df['team'] = t
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Failed to fetch {t}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def get_game_info(
    game_pk: int,
) -> dict:
    """
    Fetch detailed information for a specific game.

    Args:
        game_pk: MLB game ID (from statcast data)

    Returns:
        Dictionary with game information
    """
    # Note: pybaseball doesn't have direct game info lookup
    # You'd need to use MLB Stats API directly for this
    raise NotImplementedError(
        "Direct game info lookup not yet implemented. "
        "Use statcast data filtered by game_pk instead."
    )


def get_standings(
    season: int,
    date: str | None = None,
) -> pd.DataFrame:
    """
    Fetch standings for a season.

    Args:
        season: MLB season year
        date: Optional date for historical standings (YYYY-MM-DD)

    Returns:
        DataFrame with standings
    """
    print(f"Fetching standings for {season}...")

    # Get all team records
    dfs = []
    for team in ALL_TEAMS:
        try:
            schedule = pb.schedule_and_record(season, team)
            if not schedule.empty:
                # Get final record
                wins = (schedule['W/L'] == 'W').sum()
                losses = (schedule['W/L'] == 'L').sum()
                dfs.append({
                    'Team': team,
                    'W': wins,
                    'L': losses,
                    'Pct': wins / (wins + losses) if (wins + losses) > 0 else 0,
                })
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    standings = pd.DataFrame(dfs)
    standings = standings.sort_values('Pct', ascending=False)

    return standings.reset_index(drop=True)
