"""
Player lookup and identification functions.
"""

import pandas as pd
import pybaseball as pb
from pybaseball import cache

cache.enable()


def lookup_player(
    last_name: str,
    first_name: str | None = None,
) -> pd.DataFrame:
    """
    Look up player ID by name.

    Args:
        last_name: Player's last name
        first_name: Player's first name (optional but recommended)

    Returns:
        DataFrame with matching players and their IDs
    """
    if first_name:
        print(f"Looking up {first_name} {last_name}...")
        return pb.playerid_lookup(last_name, first_name)
    else:
        print(f"Looking up {last_name}...")
        return pb.playerid_lookup(last_name)


def get_player_id(
    last_name: str,
    first_name: str | None = None,
    id_type: str = 'mlbam',
) -> int | None:
    """
    Get a player's ID.

    Args:
        last_name: Player's last name
        first_name: Player's first name
        id_type: Type of ID to return ('mlbam', 'retro', 'bbref', 'fangraphs')

    Returns:
        Player ID, or None if not found
    """
    results = lookup_player(last_name, first_name)

    if results.empty:
        return None

    id_col_map = {
        'mlbam': 'key_mlbam',
        'retro': 'key_retro',
        'bbref': 'key_bbref',
        'fangraphs': 'key_fangraphs',
    }

    col = id_col_map.get(id_type.lower())
    if col is None:
        raise ValueError(f"Unknown ID type: {id_type}")

    if col not in results.columns:
        return None

    # Return first match
    val = results[col].iloc[0]
    if pd.isna(val):
        return None

    return int(val) if id_type == 'mlbam' else val


def search_players(
    query: str,
) -> pd.DataFrame:
    """
    Search for players by partial name.

    Args:
        query: Search string (partial name)

    Returns:
        DataFrame with matching players
    """
    # Split query into potential first/last name
    parts = query.strip().split()

    if len(parts) >= 2:
        # Try first last format
        first = parts[0]
        last = ' '.join(parts[1:])
        results = lookup_player(last, first)
        if not results.empty:
            return results

        # Try last first format
        last = parts[0]
        first = ' '.join(parts[1:])
        results = lookup_player(last, first)
        if not results.empty:
            return results

    # Try as last name only
    return lookup_player(parts[0] if parts else query)
