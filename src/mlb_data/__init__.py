"""
MLB Data - A wrapper around pybaseball for consistent baseball data collection.

Provides:
- Accurate team identification (no city name ambiguity)
- Standardized data formats
- Smart caching and rate limiting
- Pre-built aggregation functions
"""

from .teams import (
    ALL_TEAMS,
    TEAM_INFO,
    get_team_name,
    get_team_abbrev,
    list_teams,
)

from .pitchers import (
    get_pitcher_game_logs,
    get_pitcher_season_stats,
    get_pitcher_statcast,
)

from .batters import (
    get_batter_game_logs,
    get_batter_season_stats,
    get_batter_statcast,
)

from .team_stats import (
    get_team_batting,
    get_team_pitching,
)

from .games import (
    get_schedule,
    get_game_info,
)

from .players import (
    lookup_player,
    get_player_id,
)

from .pitch_stats import (
    get_statcast_pitches,
    get_pitcher_arsenal,
    get_pitcher_pitch_type_stats,
    get_batter_pitch_stats,
    get_batter_pitch_type_stats,
    get_plate_appearances,
    get_pitcher_profiles,
    get_batter_profiles,
)

__version__ = "0.1.0"
