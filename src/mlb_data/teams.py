"""
Team identification and mapping utilities.

Provides consistent team abbreviations and handles the ambiguous
city names (New York, Chicago, Los Angeles) that have two teams.
"""

from typing import Tuple

# All team abbreviations with full details: (city, name, league)
TEAM_INFO = {
    'ARI': ('Arizona', 'Diamondbacks', 'NL'),
    'ATL': ('Atlanta', 'Braves', 'NL'),
    'BAL': ('Baltimore', 'Orioles', 'AL'),
    'BOS': ('Boston', 'Red Sox', 'AL'),
    'CHC': ('Chicago', 'Cubs', 'NL'),
    'CHW': ('Chicago', 'White Sox', 'AL'),
    'CIN': ('Cincinnati', 'Reds', 'NL'),
    'CLE': ('Cleveland', 'Guardians', 'AL'),
    'COL': ('Colorado', 'Rockies', 'NL'),
    'DET': ('Detroit', 'Tigers', 'AL'),
    'HOU': ('Houston', 'Astros', 'AL'),
    'KCR': ('Kansas City', 'Royals', 'AL'),
    'LAA': ('Los Angeles', 'Angels', 'AL'),
    'LAD': ('Los Angeles', 'Dodgers', 'NL'),
    'MIA': ('Miami', 'Marlins', 'NL'),
    'MIL': ('Milwaukee', 'Brewers', 'NL'),
    'MIN': ('Minnesota', 'Twins', 'AL'),
    'NYM': ('New York', 'Mets', 'NL'),
    'NYY': ('New York', 'Yankees', 'AL'),
    'OAK': ('Oakland', 'Athletics', 'AL'),
    'PHI': ('Philadelphia', 'Phillies', 'NL'),
    'PIT': ('Pittsburgh', 'Pirates', 'NL'),
    'SDP': ('San Diego', 'Padres', 'NL'),
    'SEA': ('Seattle', 'Mariners', 'AL'),
    'SFG': ('San Francisco', 'Giants', 'NL'),
    'STL': ('St. Louis', 'Cardinals', 'NL'),
    'TBR': ('Tampa Bay', 'Rays', 'AL'),
    'TEX': ('Texas', 'Rangers', 'AL'),
    'TOR': ('Toronto', 'Blue Jays', 'AL'),
    'WSN': ('Washington', 'Nationals', 'NL'),
}

ALL_TEAMS = list(TEAM_INFO.keys())

# Single-team cities: city name -> abbreviation
CITY_TO_TEAM = {
    'Arizona': 'ARI',
    'Atlanta': 'ATL',
    'Baltimore': 'BAL',
    'Boston': 'BOS',
    'Cincinnati': 'CIN',
    'Cleveland': 'CLE',
    'Colorado': 'COL',
    'Detroit': 'DET',
    'Houston': 'HOU',
    'Kansas City': 'KCR',
    'Miami': 'MIA',
    'Milwaukee': 'MIL',
    'Minnesota': 'MIN',
    'Oakland': 'OAK',
    'Philadelphia': 'PHI',
    'Pittsburgh': 'PIT',
    'San Diego': 'SDP',
    'Seattle': 'SEA',
    'San Francisco': 'SFG',
    'St. Louis': 'STL',
    'Tampa Bay': 'TBR',
    'Texas': 'TEX',
    'Toronto': 'TOR',
    'Washington': 'WSN',
}

# Cities with two teams: city -> (AL_team, NL_team)
SHARED_CITIES = {
    'New York': ('NYY', 'NYM'),
    'Chicago': ('CHW', 'CHC'),
    'Los Angeles': ('LAA', 'LAD'),
}

# Alternative names/abbreviations that map to standard abbreviations
TEAM_ALIASES = {
    # Common alternatives
    'KC': 'KCR',
    'SD': 'SDP',
    'SF': 'SFG',
    'TB': 'TBR',
    'WAS': 'WSN',
    'WSH': 'WSN',
    'CWS': 'CHW',
    'CHI': None,  # Ambiguous - need league
    'LA': None,   # Ambiguous - need league
    'NY': None,   # Ambiguous - need league
    # Full names
    'Diamondbacks': 'ARI',
    'D-backs': 'ARI',
    'Braves': 'ATL',
    'Orioles': 'BAL',
    'Red Sox': 'BOS',
    'Cubs': 'CHC',
    'White Sox': 'CHW',
    'Reds': 'CIN',
    'Guardians': 'CLE',
    'Indians': 'CLE',  # Former name
    'Rockies': 'COL',
    'Tigers': 'DET',
    'Astros': 'HOU',
    'Royals': 'KCR',
    'Angels': 'LAA',
    'Dodgers': 'LAD',
    'Marlins': 'MIA',
    'Brewers': 'MIL',
    'Twins': 'MIN',
    'Mets': 'NYM',
    'Yankees': 'NYY',
    'Athletics': 'OAK',
    "A's": 'OAK',
    'Phillies': 'PHI',
    'Pirates': 'PIT',
    'Padres': 'SDP',
    'Mariners': 'SEA',
    'Giants': 'SFG',
    'Cardinals': 'STL',
    'Rays': 'TBR',
    'Rangers': 'TEX',
    'Blue Jays': 'TOR',
    'Nationals': 'WSN',
}


def get_team_name(abbrev: str) -> str:
    """
    Get full team name from abbreviation.

    Args:
        abbrev: Team abbreviation (e.g., "NYY")

    Returns:
        Full team name (e.g., "New York Yankees")
    """
    abbrev = abbrev.upper()
    if abbrev in TEAM_INFO:
        city, name, _ = TEAM_INFO[abbrev]
        return f"{city} {name}"
    return abbrev


def get_team_abbrev(
    identifier: str,
    league: str | None = None,
) -> str | None:
    """
    Convert any team identifier to standard abbreviation.

    Handles city names, team names, and alternative abbreviations.

    Args:
        identifier: Team identifier (city, name, or abbreviation)
        league: League hint ("AL" or "NL") for ambiguous cities

    Returns:
        Standard 3-letter abbreviation, or None if ambiguous/unknown

    Examples:
        >>> get_team_abbrev("Boston")
        'BOS'
        >>> get_team_abbrev("Yankees")
        'NYY'
        >>> get_team_abbrev("New York", league="AL")
        'NYY'
        >>> get_team_abbrev("Chicago")  # Ambiguous without league
        None
    """
    identifier = identifier.strip()

    # Already a valid abbreviation
    if identifier.upper() in ALL_TEAMS:
        return identifier.upper()

    # Check aliases (team names, alternative abbrevs)
    if identifier in TEAM_ALIASES:
        result = TEAM_ALIASES[identifier]
        if result is not None:
            return result
        # Ambiguous alias - need league info
        # Fall through to city logic

    # Check single-team cities
    if identifier in CITY_TO_TEAM:
        return CITY_TO_TEAM[identifier]

    # Check shared cities
    if identifier in SHARED_CITIES:
        al_team, nl_team = SHARED_CITIES[identifier]
        if league is None:
            return None  # Ambiguous
        league = league.upper()
        if 'AL' in league:
            return al_team
        elif 'NL' in league:
            return nl_team
        return None

    # Check case-insensitive aliases
    for alias, abbrev in TEAM_ALIASES.items():
        if alias.lower() == identifier.lower() and abbrev is not None:
            return abbrev

    return None


def resolve_team_from_game(
    city: str,
    home_team: str | None = None,
    away_team: str | None = None,
    league: str | None = None,
) -> Tuple[str, bool]:
    """
    Resolve ambiguous city name using game context.

    When you have home_team/away_team from statcast, use those.
    Otherwise falls back to league heuristic.

    Args:
        city: City name to resolve
        home_team: Home team abbreviation (if known)
        away_team: Away team abbreviation (if known)
        league: League hint as fallback

    Returns:
        Tuple of (team_abbrev, is_certain)
    """
    # Try direct resolution first
    abbrev = get_team_abbrev(city, league=league)
    if abbrev is not None:
        # Single-team city - always certain
        is_shared = city in SHARED_CITIES
        return abbrev, not is_shared

    # For shared cities, try to match against known teams in game
    if city in SHARED_CITIES:
        al_team, nl_team = SHARED_CITIES[city]

        # Check if either team is in the game
        if home_team in (al_team, nl_team):
            return home_team, True
        if away_team in (al_team, nl_team):
            return away_team, True

        # Fall back to league heuristic
        if league:
            abbrev = get_team_abbrev(city, league=league)
            if abbrev:
                return abbrev, False  # Not certain (interleague possible)

        # Default to AL team, mark uncertain
        return al_team, False

    return f"UNKNOWN_{city}", False


def list_teams(league: str | None = None) -> None:
    """
    Print all team abbreviations and names.

    Args:
        league: Filter to "AL" or "NL" (optional)
    """
    print("\nMLB Team Abbreviations:")
    print("-" * 50)

    for abbrev in sorted(ALL_TEAMS):
        city, name, team_league = TEAM_INFO[abbrev]
        if league is None or team_league == league.upper():
            print(f"  {abbrev}: {city} {name} ({team_league})")

    print()


def get_division(abbrev: str) -> str | None:
    """Get division for a team."""
    divisions = {
        'AL East': ['BAL', 'BOS', 'NYY', 'TBR', 'TOR'],
        'AL Central': ['CHW', 'CLE', 'DET', 'KCR', 'MIN'],
        'AL West': ['HOU', 'LAA', 'OAK', 'SEA', 'TEX'],
        'NL East': ['ATL', 'MIA', 'NYM', 'PHI', 'WSN'],
        'NL Central': ['CHC', 'CIN', 'MIL', 'PIT', 'STL'],
        'NL West': ['ARI', 'COL', 'LAD', 'SDP', 'SFG'],
    }

    for division, teams in divisions.items():
        if abbrev.upper() in teams:
            return division
    return None
