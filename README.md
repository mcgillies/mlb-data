# MLB Data

A wrapper around [pybaseball](https://github.com/jldbc/pybaseball) for consistent MLB data collection.

## Features

- **Accurate team identification**: No more ambiguous city names. All functions return proper team abbreviations (NYY vs NYM, CHC vs CHW, LAD vs LAA).
- **Standardized formats**: Consistent column names and data types across all functions.
- **Smart aggregation**: Convert pitch-level Statcast data to game-level stats automatically.
- **Built-in utilities**: Rolling averages, rest days calculation, date range chunking.

## Installation

```bash
# From the mlb-data directory
pip install -e .

# Or install directly
pip install -e /Users/matthewgillies/mlb-data
```

## Quick Start

```python
from mlb_data import (
    get_pitcher_game_logs,
    get_team_batting,
    get_player_id,
    list_teams,
)

# List all team abbreviations
list_teams()

# Get pitcher game logs with accurate team IDs
logs = get_pitcher_game_logs("2024-04-01", "2024-06-01")
# Returns: pitcher_id, Name, team_abbrev, opp_abbrev, SO, pitches, etc.

# Get team batting stats
batting = get_team_batting(2024)

# Look up a player ID
cole_id = get_player_id("Cole", "Gerrit")
```

## Why Use This?

### The Problem

pybaseball's `pitching_stats_range()` returns city names for teams:

```
Tm: "New York"     # Yankees or Mets?
Opp: "Chicago"     # Cubs or White Sox?
```

For cities with two teams, this is ambiguous.

### The Solution

This library uses Statcast data which provides unambiguous team abbreviations:

```
team_abbrev: "NYY"    # Definitely Yankees
opp_abbrev: "CHC"     # Definitely Cubs
```

## API Reference

### Pitcher Data

```python
# Game logs (uses Statcast - 100% accurate teams)
get_pitcher_game_logs(start_date, end_date, pitcher_id=None, team=None, starters_only=True)

# Season stats (from FanGraphs)
get_pitcher_season_stats(season, qual=0)

# Raw pitch-level data
get_pitcher_statcast(start_date, end_date, pitcher_id=None)
```

### Batter Data

```python
get_batter_game_logs(start_date, end_date, batter_id=None, team=None)
get_batter_season_stats(season, qual=0)
get_batter_statcast(start_date, end_date, batter_id=None)
```

### Team Data

```python
get_team_batting(season, teams=None)
get_team_pitching(season, teams=None)
get_team_game_logs(season, team, log_type='batting')
```

### Schedule & Games

```python
get_schedule(season, team=None)
get_standings(season)
```

### Player Lookup

```python
lookup_player(last_name, first_name=None)
get_player_id(last_name, first_name=None, id_type='mlbam')
search_players(query)
```

### Team Utilities

```python
from mlb_data import get_team_abbrev, get_team_name, ALL_TEAMS

get_team_abbrev("Boston")           # "BOS"
get_team_abbrev("Yankees")          # "NYY"
get_team_abbrev("New York", "AL")   # "NYY"
get_team_abbrev("New York", "NL")   # "NYM"

get_team_name("NYY")                # "New York Yankees"
```

### Utility Functions

```python
from mlb_data.utils import compute_rolling, compute_rest_days, chunk_date_range

# Add rolling averages to a DataFrame
df = compute_rolling(df, group_col='pitcher_id', stat_cols=['SO', 'IP'], windows=[3, 5])

# Add rest days
df = compute_rest_days(df, group_col='pitcher_id')

# Chunk large date ranges for API calls
for start, end in chunk_date_range("2024-01-01", "2024-12-31", days=30):
    data = fetch_data(start, end)
```

## Team Abbreviations

| Abbrev | Team |
|--------|------|
| ARI | Arizona Diamondbacks |
| ATL | Atlanta Braves |
| BAL | Baltimore Orioles |
| BOS | Boston Red Sox |
| CHC | Chicago Cubs |
| CHW | Chicago White Sox |
| CIN | Cincinnati Reds |
| CLE | Cleveland Guardians |
| COL | Colorado Rockies |
| DET | Detroit Tigers |
| HOU | Houston Astros |
| KCR | Kansas City Royals |
| LAA | Los Angeles Angels |
| LAD | Los Angeles Dodgers |
| MIA | Miami Marlins |
| MIL | Milwaukee Brewers |
| MIN | Minnesota Twins |
| NYM | New York Mets |
| NYY | New York Yankees |
| OAK | Oakland Athletics |
| PHI | Philadelphia Phillies |
| PIT | Pittsburgh Pirates |
| SDP | San Diego Padres |
| SEA | Seattle Mariners |
| SFG | San Francisco Giants |
| STL | St. Louis Cardinals |
| TBR | Tampa Bay Rays |
| TEX | Texas Rangers |
| TOR | Toronto Blue Jays |
| WSN | Washington Nationals |

## Data Sources

- **Statcast** (Baseball Savant): Pitch-level data, accurate team IDs
- **FanGraphs**: Season statistics, advanced metrics
- **Baseball Reference**: Game logs, historical data

All data is accessed via [pybaseball](https://github.com/jldbc/pybaseball).
