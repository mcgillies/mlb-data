"""
Advanced pitch-level statistics from Statcast.

Computes pitcher arsenal profiles (velocity, movement, spin, usage) and
batter performance profiles (whiff rates, contact rates by pitch type/velocity).
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pybaseball as pb
from pybaseball import cache

from .utils import chunk_date_range

cache.enable()

# Pitch type groupings (kept for legacy/reference)
FASTBALLS = ['FF', 'SI', 'FC']  # 4-seam, sinker, cutter
BREAKING = ['SL', 'CU', 'KC', 'SV', 'ST']  # slider, curve, knuckle-curve, sweeper
OFFSPEED = ['CH', 'FS', 'SC', 'FO']  # changeup, splitter, screwball, forkball
OTHER = ['KN', 'EP', 'CS', 'PO', 'FA']  # knuckleball, eephus, etc.

PITCH_GROUPS = {
    'fastball': FASTBALLS,
    'breaking': BREAKING,
    'offspeed': OFFSPEED,
}

# All recognized pitch types for individual stats
ALL_PITCH_TYPES = [
    'FF',  # 4-seam fastball
    'SI',  # sinker
    'FC',  # cutter
    'SL',  # slider
    'CU',  # curveball
    'CH',  # changeup
    'SV',  # sweeper
    'FS',  # splitter
    'KC',  # knuckle-curve
    'ST',  # sweeping curve / slurve
    'KN',  # knuckleball
]

# Swing events for whiff calculation
SWING_EVENTS = [
    'swinging_strike', 'swinging_strike_blocked',
    'foul', 'foul_tip', 'foul_bunt',
    'hit_into_play', 'hit_into_play_score', 'hit_into_play_no_out',
]

WHIFF_EVENTS = ['swinging_strike', 'swinging_strike_blocked']


def get_statcast_pitches(
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Fetch raw Statcast pitch data for a date range.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to yesterday

    Returns:
        DataFrame with pitch-level Statcast data
    """
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"Fetching Statcast pitches from {start_date} to {end_date}...")

    dfs = []
    for chunk_start, chunk_end in chunk_date_range(start_date, end_date, days=14):
        print(f"  {chunk_start} to {chunk_end}...")
        chunk = pb.statcast(start_dt=chunk_start, end_dt=chunk_end)
        if not chunk.empty:
            dfs.append(chunk)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total pitches: {len(df):,}")
    return df


# =============================================================================
# PITCHER ARSENAL STATS
# =============================================================================

def get_pitcher_arsenal(
    start_date: str,
    end_date: str | None = None,
    min_pitches: int = 100,
    min_pitch_type_count: int = 20,
    pitches_df: pd.DataFrame | None = None,
    use_decay: bool = True,
    arsenal_half_life: int = 365,
    usage_half_life: int = 180,
    performance_half_life: int = 90,
) -> pd.DataFrame:
    """
    Compute pitcher arsenal statistics with individual pitch type breakdowns.

    Uses exponential decay weighting so recent data is weighted more heavily.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        min_pitches: Minimum total pitches to include pitcher
        min_pitch_type_count: Minimum pitches of a type to compute stats
        pitches_df: Optional pre-loaded Statcast DataFrame
        use_decay: Whether to use exponential decay weighting (default True)
        arsenal_half_life: Half-life in days for arsenal features (velo, spin, movement)
        usage_half_life: Half-life in days for usage percentages
        performance_half_life: Half-life in days for performance metrics (whiff%, etc)

    Returns:
        DataFrame with one row per pitcher containing:
        - Per pitch type (ff, si, fc, sl, cu, ch, sv, fs, etc.):
            velo_avg, spin_avg, vmov_avg, hmov_avg, usage_pct, whiff_pct
        - primary_pitch: The pitcher's most-used pitch type
        - Overall: whiff%, csw%, zone%, chase_pct_induced, contact_pct, f_strike_pct
    """
    if pitches_df is None:
        pitches_df = get_statcast_pitches(start_date, end_date)

    if pitches_df.empty:
        return pd.DataFrame()

    df = pitches_df.copy()

    # Filter to valid pitcher data
    df = df[df['pitcher'].notna() & df['pitch_type'].notna()]

    # Calculate days ago for exponential decay
    if use_decay:
        if end_date is None:
            reference_date = datetime.now()
        else:
            reference_date = datetime.strptime(end_date, '%Y-%m-%d')
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['days_ago'] = (reference_date - df['game_date']).dt.days.clip(lower=0)

        # Precompute weights for each half-life
        df['w_arsenal'] = np.exp(-df['days_ago'] / arsenal_half_life)
        df['w_usage'] = np.exp(-df['days_ago'] / usage_half_life)
        df['w_perf'] = np.exp(-df['days_ago'] / performance_half_life)

    print("Computing pitcher arsenal stats...")

    # Group by pitcher
    results = []

    for pitcher_id, pitcher_df in df.groupby('pitcher'):
        if len(pitcher_df) < min_pitches:
            continue

        row = {'pitcher_id': int(pitcher_id)}

        # Get pitcher name and handedness
        row['pitcher_name'] = pitcher_df['player_name'].iloc[0]
        row['p_throws'] = pitcher_df['p_throws'].iloc[0]
        row['total_pitches'] = len(pitcher_df)

        # === INDIVIDUAL PITCH TYPE STATS ===
        # Use recent-weighted counts for primary pitch determination
        if use_decay:
            pitch_type_weights = pitcher_df.groupby('pitch_type')['w_usage'].sum()
            primary_pitch = pitch_type_weights.idxmax() if len(pitch_type_weights) > 0 else None
        else:
            pitch_type_counts = pitcher_df['pitch_type'].value_counts()
            primary_pitch = pitch_type_counts.index[0] if len(pitch_type_counts) > 0 else None
        row['primary_pitch'] = primary_pitch

        for pitch_type in ALL_PITCH_TYPES:
            pt_lower = pitch_type.lower()
            pt_df = pitcher_df[pitcher_df['pitch_type'] == pitch_type]

            if len(pt_df) >= min_pitch_type_count:
                if use_decay:
                    # Usage percentage (weighted)
                    row[f'{pt_lower}_usage_pct'] = pt_df['w_usage'].sum() / pitcher_df['w_usage'].sum()

                    # Velocity and movement (weighted by arsenal half-life)
                    w_ars = pt_df['w_arsenal']
                    w_sum = w_ars.sum()
                    row[f'{pt_lower}_velo_avg'] = (pt_df['release_speed'] * w_ars).sum() / w_sum if w_sum > 0 else np.nan
                    row[f'{pt_lower}_spin_avg'] = (pt_df['release_spin_rate'] * w_ars).sum() / w_sum if w_sum > 0 else np.nan
                    row[f'{pt_lower}_vmov_avg'] = (pt_df['pfx_z'] * w_ars).sum() / w_sum if w_sum > 0 else np.nan
                    row[f'{pt_lower}_hmov_avg'] = (pt_df['pfx_x'] * w_ars).sum() / w_sum if w_sum > 0 else np.nan

                    # Whiff rate for this pitch type (weighted by performance half-life)
                    pt_swings = pt_df[pt_df['description'].isin(SWING_EVENTS)]
                    pt_whiffs = pt_df[pt_df['description'].isin(WHIFF_EVENTS)]
                    swing_w = pt_swings['w_perf'].sum()
                    whiff_w = pt_whiffs['w_perf'].sum()
                    row[f'{pt_lower}_whiff_pct'] = whiff_w / swing_w if swing_w > 0 else 0
                else:
                    row[f'{pt_lower}_usage_pct'] = len(pt_df) / len(pitcher_df)
                    row[f'{pt_lower}_velo_avg'] = pt_df['release_speed'].mean()
                    row[f'{pt_lower}_spin_avg'] = pt_df['release_spin_rate'].mean()
                    row[f'{pt_lower}_vmov_avg'] = pt_df['pfx_z'].mean()
                    row[f'{pt_lower}_hmov_avg'] = pt_df['pfx_x'].mean()
                    pt_swings = pt_df[pt_df['description'].isin(SWING_EVENTS)]
                    pt_whiffs = pt_df[pt_df['description'].isin(WHIFF_EVENTS)]
                    row[f'{pt_lower}_whiff_pct'] = len(pt_whiffs) / len(pt_swings) if len(pt_swings) > 0 else 0
            else:
                # Set usage to 0 if not thrown enough (or at all)
                if use_decay:
                    row[f'{pt_lower}_usage_pct'] = pt_df['w_usage'].sum() / pitcher_df['w_usage'].sum() if len(pt_df) > 0 else 0
                else:
                    row[f'{pt_lower}_usage_pct'] = len(pt_df) / len(pitcher_df) if len(pt_df) > 0 else 0

        # === OVERALL PITCH OUTCOMES ===
        swings = pitcher_df[pitcher_df['description'].isin(SWING_EVENTS)]
        whiffs = pitcher_df[pitcher_df['description'].isin(WHIFF_EVENTS)]
        called_strikes = pitcher_df[pitcher_df['description'] == 'called_strike']

        if use_decay:
            swing_w = swings['w_perf'].sum()
            whiff_w = whiffs['w_perf'].sum()
            cs_w = called_strikes['w_perf'].sum()
            total_w = pitcher_df['w_perf'].sum()

            row['whiff_pct'] = whiff_w / swing_w if swing_w > 0 else 0
            row['swstr_pct'] = whiff_w / total_w if total_w > 0 else 0
            row['csw_pct'] = (whiff_w + cs_w) / total_w if total_w > 0 else 0
        else:
            row['whiff_pct'] = len(whiffs) / len(swings) if len(swings) > 0 else 0
            row['swstr_pct'] = len(whiffs) / len(pitcher_df)
            row['csw_pct'] = (len(whiffs) + len(called_strikes)) / len(pitcher_df)

        # Zone rate
        in_zone = pitcher_df[pitcher_df['zone'].between(1, 9)]
        if use_decay:
            row['zone_pct'] = in_zone['w_perf'].sum() / pitcher_df['w_perf'].sum() if len(pitcher_df) > 0 else 0
        else:
            row['zone_pct'] = len(in_zone) / len(pitcher_df)

        # Chase rate induced (swings on pitches outside zone)
        out_zone = pitcher_df[~pitcher_df['zone'].between(1, 9)]
        out_zone_swings = out_zone[out_zone['description'].isin(SWING_EVENTS)]
        if use_decay:
            oz_w = out_zone['w_perf'].sum()
            ozs_w = out_zone_swings['w_perf'].sum()
            row['chase_pct_induced'] = ozs_w / oz_w if oz_w > 0 else 0
        else:
            row['chase_pct_induced'] = len(out_zone_swings) / len(out_zone) if len(out_zone) > 0 else 0

        # Contact rate
        contacts = swings[~swings['description'].isin(WHIFF_EVENTS)]
        if use_decay:
            contact_w = contacts['w_perf'].sum()
            row['contact_pct'] = contact_w / swing_w if swing_w > 0 else 0
        else:
            row['contact_pct'] = len(contacts) / len(swings) if len(swings) > 0 else 0

        # First pitch strike rate
        first_pitches = pitcher_df[(pitcher_df['balls'] == 0) & (pitcher_df['strikes'] == 0)]
        first_strikes = first_pitches[
            first_pitches['description'].isin(['called_strike', 'swinging_strike', 'foul', 'hit_into_play'])
        ]
        if use_decay:
            fp_w = first_pitches['w_perf'].sum()
            fs_w = first_strikes['w_perf'].sum()
            row['f_strike_pct'] = fs_w / fp_w if fp_w > 0 else 0
        else:
            row['f_strike_pct'] = len(first_strikes) / len(first_pitches) if len(first_pitches) > 0 else 0

        results.append(row)

    result_df = pd.DataFrame(results)
    print(f"  Computed arsenal for {len(result_df)} pitchers")
    return result_df


def get_pitcher_pitch_type_stats(
    start_date: str,
    end_date: str | None = None,
    min_pitches: int = 50,
    pitches_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute detailed stats for each pitch type a pitcher throws.

    Returns one row per pitcher-pitch_type combination with:
    - Usage%, velocity, spin, movement
    - Whiff%, called strike%, contact%
    - In-play results (avg EV, xwOBA)

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        min_pitches: Minimum pitches of a type to include
        pitches_df: Optional pre-loaded Statcast DataFrame

    Returns:
        DataFrame with one row per pitcher-pitch_type
    """
    if pitches_df is None:
        pitches_df = get_statcast_pitches(start_date, end_date)

    if pitches_df.empty:
        return pd.DataFrame()

    df = pitches_df.copy()
    df = df[df['pitcher'].notna() & df['pitch_type'].notna()]

    print("Computing pitcher pitch-type stats...")

    results = []

    for (pitcher_id, pitch_type), group in df.groupby(['pitcher', 'pitch_type']):
        if len(group) < min_pitches:
            continue

        # Get pitcher's total pitches for usage calculation
        pitcher_total = len(df[df['pitcher'] == pitcher_id])

        row = {
            'pitcher_id': int(pitcher_id),
            'pitcher_name': group['player_name'].iloc[0],
            'p_throws': group['p_throws'].iloc[0],
            'pitch_type': pitch_type,
            'pitch_count': len(group),
            'usage_pct': len(group) / pitcher_total,
        }

        # Velocity and movement
        row['velo_avg'] = group['release_speed'].mean()
        row['velo_max'] = group['release_speed'].max()
        row['spin_avg'] = group['release_spin_rate'].mean()
        row['vmov_avg'] = group['pfx_z'].mean()
        row['hmov_avg'] = group['pfx_x'].mean()
        row['extension'] = group['release_extension'].mean()

        # Outcomes
        swings = group[group['description'].isin(SWING_EVENTS)]
        whiffs = group[group['description'].isin(WHIFF_EVENTS)]
        called_strikes = group[group['description'] == 'called_strike']

        row['whiff_pct'] = len(whiffs) / len(swings) if len(swings) > 0 else 0
        row['csw_pct'] = (len(whiffs) + len(called_strikes)) / len(group)
        row['swing_pct'] = len(swings) / len(group)

        # In-play results
        in_play = group[group['type'] == 'X']
        if len(in_play) >= 10:
            row['avg_exit_velo'] = in_play['launch_speed'].mean()
            row['avg_launch_angle'] = in_play['launch_angle'].mean()
            row['xwoba_against'] = in_play['estimated_woba_using_speedangle'].mean()
            row['xba_against'] = in_play['estimated_ba_using_speedangle'].mean()

        results.append(row)

    result_df = pd.DataFrame(results)
    print(f"  Computed stats for {len(result_df)} pitcher-pitch combinations")
    return result_df


# =============================================================================
# BATTER STATS VS PITCH CHARACTERISTICS
# =============================================================================

def get_batter_pitch_stats(
    start_date: str,
    end_date: str | None = None,
    min_pitches: int = 100,
    min_pitch_type_count: int = 20,
    pitches_df: pd.DataFrame | None = None,
    use_decay: bool = True,
    performance_half_life: int = 90,
    batted_ball_half_life: int = 120,
) -> pd.DataFrame:
    """
    Compute batter performance against different pitch types and characteristics.

    Uses exponential decay weighting so recent data is weighted more heavily.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        min_pitches: Minimum pitches seen to include batter
        min_pitch_type_count: Minimum pitches of a type to compute stats
        pitches_df: Optional pre-loaded Statcast DataFrame
        use_decay: Whether to use exponential decay weighting (default True)
        performance_half_life: Half-life in days for performance metrics (whiff%, etc)
        batted_ball_half_life: Half-life in days for batted ball metrics (EV, xwOBA)

    Returns:
        DataFrame with one row per batter containing:
        - Overall: whiff%, contact%, chase%, zone_swing%
        - Per pitch type (vs_ff, vs_si, vs_sl, etc.): whiff_pct, contact_pct, xwoba
        - By velocity bucket: whiff%, contact% vs 90-93, 94-96, 97+
        - By movement type: whiff% vs high rise, heavy sweep, heavy drop
    """
    if pitches_df is None:
        pitches_df = get_statcast_pitches(start_date, end_date)

    if pitches_df.empty:
        return pd.DataFrame()

    df = pitches_df.copy()
    df = df[df['batter'].notna() & df['pitch_type'].notna()]

    # Calculate days ago for exponential decay
    if use_decay:
        if end_date is None:
            reference_date = datetime.now()
        else:
            reference_date = datetime.strptime(end_date, '%Y-%m-%d')
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['days_ago'] = (reference_date - df['game_date']).dt.days.clip(lower=0)

        # Precompute weights
        df['w_perf'] = np.exp(-df['days_ago'] / performance_half_life)
        df['w_bb'] = np.exp(-df['days_ago'] / batted_ball_half_life)

    print("Computing batter pitch stats...")

    results = []

    for batter_id, batter_df in df.groupby('batter'):
        if len(batter_df) < min_pitches:
            continue

        row = {'batter_id': int(batter_id)}
        row['batter_name'] = _get_batter_name(batter_df)
        row['stand'] = batter_df['stand'].iloc[0]
        row['total_pitches'] = len(batter_df)

        # === OVERALL STATS ===
        swings = batter_df[batter_df['description'].isin(SWING_EVENTS)]
        whiffs = batter_df[batter_df['description'].isin(WHIFF_EVENTS)]
        called_strikes = batter_df[batter_df['description'] == 'called_strike']

        if use_decay:
            total_w = batter_df['w_perf'].sum()
            swing_w = swings['w_perf'].sum()
            whiff_w = whiffs['w_perf'].sum()
            cs_w = called_strikes['w_perf'].sum()

            row['swing_pct'] = swing_w / total_w if total_w > 0 else 0
            row['whiff_pct'] = whiff_w / swing_w if swing_w > 0 else 0
            row['contact_pct'] = 1 - row['whiff_pct']
            row['csw_pct'] = (whiff_w + cs_w) / total_w if total_w > 0 else 0
        else:
            row['swing_pct'] = len(swings) / len(batter_df)
            row['whiff_pct'] = len(whiffs) / len(swings) if len(swings) > 0 else 0
            row['contact_pct'] = 1 - row['whiff_pct']
            row['csw_pct'] = (len(whiffs) + len(called_strikes)) / len(batter_df)

        # Zone discipline
        in_zone = batter_df[batter_df['zone'].between(1, 9)]
        out_zone = batter_df[~batter_df['zone'].between(1, 9)]
        zone_swings = in_zone[in_zone['description'].isin(SWING_EVENTS)]
        chase_swings = out_zone[out_zone['description'].isin(SWING_EVENTS)]

        if use_decay:
            iz_w = in_zone['w_perf'].sum()
            oz_w = out_zone['w_perf'].sum()
            zs_w = zone_swings['w_perf'].sum()
            cs_w = chase_swings['w_perf'].sum()

            row['zone_swing_pct'] = zs_w / iz_w if iz_w > 0 else 0
            row['chase_pct'] = cs_w / oz_w if oz_w > 0 else 0
        else:
            row['zone_swing_pct'] = len(zone_swings) / len(in_zone) if len(in_zone) > 0 else 0
            row['chase_pct'] = len(chase_swings) / len(out_zone) if len(out_zone) > 0 else 0

        # In-play quality (using batted ball half-life)
        in_play = batter_df[batter_df['type'] == 'X']
        if len(in_play) >= 20:
            if use_decay:
                w_bb = in_play['w_bb']
                w_sum = w_bb.sum()
                row['avg_exit_velo'] = (in_play['launch_speed'] * w_bb).sum() / w_sum if w_sum > 0 else np.nan
                row['max_exit_velo'] = in_play['launch_speed'].max()  # max is max, no weighting
                row['avg_launch_angle'] = (in_play['launch_angle'] * w_bb).sum() / w_sum if w_sum > 0 else np.nan
                row['xwoba'] = (in_play['estimated_woba_using_speedangle'] * w_bb).sum() / w_sum if w_sum > 0 else np.nan
                row['xba'] = (in_play['estimated_ba_using_speedangle'] * w_bb).sum() / w_sum if w_sum > 0 else np.nan

                # Barrel rate (weighted)
                barrels = in_play[
                    (in_play['launch_speed'] >= 98) &
                    (in_play['launch_angle'].between(26, 30))
                ]
                row['barrel_pct'] = barrels['w_bb'].sum() / w_sum if w_sum > 0 else 0
            else:
                row['avg_exit_velo'] = in_play['launch_speed'].mean()
                row['max_exit_velo'] = in_play['launch_speed'].max()
                row['avg_launch_angle'] = in_play['launch_angle'].mean()
                row['xwoba'] = in_play['estimated_woba_using_speedangle'].mean()
                row['xba'] = in_play['estimated_ba_using_speedangle'].mean()
                barrels = in_play[
                    (in_play['launch_speed'] >= 98) &
                    (in_play['launch_angle'].between(26, 30))
                ]
                row['barrel_pct'] = len(barrels) / len(in_play)

        # === BY INDIVIDUAL PITCH TYPE ===
        for pitch_type in ALL_PITCH_TYPES:
            pt_lower = pitch_type.lower()
            pt_df = batter_df[batter_df['pitch_type'] == pitch_type]

            if len(pt_df) >= min_pitch_type_count:
                pt_swings = pt_df[pt_df['description'].isin(SWING_EVENTS)]
                pt_whiffs = pt_df[pt_df['description'].isin(WHIFF_EVENTS)]

                if use_decay:
                    pts_w = pt_swings['w_perf'].sum()
                    ptw_w = pt_whiffs['w_perf'].sum()
                    row[f'vs_{pt_lower}_whiff_pct'] = ptw_w / pts_w if pts_w > 0 else 0
                else:
                    row[f'vs_{pt_lower}_whiff_pct'] = len(pt_whiffs) / len(pt_swings) if len(pt_swings) > 0 else 0
                row[f'vs_{pt_lower}_contact_pct'] = 1 - row[f'vs_{pt_lower}_whiff_pct']

                # In-play results vs this pitch type
                pt_in_play = pt_df[pt_df['type'] == 'X']
                if len(pt_in_play) >= 5:
                    if use_decay:
                        w_bb = pt_in_play['w_bb']
                        w_sum = w_bb.sum()
                        row[f'vs_{pt_lower}_xwoba'] = (pt_in_play['estimated_woba_using_speedangle'] * w_bb).sum() / w_sum if w_sum > 0 else np.nan
                    else:
                        row[f'vs_{pt_lower}_xwoba'] = pt_in_play['estimated_woba_using_speedangle'].mean()

        # === BY VELOCITY BUCKET ===
        velo_buckets = [
            ('velo_90_93', 90, 93.99),
            ('velo_94_96', 94, 96.99),
            ('velo_97_plus', 97, 110),
        ]

        for bucket_name, velo_min, velo_max in velo_buckets:
            bucket = batter_df[batter_df['release_speed'].between(velo_min, velo_max)]
            if len(bucket) >= 30:
                b_swings = bucket[bucket['description'].isin(SWING_EVENTS)]
                b_whiffs = bucket[bucket['description'].isin(WHIFF_EVENTS)]

                row[f'{bucket_name}_pitches'] = len(bucket)
                if use_decay:
                    bs_w = b_swings['w_perf'].sum()
                    bw_w = b_whiffs['w_perf'].sum()
                    bucket_w = bucket['w_perf'].sum()
                    row[f'{bucket_name}_whiff_pct'] = bw_w / bs_w if bs_w > 0 else 0
                    row[f'{bucket_name}_swing_pct'] = bs_w / bucket_w if bucket_w > 0 else 0
                else:
                    row[f'{bucket_name}_whiff_pct'] = len(b_whiffs) / len(b_swings) if len(b_swings) > 0 else 0
                    row[f'{bucket_name}_swing_pct'] = len(b_swings) / len(bucket)

        # === BY MOVEMENT TYPE ===
        # High vertical movement (rising fastballs)
        high_rise = batter_df[batter_df['pfx_z'] > 14]
        if len(high_rise) >= 30:
            hr_swings = high_rise[high_rise['description'].isin(SWING_EVENTS)]
            hr_whiffs = high_rise[high_rise['description'].isin(WHIFF_EVENTS)]
            if use_decay:
                hrs_w = hr_swings['w_perf'].sum()
                hrw_w = hr_whiffs['w_perf'].sum()
                row['high_rise_whiff_pct'] = hrw_w / hrs_w if hrs_w > 0 else 0
            else:
                row['high_rise_whiff_pct'] = len(hr_whiffs) / len(hr_swings) if len(hr_swings) > 0 else 0

        # Heavy horizontal sweep (sweepers, sliders)
        heavy_sweep = batter_df[abs(batter_df['pfx_x']) > 12]
        if len(heavy_sweep) >= 30:
            hs_swings = heavy_sweep[heavy_sweep['description'].isin(SWING_EVENTS)]
            hs_whiffs = heavy_sweep[heavy_sweep['description'].isin(WHIFF_EVENTS)]
            if use_decay:
                hss_w = hs_swings['w_perf'].sum()
                hsw_w = hs_whiffs['w_perf'].sum()
                row['heavy_sweep_whiff_pct'] = hsw_w / hss_w if hss_w > 0 else 0
            else:
                row['heavy_sweep_whiff_pct'] = len(hs_whiffs) / len(hs_swings) if len(hs_swings) > 0 else 0

        # Heavy vertical drop (curves, splitters)
        heavy_drop = batter_df[batter_df['pfx_z'] < -4]
        if len(heavy_drop) >= 30:
            hd_swings = heavy_drop[heavy_drop['description'].isin(SWING_EVENTS)]
            hd_whiffs = heavy_drop[heavy_drop['description'].isin(WHIFF_EVENTS)]
            if use_decay:
                hds_w = hd_swings['w_perf'].sum()
                hdw_w = hd_whiffs['w_perf'].sum()
                row['heavy_drop_whiff_pct'] = hdw_w / hds_w if hds_w > 0 else 0
            else:
                row['heavy_drop_whiff_pct'] = len(hd_whiffs) / len(hd_swings) if len(hd_swings) > 0 else 0

        # === VS HANDEDNESS ===
        for hand in ['L', 'R']:
            vs_hand = batter_df[batter_df['p_throws'] == hand]
            if len(vs_hand) >= 50:
                vh_swings = vs_hand[vs_hand['description'].isin(SWING_EVENTS)]
                vh_whiffs = vs_hand[vs_hand['description'].isin(WHIFF_EVENTS)]

                row[f'vs_{hand}HP_pitches'] = len(vs_hand)
                if use_decay:
                    vhs_w = vh_swings['w_perf'].sum()
                    vhw_w = vh_whiffs['w_perf'].sum()
                    row[f'vs_{hand}HP_whiff_pct'] = vhw_w / vhs_w if vhs_w > 0 else 0
                else:
                    row[f'vs_{hand}HP_whiff_pct'] = len(vh_whiffs) / len(vh_swings) if len(vh_swings) > 0 else 0

                vh_in_play = vs_hand[vs_hand['type'] == 'X']
                if len(vh_in_play) >= 10:
                    if use_decay:
                        w_bb = vh_in_play['w_bb']
                        w_sum = w_bb.sum()
                        row[f'vs_{hand}HP_xwoba'] = (vh_in_play['estimated_woba_using_speedangle'] * w_bb).sum() / w_sum if w_sum > 0 else np.nan
                    else:
                        row[f'vs_{hand}HP_xwoba'] = vh_in_play['estimated_woba_using_speedangle'].mean()

        results.append(row)

    result_df = pd.DataFrame(results)
    print(f"  Computed pitch stats for {len(result_df)} batters")
    return result_df


def get_batter_pitch_type_stats(
    start_date: str,
    end_date: str | None = None,
    min_pitches: int = 30,
    pitches_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute detailed stats for each pitch type a batter faces.

    Returns one row per batter-pitch_type combination.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        min_pitches: Minimum pitches seen of a type to include
        pitches_df: Optional pre-loaded Statcast DataFrame

    Returns:
        DataFrame with one row per batter-pitch_type
    """
    if pitches_df is None:
        pitches_df = get_statcast_pitches(start_date, end_date)

    if pitches_df.empty:
        return pd.DataFrame()

    df = pitches_df.copy()
    df = df[df['batter'].notna() & df['pitch_type'].notna()]

    print("Computing batter pitch-type stats...")

    results = []

    for (batter_id, pitch_type), group in df.groupby(['batter', 'pitch_type']):
        if len(group) < min_pitches:
            continue

        batter_total = len(df[df['batter'] == batter_id])

        row = {
            'batter_id': int(batter_id),
            'batter_name': _get_batter_name(group),
            'stand': group['stand'].iloc[0],
            'pitch_type': pitch_type,
            'pitch_count': len(group),
            'pct_of_pitches_seen': len(group) / batter_total,
        }

        # Swing/whiff stats
        swings = group[group['description'].isin(SWING_EVENTS)]
        whiffs = group[group['description'].isin(WHIFF_EVENTS)]

        row['swing_pct'] = len(swings) / len(group)
        row['whiff_pct'] = len(whiffs) / len(swings) if len(swings) > 0 else 0
        row['contact_pct'] = 1 - row['whiff_pct']

        # In-play results
        in_play = group[group['type'] == 'X']
        if len(in_play) >= 5:
            row['avg_exit_velo'] = in_play['launch_speed'].mean()
            row['avg_launch_angle'] = in_play['launch_angle'].mean()
            row['xwoba'] = in_play['estimated_woba_using_speedangle'].mean()
            row['xba'] = in_play['estimated_ba_using_speedangle'].mean()

        results.append(row)

    result_df = pd.DataFrame(results)
    print(f"  Computed stats for {len(result_df)} batter-pitch combinations")
    return result_df


def _get_batter_name(df: pd.DataFrame) -> str:
    """Extract batter name from Statcast data (not directly available)."""
    # Statcast doesn't have batter name directly, would need player lookup
    # For now return empty, can be joined later with player lookup
    return ""


# =============================================================================
# PLATE APPEARANCE OUTCOMES
# =============================================================================

def get_plate_appearances(
    start_date: str,
    end_date: str | None = None,
    pitches_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Extract plate appearance outcomes from Statcast data.

    Returns one row per plate appearance with pitcher/batter info and outcome.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        pitches_df: Optional pre-loaded Statcast DataFrame

    Returns:
        DataFrame with PA-level data including outcome categorization
    """
    if pitches_df is None:
        pitches_df = get_statcast_pitches(start_date, end_date)

    if pitches_df.empty:
        return pd.DataFrame()

    # Filter to pitches where PA ended (events column is populated)
    pa_df = pitches_df[pitches_df['events'].notna()].copy()

    print(f"Extracting {len(pa_df):,} plate appearances...")

    # Categorize outcomes
    outcome_map = {
        'strikeout': 'K',
        'strikeout_double_play': 'K',
        'walk': 'BB',
        'hit_by_pitch': 'HBP',
        'single': '1B',
        'double': '2B',
        'triple': '3B',
        'home_run': 'HR',
        'field_out': 'OUT',
        'grounded_into_double_play': 'OUT',
        'force_out': 'OUT',
        'sac_fly': 'OUT',
        'sac_bunt': 'OUT',
        'fielders_choice': 'OUT',
        'fielders_choice_out': 'OUT',
        'double_play': 'OUT',
        'triple_play': 'OUT',
        'field_error': 'OUT',
        'catcher_interf': 'OTHER',
        'sac_fly_double_play': 'OUT',
    }

    pa_df['outcome'] = pa_df['events'].map(outcome_map).fillna('OTHER')

    # Select relevant columns
    cols = [
        'game_pk', 'game_date', 'at_bat_number', 'inning', 'outs_when_up',
        'pitcher', 'player_name', 'p_throws',
        'batter', 'stand',
        'home_team', 'away_team',
        'events', 'outcome',
        'launch_speed', 'launch_angle',
        'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle',
    ]

    available_cols = [c for c in cols if c in pa_df.columns]
    result = pa_df[available_cols].copy()

    # Rename for clarity
    result = result.rename(columns={
        'pitcher': 'pitcher_id',
        'player_name': 'pitcher_name',
        'batter': 'batter_id',
    })

    # Add team columns
    result['is_home_batter'] = True  # Will compute properly
    # (inning_topbot would tell us but keep simple for now)

    return result


# =============================================================================
# COMBINED PROFILES
# =============================================================================

def get_pitcher_profiles(
    start_date: str,
    end_date: str | None = None,
    min_pitches: int = 200,
) -> pd.DataFrame:
    """
    Convenience function to get complete pitcher profiles.

    Combines arsenal stats with pitch-type breakdowns.
    """
    pitches = get_statcast_pitches(start_date, end_date)

    arsenal = get_pitcher_arsenal(
        start_date, end_date,
        min_pitches=min_pitches,
        pitches_df=pitches
    )

    return arsenal


def get_batter_profiles(
    start_date: str,
    end_date: str | None = None,
    min_pitches: int = 200,
) -> pd.DataFrame:
    """
    Convenience function to get complete batter profiles.

    Returns batter performance vs different pitch characteristics.
    """
    pitches = get_statcast_pitches(start_date, end_date)

    profiles = get_batter_pitch_stats(
        start_date, end_date,
        min_pitches=min_pitches,
        pitches_df=pitches
    )

    return profiles
