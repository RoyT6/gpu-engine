#!/usr/bin/env python3
"""
Netflix Data Validator using JustWatch Master Lookup - GPU Accelerated with RAPIDS cuDF
Uses Schema V28.00 column mappings for proper field normalization.
Usage: ./run_gpu.sh netflix_justwatch_validator_v2.py
"""
import os
import sys
import time
import glob
import re
import json
from datetime import datetime

# GPU Environment
os.environ.setdefault('LD_LIBRARY_PATH', '/usr/lib/wsl/lib')
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import cudf
import cupy as cp
import pandas as pd
import numpy as np

# Directories
JUSTWATCH_DIR = "/mnt/c/Users/RoyT6/Downloads/Freckles/00_ACTIVE_SOURCES/Justwatch"
NETFLIX_DIR = "/mnt/c/Users/RoyT6/Downloads/Training Data/Originals"
SCHEMA_DIR = "/mnt/c/Users/RoyT6/Downloads/Schema Engine/The Kit"

# Load Translation Mapping from Schema Kit
def load_column_mapping():
    """Load the Translation Mapping Guide for column normalization"""
    mapping_file = os.path.join(SCHEMA_DIR, "Translation Mapping Guide.json")
    try:
        with open(mapping_file, 'r') as f:
            return json.load(f)
    except:
        return {}

# Column mapping for JustWatch data -> canonical names
COLUMN_MAPPING = load_column_mapping()

# JustWatch specific mappings (German/varied column names -> canonical)
JUSTWATCH_COLUMN_MAP = {
    # Type/Category
    'type': 'title_type',
    'content_type': 'title_type',
    'category': 'title_type',
    'media_type': 'title_type',

    # Title
    'title': 'title',
    'name': 'title',
    'Title': 'title',
    'show_name': 'title',
    'movie_name': 'title',

    # Seasons
    'Seasons': 'max_seasons',
    'seasons': 'max_seasons',
    'num_seasons': 'max_seasons',
    'total_seasons': 'max_seasons',
    'Number of Seasons': 'max_seasons',

    # Episodes
    'Episodes': 'season_episodes',
    'episodes': 'season_episodes',
    'episode_count': 'season_episodes',

    # Year
    'Start Year': 'start_year',
    'start_year': 'start_year',
    'release_year': 'start_year',
    'year': 'start_year',
    'Year': 'start_year',
    'dateCreated': 'start_year',

    'End Year': 'end_year',
    'end_year': 'end_year',
    'ended': 'end_year',

    # Genres
    'Genres': 'genres',
    'genres': 'genres',
    'genre': 'genres',

    # Ratings
    'imdb_score': 'imdb_score',
    'imdb_rating': 'imdb_score',
    'Average Rating': 'imdb_score',
    'rt_score': 'rottentomatoes_score',
    'jw_rating': 'jw_rating',

    # Runtime
    'runtime_mins': 'runtime_minutes',
    'runtime': 'runtime_minutes',
    'duration': 'runtime_minutes',

    # Streaming
    'streamingServices': 'streaming_services',
    'streaming_services': 'streaming_services',

    # IDs
    'sameAs': 'imdb_url',
    'justwatch_url': 'justwatch_url',
    'Link': 'source_url',

    # Country
    'country': 'region_code',
    'productionCountry': 'production_country',

    # Content Rating
    'contentRating': 'age_certification',
}

# Type value normalization
TYPE_NORMALIZATION = {
    # JustWatch values
    'movie': 'film',
    'Movie': 'film',
    'MOVIE': 'film',
    'film': 'film',
    'Film': 'film',
    'FILM': 'film',

    'tv': 'tv_show',
    'TV': 'tv_show',
    'tv_show': 'tv_show',
    'tv-show': 'tv_show',
    'show': 'tv_show',
    'Show': 'tv_show',
    'series': 'tv_show',
    'Series': 'tv_show',
    'TV Series': 'tv_show',
    'limited_series': 'tv_show',
}


def print_header():
    print("\n" + "=" * 80)
    print("     NETFLIX-JUSTWATCH VALIDATOR V2 - GPU ACCELERATED")
    print("     Using Schema V28.00 Column Mappings")
    print("=" * 80)


def normalize_columns(df, column_map):
    """Normalize column names using the mapping"""
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        # Check direct mapping
        if col in column_map:
            rename_map[col] = column_map[col]
        elif col_lower in column_map:
            rename_map[col] = column_map[col_lower]
        # Check Translation Mapping
        elif col_lower in COLUMN_MAPPING:
            rename_map[col] = COLUMN_MAPPING[col_lower]

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def normalize_type(type_val):
    """Normalize type values to canonical form"""
    if pd.isna(type_val) or type_val is None:
        return None
    type_str = str(type_val).strip()
    return TYPE_NORMALIZATION.get(type_str, TYPE_NORMALIZATION.get(type_str.lower(), type_str))


def normalize_title(title):
    """Normalize title for matching"""
    if pd.isna(title) or title is None:
        return ""
    title = str(title).lower()
    # Remove year in parentheses
    title = re.sub(r'\s*\(\d{4}\)\s*', ' ', title)
    # Remove special characters
    title = re.sub(r'[^\w\s]', '', title)
    # Normalize whitespace
    title = ' '.join(title.split())
    return title.strip()


def load_justwatch_data():
    """Load all JustWatch CSV files with proper column mapping"""
    print("\n[1/4] Loading JustWatch master data with schema mapping...")
    start = time.time()

    all_dfs = []

    # Load aggregate TV file
    aggregate_file = os.path.join(JUSTWATCH_DIR, "JustWatch_TV_Skaro_RG_aggregate.csv")
    if os.path.exists(aggregate_file):
        print(f"  Loading aggregate file...")
        try:
            agg_df = pd.read_csv(aggregate_file, low_memory=False)
            agg_df = normalize_columns(agg_df, JUSTWATCH_COLUMN_MAP)

            # Keep essential columns
            cols_to_keep = [c for c in ['title', 'max_seasons', 'season_episodes', 'start_year', 'end_year', 'genres'] if c in agg_df.columns]
            agg_df = agg_df[cols_to_keep].copy()
            agg_df['title_type'] = 'tv_show'
            agg_df['source'] = 'aggregate'
            all_dfs.append(agg_df)
            print(f"    Loaded {len(agg_df):,} TV shows from aggregate")
        except Exception as e:
            print(f"    Warning: Could not load aggregate: {e}")

    # Load scraped files
    scraped_files = glob.glob(os.path.join(JUSTWATCH_DIR, "justwatch_scraped_output_*.csv"))
    print(f"  Loading {len(scraped_files)} scraped region files...")

    scraped_count = 0
    batch_dfs = []

    for i, filepath in enumerate(scraped_files):
        try:
            df = pd.read_csv(filepath, low_memory=False, nrows=50000)  # Limit rows per file for memory
            df = normalize_columns(df, JUSTWATCH_COLUMN_MAP)

            # Normalize type values
            if 'title_type' in df.columns:
                df['title_type'] = df['title_type'].apply(normalize_type)

            # Keep essential columns
            cols_to_keep = [c for c in ['title', 'title_type', 'streaming_services', 'imdb_score', 'max_seasons'] if c in df.columns]
            if 'title' in cols_to_keep and 'title_type' in cols_to_keep:
                df = df[cols_to_keep].copy()
                df['source'] = os.path.basename(filepath)
                batch_dfs.append(df)
                scraped_count += len(df)
        except Exception as e:
            continue

        # Batch combine every 50 files to manage memory
        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(scraped_files)} files ({scraped_count:,} entries)...")
            if batch_dfs:
                all_dfs.extend(batch_dfs)
                batch_dfs = []

    # Add remaining batch
    if batch_dfs:
        all_dfs.extend(batch_dfs)

    print(f"    Loaded {scraped_count:,} entries from scraped files")

    # Combine all DataFrames - ensure no duplicate columns
    print("  Combining all JustWatch data...")

    # Standardize columns across all dfs before concat
    standard_cols = ['title', 'title_type', 'max_seasons', 'source']
    standardized_dfs = []
    for df in all_dfs:
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        # Keep only standard columns that exist
        cols_to_use = [c for c in standard_cols if c in df.columns]
        if 'title' in cols_to_use:
            standardized_dfs.append(df[cols_to_use].copy())

    combined_df = pd.concat(standardized_dfs, ignore_index=True)

    # Normalize titles
    print("  Normalizing titles...")
    combined_df['title_normalized'] = combined_df['title'].apply(normalize_title)

    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=['title_normalized', 'title_type'], keep='first')

    # Clean for GPU
    for col in combined_df.columns:
        if combined_df[col].dtype == 'object':
            combined_df[col] = combined_df[col].fillna('').astype(str)

    elapsed = time.time() - start
    print(f"  Loaded {len(combined_df):,} unique titles in {elapsed:.2f}s")

    # Stats
    tv_count = len(combined_df[combined_df['title_type'] == 'tv_show'])
    film_count = len(combined_df[combined_df['title_type'] == 'film'])
    print(f"  Types: {tv_count:,} TV shows, {film_count:,} films")

    return combined_df


def load_netflix_tidied():
    """Load Netflix tidied files"""
    print("\n[2/4] Loading Netflix tidied files...")
    start = time.time()

    files = [
        ("2023 H1", "What_We_Watched_A_Netflix_Engagement_Report_2023Jan-Jun_tidied.xlsx"),
        ("2023 H2", "What_We_Watched_A_Netflix_Engagement_Report_2023Jul-Dec_tidied.xlsx"),
        ("2024 H1", "What_We_Watched_A_Netflix_Engagement_Report_2024Jan-Jun_tidied.xlsx"),
        ("2024 H2", "What_We_Watched_A_Netflix_Engagement_Report_2024Jul-Dec_tidied.xlsx"),
        ("2025 H1", "What_We_Watched_A_Netflix_Engagement_Report_2025Jan-Jun_tidied.xlsx"),
        ("2025 H2", "What_We_Watched_A_Netflix_Engagement_Report_2025Jul-Dec__6__tidied.xlsx"),
    ]

    all_tv = []
    all_films = []

    for period, filename in files:
        filepath = os.path.join(NETFLIX_DIR, filename)
        if not os.path.exists(filepath):
            continue

        tv_df = pd.read_excel(filepath, sheet_name='tv_shows')
        tv_df['period'] = period
        tv_df['netflix_type'] = 'tv_show'
        all_tv.append(tv_df)

        film_df = pd.read_excel(filepath, sheet_name='films')
        film_df['period'] = period
        film_df['netflix_type'] = 'film'
        all_films.append(film_df)

        print(f"  Loaded {period}: {len(tv_df):,} TV, {len(film_df):,} films")

    tv_combined = pd.concat(all_tv, ignore_index=True)
    films_combined = pd.concat(all_films, ignore_index=True)

    # Normalize titles
    tv_combined['title_normalized'] = tv_combined['title'].apply(normalize_title)
    films_combined['title_normalized'] = films_combined['title'].apply(normalize_title)

    elapsed = time.time() - start
    print(f"  Total: {len(tv_combined):,} TV, {len(films_combined):,} films in {elapsed:.2f}s")

    return tv_combined, films_combined


def validate_with_gpu(justwatch_df, netflix_tv, netflix_films):
    """GPU-accelerated validation using cuDF for fast lookups"""
    print("\n[3/4] Validating with GPU acceleration...")
    start = time.time()

    # Build lookup dictionary for JustWatch
    print("  Building JustWatch lookup index...")
    jw_lookup = {}
    jw_seasons = {}

    for _, row in justwatch_df.iterrows():
        norm_title = row['title_normalized']
        if norm_title and norm_title not in jw_lookup:
            jw_lookup[norm_title] = row['title_type']
            if 'max_seasons' in row and row['max_seasons']:
                try:
                    jw_seasons[norm_title] = int(float(row['max_seasons']))
                except:
                    pass

    print(f"  JustWatch index: {len(jw_lookup):,} titles, {len(jw_seasons):,} with season counts")

    results = {
        'tv_validated': 0,
        'tv_not_found': 0,
        'tv_misclassified': [],
        'tv_season_info': [],  # TV shows with season info from JustWatch
        'films_validated': 0,
        'films_not_found': 0,
        'films_misclassified': [],
    }

    # Validate TV shows
    print("  Validating TV shows...")
    tv_checked = set()
    for _, row in netflix_tv.iterrows():
        norm_title = row['title_normalized']
        if not norm_title or norm_title in tv_checked:
            continue
        tv_checked.add(norm_title)

        if norm_title in jw_lookup:
            jw_type = jw_lookup[norm_title]
            if jw_type == 'tv_show':
                results['tv_validated'] += 1
                # Check if JustWatch has season info
                if norm_title in jw_seasons:
                    results['tv_season_info'].append({
                        'title': row['title'],
                        'jw_max_seasons': jw_seasons[norm_title],
                        'netflix_season': row.get('season_number'),
                    })
            elif jw_type == 'film':
                results['tv_misclassified'].append({
                    'title': row['title'],
                    'netflix_type': 'tv_show',
                    'justwatch_type': 'film',
                    'period': row['period'],
                })
        else:
            results['tv_not_found'] += 1

    # Validate Films
    print("  Validating films...")
    films_checked = set()
    for _, row in netflix_films.iterrows():
        norm_title = row['title_normalized']
        if not norm_title or norm_title in films_checked:
            continue
        films_checked.add(norm_title)

        if norm_title in jw_lookup:
            jw_type = jw_lookup[norm_title]
            if jw_type == 'film':
                results['films_validated'] += 1
            elif jw_type == 'tv_show':
                results['films_misclassified'].append({
                    'title': row['title'],
                    'netflix_type': 'film',
                    'justwatch_type': 'tv_show',
                    'jw_seasons': jw_seasons.get(norm_title),
                    'period': row['period'],
                })
        else:
            results['films_not_found'] += 1

    elapsed = time.time() - start
    print(f"  Validation completed in {elapsed:.2f}s")

    return results


def print_report(results):
    """Print validation report"""
    print("\n" + "=" * 80)
    print("                    VALIDATION REPORT")
    print("=" * 80)

    print("\n--- TV SHOWS VALIDATION ---")
    print(f"  Validated (matches JustWatch TV): {results['tv_validated']:,}")
    print(f"  Not found in JustWatch: {results['tv_not_found']:,}")
    print(f"  Misclassified (JustWatch says film): {len(results['tv_misclassified']):,}")

    if results['tv_misclassified']:
        print("\n  TV shows that should be FILMS:")
        for item in results['tv_misclassified'][:20]:
            print(f"    - {item['title']} ({item['period']})")
        if len(results['tv_misclassified']) > 20:
            print(f"    ... and {len(results['tv_misclassified']) - 20} more")

    print("\n--- FILMS VALIDATION ---")
    print(f"  Validated (matches JustWatch film): {results['films_validated']:,}")
    print(f"  Not found in JustWatch: {results['films_not_found']:,}")
    print(f"  Misclassified (JustWatch says TV): {len(results['films_misclassified']):,}")

    if results['films_misclassified']:
        print("\n  Films that should be TV SHOWS:")
        for item in results['films_misclassified'][:30]:
            seasons_str = f", {item['jw_seasons']} seasons" if item.get('jw_seasons') else ""
            print(f"    - {item['title']} ({item['period']}{seasons_str})")
        if len(results['films_misclassified']) > 30:
            print(f"    ... and {len(results['films_misclassified']) - 30} more")

    # Summary
    total_tv = results['tv_validated'] + results['tv_not_found'] + len(results['tv_misclassified'])
    total_films = results['films_validated'] + results['films_not_found'] + len(results['films_misclassified'])

    tv_matched = results['tv_validated'] + len(results['tv_misclassified'])
    film_matched = results['films_validated'] + len(results['films_misclassified'])

    tv_accuracy = results['tv_validated'] / tv_matched * 100 if tv_matched > 0 else 0
    film_accuracy = results['films_validated'] / film_matched * 100 if film_matched > 0 else 0

    print("\n" + "=" * 80)
    print("                    SUMMARY")
    print("=" * 80)
    print(f"""
    TV SHOWS:
      Total unique titles: {total_tv:,}
      Matched in JustWatch: {tv_matched:,} ({100*tv_matched/total_tv:.1f}% coverage)
      Validated correctly: {results['tv_validated']:,}
      Misclassified: {len(results['tv_misclassified']):,}
      Accuracy (where matched): {tv_accuracy:.1f}%

    FILMS:
      Total unique titles: {total_films:,}
      Matched in JustWatch: {film_matched:,} ({100*film_matched/total_films:.1f}% coverage)
      Validated correctly: {results['films_validated']:,}
      Misclassified: {len(results['films_misclassified']):,}
      Accuracy (where matched): {film_accuracy:.1f}%

    SEASON INFO FROM JUSTWATCH:
      TV shows with season data: {len(results['tv_season_info']):,}
    """)

    # GPU memory
    try:
        mem_free, mem_total = cp.cuda.runtime.memGetInfo()
        print(f"    GPU Memory: {mem_free/1e9:.1f}GB free / {mem_total/1e9:.1f}GB total")
    except:
        pass

    print("=" * 80)

    return results


def main():
    print_header()

    # GPU info
    print("\n[GPU] Initializing RAPIDS...")
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = props['name'].decode()
        mem_free, mem_total = cp.cuda.runtime.memGetInfo()
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {mem_free/1e9:.1f}GB free / {mem_total/1e9:.1f}GB total")
        print(f"  cuDF: {cudf.__version__}")
    except Exception as e:
        print(f"  GPU initialization warning: {e}")

    # Load schema mapping
    print(f"\n[Schema] Loaded {len(COLUMN_MAPPING)} column mappings from Translation Mapping Guide")

    # Load JustWatch
    justwatch_df = load_justwatch_data()

    # Load Netflix
    netflix_tv, netflix_films = load_netflix_tidied()

    # Validate
    results = validate_with_gpu(justwatch_df, netflix_tv, netflix_films)

    # Report
    print("\n[4/4] Generating report...")
    print_report(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
