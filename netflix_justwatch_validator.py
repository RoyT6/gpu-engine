#!/usr/bin/env python3
"""
Netflix Data Validator using JustWatch Master Lookup - GPU Accelerated with RAPIDS cuDF
Cross-references Netflix tidied files against JustWatch data to validate film vs TV classification.
Usage: ./run_gpu.sh netflix_justwatch_validator.py
"""
import os
import sys
import time
import glob
import re
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


def print_header():
    print("\n" + "=" * 80)
    print("     NETFLIX-JUSTWATCH VALIDATOR - GPU ACCELERATED")
    print("=" * 80)


def normalize_title(title):
    """Normalize title for matching - lowercase, remove special chars"""
    if pd.isna(title) or title is None:
        return ""
    title = str(title).lower()
    # Remove year in parentheses like "(2005)"
    title = re.sub(r'\s*\(\d{4}\)\s*', ' ', title)
    # Remove special characters
    title = re.sub(r'[^\w\s]', '', title)
    # Normalize whitespace
    title = ' '.join(title.split())
    return title.strip()


def load_justwatch_data():
    """Load all JustWatch CSV files into a single cuDF DataFrame"""
    print("\n[1/4] Loading JustWatch master data...")
    start = time.time()

    # Load aggregate TV file first
    aggregate_file = os.path.join(JUSTWATCH_DIR, "JustWatch_TV_Skaro_RG_aggregate.csv")

    all_dfs = []

    # Load aggregate (TV shows with detailed info)
    if os.path.exists(aggregate_file):
        print(f"  Loading aggregate file...")
        try:
            agg_df = pd.read_csv(aggregate_file, low_memory=False)
            # Extract key columns
            agg_df = agg_df[['Title', 'Seasons', 'Episodes', 'Start Year', 'End Year', 'Genres']].copy()
            agg_df.columns = ['title', 'seasons', 'episodes', 'start_year', 'end_year', 'genres']
            agg_df['type'] = 'tv'
            agg_df['source'] = 'aggregate'
            all_dfs.append(agg_df)
            print(f"    Loaded {len(agg_df):,} TV shows from aggregate")
        except Exception as e:
            print(f"    Warning: Could not load aggregate: {e}")

    # Load scraped files (have both movies and TV)
    scraped_files = glob.glob(os.path.join(JUSTWATCH_DIR, "justwatch_scraped_output_*.csv"))
    print(f"  Loading {len(scraped_files)} scraped region files...")

    scraped_count = 0
    for i, filepath in enumerate(scraped_files):
        try:
            df = pd.read_csv(filepath, low_memory=False)
            if 'title' in df.columns and 'type' in df.columns:
                # Keep essential columns
                cols_to_keep = ['title', 'type']
                if 'streamingServices' in df.columns:
                    cols_to_keep.append('streamingServices')
                if 'jw_rating' in df.columns:
                    cols_to_keep.append('jw_rating')
                if 'imdb_score' in df.columns:
                    cols_to_keep.append('imdb_score')

                df = df[cols_to_keep].copy()
                df['source'] = os.path.basename(filepath)
                all_dfs.append(df)
                scraped_count += len(df)
        except Exception as e:
            continue

        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(scraped_files)} files...")

    print(f"    Loaded {scraped_count:,} entries from scraped files")

    # Combine all DataFrames
    print("  Combining all JustWatch data...")
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Normalize titles for matching
    print("  Normalizing titles for matching...")
    combined_df['title_normalized'] = combined_df['title'].apply(normalize_title)

    # Remove duplicates - keep first occurrence (prefer aggregate data)
    combined_df = combined_df.drop_duplicates(subset=['title_normalized', 'type'], keep='first')

    # Convert to cuDF for GPU operations
    print("  Converting to cuDF...")
    try:
        # Clean for cuDF
        for col in combined_df.columns:
            if combined_df[col].dtype == 'object':
                combined_df[col] = combined_df[col].fillna('').astype(str)

        gdf = cudf.DataFrame.from_pandas(combined_df)
    except Exception as e:
        print(f"  Warning: cuDF conversion failed, using pandas: {e}")
        gdf = combined_df

    elapsed = time.time() - start
    print(f"  Loaded {len(gdf):,} unique titles in {elapsed:.2f}s")

    # Stats
    if hasattr(gdf, 'to_pandas'):
        pdf = gdf.to_pandas()
    else:
        pdf = gdf
    tv_count = len(pdf[pdf['type'] == 'tv'])
    movie_count = len(pdf[pdf['type'] == 'movie'])
    print(f"  Types: {tv_count:,} TV shows, {movie_count:,} movies")

    return gdf


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

        # TV shows
        tv_df = pd.read_excel(filepath, sheet_name='tv_shows')
        tv_df['period'] = period
        tv_df['netflix_type'] = 'tv_show'
        all_tv.append(tv_df)

        # Films
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


def validate_classifications(justwatch_gdf, netflix_tv, netflix_films):
    """Cross-reference Netflix data against JustWatch to find misclassifications"""
    print("\n[3/4] Validating classifications with GPU...")
    start = time.time()

    # Convert JustWatch to pandas for merge (cuDF merge can be tricky with strings)
    if hasattr(justwatch_gdf, 'to_pandas'):
        jw_pdf = justwatch_gdf.to_pandas()
    else:
        jw_pdf = justwatch_gdf

    results = {
        'tv_validated': 0,
        'tv_not_found': 0,
        'tv_misclassified': [],  # Should be films
        'films_validated': 0,
        'films_not_found': 0,
        'films_misclassified': [],  # Should be TV
    }

    # Create lookup dict for faster matching
    print("  Building JustWatch index...")
    jw_lookup = {}
    for _, row in jw_pdf.iterrows():
        norm_title = row['title_normalized']
        if norm_title and norm_title not in jw_lookup:
            jw_lookup[norm_title] = row['type']

    print(f"  JustWatch index: {len(jw_lookup):,} unique normalized titles")

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
            if jw_type == 'tv':
                results['tv_validated'] += 1
            elif jw_type == 'movie':
                results['tv_misclassified'].append({
                    'title': row['title'],
                    'netflix_type': 'tv_show',
                    'justwatch_type': 'movie',
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
            if jw_type == 'movie':
                results['films_validated'] += 1
            elif jw_type == 'tv':
                results['films_misclassified'].append({
                    'title': row['title'],
                    'netflix_type': 'film',
                    'justwatch_type': 'tv',
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
    print(f"  Misclassified (JustWatch says movie): {len(results['tv_misclassified']):,}")

    if results['tv_misclassified']:
        print("\n  TV shows that should be FILMS:")
        for item in results['tv_misclassified'][:20]:
            print(f"    - {item['title']} ({item['period']})")
        if len(results['tv_misclassified']) > 20:
            print(f"    ... and {len(results['tv_misclassified']) - 20} more")

    print("\n--- FILMS VALIDATION ---")
    print(f"  Validated (matches JustWatch movie): {results['films_validated']:,}")
    print(f"  Not found in JustWatch: {results['films_not_found']:,}")
    print(f"  Misclassified (JustWatch says TV): {len(results['films_misclassified']):,}")

    if results['films_misclassified']:
        print("\n  Films that should be TV SHOWS:")
        for item in results['films_misclassified'][:20]:
            print(f"    - {item['title']} ({item['period']})")
        if len(results['films_misclassified']) > 20:
            print(f"    ... and {len(results['films_misclassified']) - 20} more")

    # Summary
    total_tv = results['tv_validated'] + results['tv_not_found'] + len(results['tv_misclassified'])
    total_films = results['films_validated'] + results['films_not_found'] + len(results['films_misclassified'])

    tv_accuracy = results['tv_validated'] / (results['tv_validated'] + len(results['tv_misclassified'])) * 100 if (results['tv_validated'] + len(results['tv_misclassified'])) > 0 else 0
    film_accuracy = results['films_validated'] / (results['films_validated'] + len(results['films_misclassified'])) * 100 if (results['films_validated'] + len(results['films_misclassified'])) > 0 else 0

    print("\n" + "=" * 80)
    print("                    SUMMARY")
    print("=" * 80)
    print(f"""
    TV SHOWS:
      Total unique titles checked: {total_tv:,}
      Validated against JustWatch: {results['tv_validated']:,}
      Potential misclassifications: {len(results['tv_misclassified']):,}
      Coverage: {100 * (results['tv_validated'] + len(results['tv_misclassified'])) / total_tv:.1f}%
      Accuracy (where matched): {tv_accuracy:.1f}%

    FILMS:
      Total unique titles checked: {total_films:,}
      Validated against JustWatch: {results['films_validated']:,}
      Potential misclassifications: {len(results['films_misclassified']):,}
      Coverage: {100 * (results['films_validated'] + len(results['films_misclassified'])) / total_films:.1f}%
      Accuracy (where matched): {film_accuracy:.1f}%
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

    # Load JustWatch master data
    justwatch_gdf = load_justwatch_data()

    # Load Netflix tidied data
    netflix_tv, netflix_films = load_netflix_tidied()

    # Validate
    results = validate_classifications(justwatch_gdf, netflix_tv, netflix_films)

    # Report
    print("\n[4/4] Generating report...")
    print_report(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
