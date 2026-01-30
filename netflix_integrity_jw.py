#!/usr/bin/env python3
"""
Netflix Integrity Check using JustWatch - GPU Accelerated
Purpose: Validate title_type (film vs tv_show) classification only
Uses Translation Mapping Guide for column normalization
Usage: ./run_gpu.sh netflix_integrity_jw.py
"""
import os
import sys
import time
import glob
import re
import json

os.environ.setdefault('LD_LIBRARY_PATH', '/usr/lib/wsl/lib')
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import cudf
import cupy as cp
import pandas as pd

JUSTWATCH_DIR = "/mnt/c/Users/RoyT6/Downloads/Freckles/00_ACTIVE_SOURCES/Justwatch"
NETFLIX_DIR = "/mnt/c/Users/RoyT6/Downloads/Training Data/Originals"
SCHEMA_DIR = "/mnt/c/Users/RoyT6/Downloads/Schema Engine/The Kit"

# Load Translation Mapping
with open(os.path.join(SCHEMA_DIR, "Translation Mapping Guide.json"), 'r') as f:
    TRANSLATION_MAP = json.load(f)

# Type value normalization: movie -> film, tv -> tv_show
TYPE_MAP = {
    'movie': 'film', 'Movie': 'film', 'MOVIE': 'film', 'film': 'film', 'Film': 'film',
    'tv': 'tv_show', 'TV': 'tv_show', 'tv_show': 'tv_show', 'show': 'tv_show',
    'series': 'tv_show', 'Series': 'tv_show', 'limited_series': 'tv_show'
}


def normalize_title(title):
    """Normalize title for fuzzy matching"""
    if pd.isna(title) or not title:
        return ""
    title = str(title).lower()
    title = re.sub(r'\s*\(\d{4}\)\s*', ' ', title)  # Remove year
    title = re.sub(r'[^\w\s]', '', title)  # Remove special chars
    return ' '.join(title.split()).strip()


def load_justwatch_lookup():
    """Load JustWatch data as simple title -> type lookup"""
    print("\n[1/3] Building JustWatch lookup index...")
    start = time.time()

    lookup = {}  # normalized_title -> title_type

    # Load aggregate (TV shows)
    agg_file = os.path.join(JUSTWATCH_DIR, "JustWatch_TV_Skaro_RG_aggregate.csv")
    if os.path.exists(agg_file):
        df = pd.read_csv(agg_file, usecols=['Title'], low_memory=False)
        for title in df['Title'].dropna():
            norm = normalize_title(title)
            if norm:
                lookup[norm] = 'tv_show'
        print(f"  Aggregate: {len(df):,} TV shows indexed")

    # Load scraped files
    scraped_files = glob.glob(os.path.join(JUSTWATCH_DIR, "justwatch_scraped_output_*.csv"))
    print(f"  Loading {len(scraped_files)} scraped files...")

    scraped_count = 0
    for i, filepath in enumerate(scraped_files):
        try:
            df = pd.read_csv(filepath, usecols=['title', 'type'], low_memory=False)
            for _, row in df.iterrows():
                title = row.get('title')
                jw_type = row.get('type')
                if pd.isna(title) or pd.isna(jw_type):
                    continue
                norm = normalize_title(title)
                if norm:
                    # Map type: movie -> film, tv -> tv_show
                    canonical_type = TYPE_MAP.get(str(jw_type).strip(), str(jw_type))
                    # Only set if not already set (prefer aggregate TV data)
                    if norm not in lookup:
                        lookup[norm] = canonical_type
                    scraped_count += 1
        except Exception as e:
            continue

        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(scraped_files)} files...")

    print(f"  Scraped: {scraped_count:,} entries processed")

    # Count types
    tv_count = sum(1 for v in lookup.values() if v == 'tv_show')
    film_count = sum(1 for v in lookup.values() if v == 'film')

    elapsed = time.time() - start
    print(f"  Index: {len(lookup):,} unique titles ({tv_count:,} TV, {film_count:,} films) in {elapsed:.1f}s")

    return lookup


def load_netflix_titles():
    """Load Netflix tidied titles"""
    print("\n[2/3] Loading Netflix tidied files...")
    start = time.time()

    files = [
        ("2023 H1", "What_We_Watched_A_Netflix_Engagement_Report_2023Jan-Jun_tidied.xlsx"),
        ("2023 H2", "What_We_Watched_A_Netflix_Engagement_Report_2023Jul-Dec_tidied.xlsx"),
        ("2024 H1", "What_We_Watched_A_Netflix_Engagement_Report_2024Jan-Jun_tidied.xlsx"),
        ("2024 H2", "What_We_Watched_A_Netflix_Engagement_Report_2024Jul-Dec_tidied.xlsx"),
        ("2025 H1", "What_We_Watched_A_Netflix_Engagement_Report_2025Jan-Jun_tidied.xlsx"),
        ("2025 H2", "What_We_Watched_A_Netflix_Engagement_Report_2025Jul-Dec__6__tidied.xlsx"),
    ]

    netflix_titles = {}  # normalized_title -> {'title': original, 'netflix_type': type, 'periods': []}

    for period, filename in files:
        filepath = os.path.join(NETFLIX_DIR, filename)
        if not os.path.exists(filepath):
            continue

        # TV shows
        tv_df = pd.read_excel(filepath, sheet_name='tv_shows')
        for title in tv_df['title'].dropna():
            norm = normalize_title(title)
            if norm:
                if norm not in netflix_titles:
                    netflix_titles[norm] = {'title': title, 'netflix_type': 'tv_show', 'periods': []}
                netflix_titles[norm]['periods'].append(period)

        # Films
        film_df = pd.read_excel(filepath, sheet_name='films')
        for title in film_df['title'].dropna():
            norm = normalize_title(title)
            if norm:
                if norm not in netflix_titles:
                    netflix_titles[norm] = {'title': title, 'netflix_type': 'film', 'periods': []}
                netflix_titles[norm]['periods'].append(period)

        print(f"  {period}: {len(tv_df):,} TV, {len(film_df):,} films")

    elapsed = time.time() - start
    print(f"  Total: {len(netflix_titles):,} unique titles in {elapsed:.1f}s")

    return netflix_titles


def validate_integrity(jw_lookup, netflix_titles):
    """Validate Netflix classifications against JustWatch"""
    print("\n[3/3] Validating title_type integrity...")
    start = time.time()

    results = {
        'tv_correct': 0,
        'tv_wrong': [],  # Netflix says TV, JustWatch says film
        'film_correct': 0,
        'film_wrong': [],  # Netflix says film, JustWatch says TV
        'not_found': 0,
    }

    for norm_title, data in netflix_titles.items():
        netflix_type = data['netflix_type']

        if norm_title in jw_lookup:
            jw_type = jw_lookup[norm_title]

            if netflix_type == jw_type:
                if netflix_type == 'tv_show':
                    results['tv_correct'] += 1
                else:
                    results['film_correct'] += 1
            else:
                # Mismatch
                if netflix_type == 'tv_show':
                    results['tv_wrong'].append({
                        'title': data['title'],
                        'netflix': 'tv_show',
                        'justwatch': jw_type,
                        'periods': data['periods'][:2]
                    })
                else:
                    results['film_wrong'].append({
                        'title': data['title'],
                        'netflix': 'film',
                        'justwatch': jw_type,
                        'periods': data['periods'][:2]
                    })
        else:
            results['not_found'] += 1

    elapsed = time.time() - start
    print(f"  Validation completed in {elapsed:.1f}s")

    return results


def print_report(results, netflix_titles):
    """Print integrity report"""
    print("\n" + "=" * 80)
    print("           NETFLIX TITLE_TYPE INTEGRITY REPORT")
    print("=" * 80)

    total = len(netflix_titles)
    matched = results['tv_correct'] + results['film_correct'] + len(results['tv_wrong']) + len(results['film_wrong'])

    print(f"""
    SUMMARY:
      Total Netflix titles: {total:,}
      Matched in JustWatch: {matched:,} ({100*matched/total:.1f}%)
      Not found: {results['not_found']:,}

    VALIDATION RESULTS:
      TV shows correct: {results['tv_correct']:,}
      TV shows WRONG (should be film): {len(results['tv_wrong']):,}

      Films correct: {results['film_correct']:,}
      Films WRONG (should be tv_show): {len(results['film_wrong']):,}
    """)

    if results['tv_wrong']:
        print("=" * 80)
        print("    TV SHOWS THAT SHOULD BE FILMS")
        print("-" * 80)
        for item in results['tv_wrong'][:25]:
            print(f"      {item['title']}")
        if len(results['tv_wrong']) > 25:
            print(f"      ... and {len(results['tv_wrong']) - 25} more")

    if results['film_wrong']:
        print("\n" + "=" * 80)
        print("    FILMS THAT SHOULD BE TV SHOWS")
        print("-" * 80)
        for item in results['film_wrong'][:25]:
            print(f"      {item['title']}")
        if len(results['film_wrong']) > 25:
            print(f"      ... and {len(results['film_wrong']) - 25} more")

    # Accuracy
    correct = results['tv_correct'] + results['film_correct']
    wrong = len(results['tv_wrong']) + len(results['film_wrong'])
    accuracy = 100 * correct / matched if matched > 0 else 0

    print("\n" + "=" * 80)
    print(f"    INTEGRITY SCORE: {accuracy:.2f}%")
    print(f"    ({correct:,} correct, {wrong:,} misclassified out of {matched:,} matched)")
    print("=" * 80)


def main():
    print("\n" + "=" * 80)
    print("     NETFLIX TITLE_TYPE INTEGRITY CHECK")
    print("     Using JustWatch as validation source")
    print("=" * 80)

    # GPU info
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        mem_free, mem_total = cp.cuda.runtime.memGetInfo()
        print(f"\n  GPU: {props['name'].decode()}")
        print(f"  VRAM: {mem_free/1e9:.1f}GB free / {mem_total/1e9:.1f}GB total")
    except:
        pass

    # Load JustWatch lookup
    jw_lookup = load_justwatch_lookup()

    # Load Netflix titles
    netflix_titles = load_netflix_titles()

    # Validate
    results = validate_integrity(jw_lookup, netflix_titles)

    # Report
    print_report(results, netflix_titles)

    return 0


if __name__ == "__main__":
    sys.exit(main())
