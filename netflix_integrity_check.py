#!/usr/bin/env python3
"""
Netflix Data Integrity Check - GPU Accelerated with RAPIDS cuDF
Verifies film vs TV show classification across all tidied files.
Usage: ./run_gpu.sh netflix_integrity_check.py
"""
import os
import sys
import time
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
import re

# Data directory
DATA_DIR = "/mnt/c/Users/RoyT6/Downloads/Training Data/Originals"

# Known film franchises (should NOT be in TV shows)
FILM_FRANCHISE_PATTERNS = [
    r'^Iron Man',
    r'^Spider-Man',
    r'^Avengers',
    r'^Thor\s*\d*$',
    r'^Shrek',
    r'^Toy Story',
    r'^John Wick',
    r'^Fast.*Furious',
    r'^Transformers',
    r'^Harry Potter',
    r'^The Godfather',
    r'^The Matrix',
]

# Known TV shows that use "Title Number" format (NOT "Title: Season X")
KNOWN_TITLE_NUMBER_SHOWS = [
    'Stranger Things',
    'Avatar: The Last Airbender',
    'The Great British Baking Show',
    'My Little Pony',
    'Unicorn Academy',
    'The Creature Cases',
    'Downton Abbey',
    'Call the Midwife',
    'DOTA',
    'Weak Hero',
]

# Patterns that indicate TV content
TV_PATTERNS = [
    r': Season \d+',
    r': Limited Series',
    r': Part \d+',
    r': Volume \d+',
    r': Series \d+',
    r': Book \d+',
    r': Chapter \d+',
    r': Collection \d+',
    r': Temporada \d+',
]


def print_header():
    print("\n" + "=" * 80)
    print("     NETFLIX DATA INTEGRITY CHECK - GPU ACCELERATED")
    print("=" * 80)


def load_tidied_files():
    """Load all tidied Excel files into cuDF DataFrames"""
    files = [
        ("2023 H1", "What_We_Watched_A_Netflix_Engagement_Report_2023Jan-Jun_tidied.xlsx"),
        ("2023 H2", "What_We_Watched_A_Netflix_Engagement_Report_2023Jul-Dec_tidied.xlsx"),
        ("2024 H1", "What_We_Watched_A_Netflix_Engagement_Report_2024Jan-Jun_tidied.xlsx"),
        ("2024 H2", "What_We_Watched_A_Netflix_Engagement_Report_2024Jul-Dec_tidied.xlsx"),
        ("2025 H1", "What_We_Watched_A_Netflix_Engagement_Report_2025Jan-Jun_tidied.xlsx"),
        ("2025 H2", "What_We_Watched_A_Netflix_Engagement_Report_2025Jul-Dec__6__tidied.xlsx"),
    ]

    all_data = []

    for period, filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  WARNING: File not found: {filename}")
            continue

        # Load TV shows
        tv_pdf = pd.read_excel(filepath, sheet_name='tv_shows')
        tv_pdf['source_sheet'] = 'tv_shows'
        tv_pdf['period'] = period

        # Load Films
        film_pdf = pd.read_excel(filepath, sheet_name='films')
        film_pdf['source_sheet'] = 'films'
        film_pdf['period'] = period

        all_data.append(('tv', period, tv_pdf))
        all_data.append(('film', period, film_pdf))

        print(f"  Loaded {period}: {len(tv_pdf)} TV, {len(film_pdf)} Films")

    return all_data


def check_tv_missing_seasons(tv_df):
    """Find TV shows that should have season numbers but don't"""
    issues = []

    # Pattern: Title ends with space + number (1-20 range, likely season)
    title_number_pattern = re.compile(r'^(.+?)\s+(\d+)$')

    for idx, row in tv_df.iterrows():
        title = str(row.get('title', ''))
        season_num = row.get('season_number')

        if pd.isna(season_num) or season_num is None:
            match = title_number_pattern.match(title)
            if match:
                potential_season = int(match.group(2))
                if 1 <= potential_season <= 20:
                    issues.append({
                        'title': title,
                        'potential_base': match.group(1),
                        'potential_season': potential_season,
                        'issue': 'missing_season_extraction'
                    })

    return issues


def check_films_look_like_tv(film_df):
    """Find films that have TV-like patterns in their titles"""
    issues = []

    for idx, row in film_df.iterrows():
        title = str(row.get('title', ''))

        for pattern in TV_PATTERNS:
            if re.search(pattern, title, re.IGNORECASE):
                issues.append({
                    'title': title,
                    'pattern_matched': pattern,
                    'issue': 'film_looks_like_tv'
                })
                break

    return issues


def check_tv_looks_like_film(tv_df):
    """Find TV entries that might actually be films"""
    issues = []

    for idx, row in tv_df.iterrows():
        title = str(row.get('title', ''))

        for pattern in FILM_FRANCHISE_PATTERNS:
            if re.match(pattern, title, re.IGNORECASE):
                # Check if it's actually a known TV adaptation
                is_known_tv = False
                for known in KNOWN_TITLE_NUMBER_SHOWS:
                    if known.lower() in title.lower():
                        is_known_tv = True
                        break

                if not is_known_tv:
                    issues.append({
                        'title': title,
                        'pattern_matched': pattern,
                        'issue': 'tv_looks_like_film'
                    })
                break

    return issues


def analyze_with_gpu(all_data):
    """GPU-accelerated analysis using cuDF"""
    print("\n[GPU] Converting to cuDF DataFrames...")

    results = {
        'summary': [],
        'tv_missing_seasons': [],
        'films_look_like_tv': [],
        'tv_looks_like_film': [],
    }

    for content_type, period, pdf in all_data:
        start = time.time()

        # Clean DataFrame for cuDF compatibility - convert mixed types to strings
        pdf_clean = pdf.copy()
        for col in pdf_clean.columns:
            # Convert object columns to string to avoid mixed type errors
            if pdf_clean[col].dtype == 'object':
                pdf_clean[col] = pdf_clean[col].astype(str).replace('nan', '')
                pdf_clean[col] = pdf_clean[col].replace('None', '')

        # Convert to cuDF for GPU acceleration
        try:
            gdf = cudf.DataFrame.from_pandas(pdf_clean)
        except Exception as e:
            print(f"  [WARNING] cuDF conversion failed for {period} {content_type}, using pandas: {e}")
            gdf = pdf  # Fall back to pandas

        if content_type == 'tv':
            # Count statistics using GPU
            total = len(gdf)

            # Season number analysis
            if 'season_number' in gdf.columns:
                with_season = int((~gdf['season_number'].isna()).sum())
                without_season = int(gdf['season_number'].isna().sum())
            else:
                with_season = 0
                without_season = total

            # Title type counts
            if 'title_type' in gdf.columns:
                limited_series = int((gdf['title_type'] == 'limited_series').sum())
            else:
                limited_series = 0

            results['summary'].append({
                'period': period,
                'type': 'tv_shows',
                'total': total,
                'with_season': with_season,
                'without_season': without_season,
                'limited_series': limited_series,
            })

            # Check for missing season extractions (CPU for regex)
            issues = check_tv_missing_seasons(pdf)
            for issue in issues:
                issue['period'] = period
            results['tv_missing_seasons'].extend(issues)

            # Check for TV that looks like film
            issues = check_tv_looks_like_film(pdf)
            for issue in issues:
                issue['period'] = period
            results['tv_looks_like_film'].extend(issues)

        else:  # film
            total = len(gdf)

            results['summary'].append({
                'period': period,
                'type': 'films',
                'total': total,
            })

            # Check for films that look like TV
            issues = check_films_look_like_tv(pdf)
            for issue in issues:
                issue['period'] = period
            results['films_look_like_tv'].extend(issues)

        elapsed = time.time() - start
        print(f"  [GPU] Processed {period} {content_type}: {total:,} rows in {elapsed:.3f}s")

    return results


def print_report(results):
    """Print comprehensive integrity report"""
    print("\n" + "=" * 80)
    print("                    INTEGRITY REPORT")
    print("=" * 80)

    # Summary table
    print("\n--- SUMMARY BY PERIOD ---")
    print("-" * 80)
    print(f"{'Period':<12} | {'Type':<10} | {'Total':<8} | {'w/Season':<10} | {'No Season':<10} | {'Limited':<8}")
    print("-" * 80)

    tv_total = 0
    tv_with_season = 0
    tv_without_season = 0
    film_total = 0

    for s in results['summary']:
        if s['type'] == 'tv_shows':
            tv_total += s['total']
            tv_with_season += s.get('with_season', 0)
            tv_without_season += s.get('without_season', 0)
            print(f"{s['period']:<12} | {s['type']:<10} | {s['total']:<8,} | {s.get('with_season', 0):<10,} | {s.get('without_season', 0):<10,} | {s.get('limited_series', 0):<8,}")
        else:
            film_total += s['total']
            print(f"{s['period']:<12} | {s['type']:<10} | {s['total']:<8,} | {'N/A':<10} | {'N/A':<10} | {'N/A':<8}")

    print("-" * 80)
    print(f"{'TOTAL TV':<12} | {'':<10} | {tv_total:<8,} | {tv_with_season:<10,} | {tv_without_season:<10,} |")
    print(f"{'TOTAL FILM':<12} | {'':<10} | {film_total:<8,} |")

    # Issue 1: TV shows missing season extraction
    print("\n" + "=" * 80)
    print("ISSUE 1: TV SHOWS MISSING SEASON EXTRACTION")
    print("         (Title ends with number but season_number is NULL)")
    print("=" * 80)

    tv_issues = results['tv_missing_seasons']
    print(f"\nTotal issues: {len(tv_issues)}")

    # Group by period
    by_period = {}
    for issue in tv_issues:
        p = issue['period']
        if p not in by_period:
            by_period[p] = []
        by_period[p].append(issue)

    for period in sorted(by_period.keys()):
        issues = by_period[period]
        print(f"\n  {period} ({len(issues)} issues):")
        for issue in issues[:10]:
            print(f"    '{issue['title']}' -> base: '{issue['potential_base']}', season: {issue['potential_season']}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")

    # Issue 2: Films that look like TV
    print("\n" + "=" * 80)
    print("ISSUE 2: FILMS THAT LOOK LIKE TV SHOWS")
    print("         (Film title contains Season/Series/Part patterns)")
    print("=" * 80)

    film_issues = results['films_look_like_tv']
    print(f"\nTotal issues: {len(film_issues)}")

    by_period = {}
    for issue in film_issues:
        p = issue['period']
        if p not in by_period:
            by_period[p] = []
        by_period[p].append(issue)

    for period in sorted(by_period.keys()):
        issues = by_period[period]
        print(f"\n  {period} ({len(issues)} issues):")
        for issue in issues[:10]:
            print(f"    '{issue['title']}' (pattern: {issue['pattern_matched']})")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")

    # Issue 3: TV that looks like film
    print("\n" + "=" * 80)
    print("ISSUE 3: TV SHOWS THAT MIGHT BE FILMS")
    print("         (TV title matches known film franchise pattern)")
    print("=" * 80)

    tv_film_issues = results['tv_looks_like_film']
    print(f"\nTotal issues: {len(tv_film_issues)}")

    by_period = {}
    for issue in tv_film_issues:
        p = issue['period']
        if p not in by_period:
            by_period[p] = []
        by_period[p].append(issue)

    for period in sorted(by_period.keys()):
        issues = by_period[period]
        print(f"\n  {period} ({len(issues)} issues):")
        for issue in issues[:10]:
            print(f"    '{issue['title']}' (pattern: {issue['pattern_matched']})")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")

    # Final assessment
    print("\n" + "=" * 80)
    print("                    INTEGRITY ASSESSMENT")
    print("=" * 80)

    tv_issue_count = len(tv_issues)
    film_issue_count = len(film_issues)

    tv_confidence = 100 - (100 * tv_issue_count / tv_total) if tv_total > 0 else 0
    film_confidence = 100 - (100 * film_issue_count / film_total) if film_total > 0 else 0

    print(f"""
    TOTALS:
      TV Shows: {tv_total:,}
        - With season number: {tv_with_season:,} ({100*tv_with_season/tv_total:.1f}%)
        - Without season number: {tv_without_season:,} ({100*tv_without_season/tv_total:.1f}%)

      Films: {film_total:,}

    POTENTIAL ISSUES:
      TV shows needing season extraction: {tv_issue_count:,} ({100*tv_issue_count/tv_total:.2f}% of TV)
      Films possibly misclassified as TV: {film_issue_count:,} ({100*film_issue_count/film_total:.2f}% of films)
      TV shows possibly misclassified films: {len(tv_film_issues):,}

    CLASSIFICATION CONFIDENCE:
      TV Shows: {tv_confidence:.1f}%
      Films: {film_confidence:.1f}%
    """)

    # GPU memory report
    try:
        mem_free, mem_total = cp.cuda.runtime.memGetInfo()
        print(f"    GPU Memory: {mem_free/1e9:.1f}GB free / {mem_total/1e9:.1f}GB total")
    except:
        pass

    print("=" * 80)


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
        print(f"  GPU initialization failed: {e}")
        return 1

    # Load data
    print("\n[1/3] Loading tidied files...")
    start = time.time()
    all_data = load_tidied_files()
    print(f"  Loaded in {time.time() - start:.2f}s")

    # Analyze
    print("\n[2/3] Analyzing classification integrity...")
    start = time.time()
    results = analyze_with_gpu(all_data)
    print(f"  Analysis completed in {time.time() - start:.2f}s")

    # Report
    print("\n[3/3] Generating report...")
    print_report(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
