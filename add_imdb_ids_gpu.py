#!/usr/bin/env python3
"""
Add IMDB IDs to H2 2024 v2.30 files
GPU-accelerated with RAPIDS cuDF
"""

import cudf
import cupy as cp
import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path

print('='*100)
print('ADDING IMDB IDs TO H2 2024 V2.30 FILES (GPU ACCELERATED)')
print('='*100)

# Check GPU
props = cp.cuda.runtime.getDeviceProperties(0)
print(f'GPU: {props["name"].decode()}')
mem_free, mem_total = cp.cuda.runtime.memGetInfo()
print(f'VRAM: {mem_free/1e9:.1f}GB free / {mem_total/1e9:.1f}GB total')
print()

def normalize_title(title):
    """Normalize title for matching"""
    if pd.isna(title) or title is None:
        return ''
    t = str(title).lower().strip()
    t = re.sub(r'\s*\(.*?\)\s*$', '', t)
    t = re.sub(r'\s*:\s*season\s*\d+', '', t)
    t = re.sub(r'\s*:\s*part\s*\d+', '', t)
    t = re.sub(r'\s*:\s*limited series', '', t)
    t = re.sub(r'\s*:\s*miniseries', '', t)
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# ============================================================================
# LOAD BEST SOURCE WITH cuDF
# ============================================================================
print('### Loading Primary Source with cuDF ###')
print('File: IMDB Seasons Allocated Correct - Translated.csv')

# Use cuDF for fast CSV loading
imdb_source_path = '/mnt/c/Users/RoyT6/Downloads/Schema Engine/IMDB Seasons Allocated Correct - Translated.csv'
gdf = cudf.read_csv(imdb_source_path)
print(f'Loaded {len(gdf):,} rows on GPU')

# Transfer to pandas for dict building (small enough)
imdb_source = gdf.to_pandas()
print(f'Unique titles: {imdb_source["title"].nunique():,}')

# Build lookups
tv_lookup = {}
film_lookup = {}

print('\nBuilding lookup dictionaries...')
for _, row in imdb_source.iterrows():
    title_norm = normalize_title(row['title'])
    original_norm = normalize_title(row.get('original_title', ''))
    imdb_id = str(row['imdb_id']).strip() if pd.notna(row['imdb_id']) else None
    fc_uid = str(row['fc_uid']).strip() if pd.notna(row['fc_uid']) else None
    title_type = str(row.get('title_type', '')).lower()
    season = row.get('season_number')

    if not imdb_id or not imdb_id.startswith('tt'):
        continue

    if 'movie' in title_type or 'film' in title_type:
        if title_norm and title_norm not in film_lookup:
            film_lookup[title_norm] = (imdb_id, fc_uid or imdb_id)
        if original_norm and original_norm not in film_lookup:
            film_lookup[original_norm] = (imdb_id, fc_uid or imdb_id)
    else:
        if title_norm:
            if title_norm not in tv_lookup:
                tv_lookup[title_norm] = (imdb_id, fc_uid or imdb_id)
            if pd.notna(season):
                key = f"{title_norm}_s{int(season)}"
                if key not in tv_lookup:
                    tv_lookup[key] = (imdb_id, fc_uid)
        if original_norm:
            if original_norm not in tv_lookup:
                tv_lookup[original_norm] = (imdb_id, fc_uid or imdb_id)
            if pd.notna(season):
                key = f"{original_norm}_s{int(season)}"
                if key not in tv_lookup:
                    tv_lookup[key] = (imdb_id, fc_uid)

print(f'  TV: {len(tv_lookup):,} entries')
print(f'  Films: {len(film_lookup):,} entries')

# Free GPU memory
del gdf
cp.get_default_memory_pool().free_all_blocks()

# ============================================================================
# LOAD EXISTING V2.30 FILES FOR ADDITIONAL IDs
# ============================================================================
print('\n### Loading Secondary Source (Existing v2.30 files) ###')

existing_files = [
    ('/mnt/c/Users/RoyT6/Downloads/Training Data/netflix totals/Best Set/2.30/netflix_films_h1_2023_v2.30.xlsx', 'film'),
    ('/mnt/c/Users/RoyT6/Downloads/Training Data/netflix totals/Best Set/2.30/netflix_films_h2_2023_v2.30.xlsx', 'film'),
    ('/mnt/c/Users/RoyT6/Downloads/Training Data/netflix totals/Best Set/2.30/netflix_films_h1_2024_v2.30.xlsx', 'film'),
    ('/mnt/c/Users/RoyT6/Downloads/Training Data/netflix totals/Best Set/2.30/netflix_films_h1_2025_v2.30.xlsx', 'film'),
    ('/mnt/c/Users/RoyT6/Downloads/Training Data/netflix totals/Best Set/2.30/netflix_films_h2_2025_v2.30.xlsx', 'film'),
    ('/mnt/c/Users/RoyT6/Downloads/Training Data/netflix totals/Best Set/2.30/netflix_tv_h1_2023_v2.30.xlsx', 'tv'),
    ('/mnt/c/Users/RoyT6/Downloads/Training Data/netflix totals/Best Set/2.30/netflix_tv_h2_2023_v2.30.xlsx', 'tv'),
    ('/mnt/c/Users/RoyT6/Downloads/Training Data/netflix totals/Best Set/2.30/netflix_tv_h1_2024_v2.30.xlsx', 'tv'),
    ('/mnt/c/Users/RoyT6/Downloads/Training Data/netflix totals/Best Set/2.30/netflix_tv_h1_2025_v2.30.xlsx', 'tv'),
    ('/mnt/c/Users/RoyT6/Downloads/Training Data/netflix totals/Best Set/2.30/netflix_tv_h2_2025_v2.30.xlsx', 'tv'),
]

added_film = 0
added_tv = 0

for path, content_type in existing_files:
    try:
        df = pd.read_excel(path)
        for _, row in df[df['imdb_id'].notna()].iterrows():
            title_norm = normalize_title(row['title'])
            imdb_id = str(row['imdb_id']).strip()
            fc_uid = str(row.get('fc_uid', imdb_id)).strip() if pd.notna(row.get('fc_uid')) else imdb_id

            if content_type == 'film':
                if title_norm and title_norm not in film_lookup:
                    film_lookup[title_norm] = (imdb_id, fc_uid)
                    added_film += 1
            else:
                season = row.get('season_number')
                if title_norm:
                    if title_norm not in tv_lookup:
                        tv_lookup[title_norm] = (imdb_id, fc_uid)
                        added_tv += 1
                    if pd.notna(season):
                        key = f"{title_norm}_s{int(season)}"
                        if key not in tv_lookup:
                            tv_lookup[key] = (imdb_id, fc_uid)
    except Exception as e:
        print(f'  Error with {path}: {e}')

print(f'Added from existing: {added_film} films, {added_tv} TV')
print(f'\nFinal lookup size:')
print(f'  TV: {len(tv_lookup):,} entries')
print(f'  Films: {len(film_lookup):,} entries')

# ============================================================================
# MATCH IDs TO H2 2024 FILES
# ============================================================================

def match_id(title, season, lookup_dict, is_tv=False):
    """Find IMDB ID and fc_uid for a title"""
    title_norm = normalize_title(title)

    if is_tv and pd.notna(season):
        key = f"{title_norm}_s{int(season)}"
        if key in lookup_dict:
            return lookup_dict[key]

    if title_norm in lookup_dict:
        return lookup_dict[title_norm]

    return (None, None)

# Process Films
print('\n' + '='*100)
print('MATCHING IDs TO H2 2024 FILMS')
print('='*100)

film_path = '/mnt/c/Users/RoyT6/Downloads/Training Data/netflix totals/Best Set/2.30/netflix_films_h2_2024_v2.30.xlsx'
df_film = pd.read_excel(film_path)
df_film['imdb_id'] = df_film['imdb_id'].astype(object)
df_film['fc_uid'] = df_film['fc_uid'].astype(object)

matched = 0
for idx, row in df_film.iterrows():
    imdb_id, fc_uid = match_id(row['title'], None, film_lookup, is_tv=False)
    if imdb_id:
        df_film.at[idx, 'imdb_id'] = imdb_id
        df_film.at[idx, 'fc_uid'] = fc_uid or imdb_id
        matched += 1

print(f'Matched: {matched}/{len(df_film)} ({100*matched/len(df_film):.1f}%)')
df_film.to_excel(film_path, index=False, engine='openpyxl')
print(f'Saved: {film_path}')

# Process TV
print('\n' + '='*100)
print('MATCHING IDs TO H2 2024 TV')
print('='*100)

tv_path = '/mnt/c/Users/RoyT6/Downloads/Training Data/netflix totals/Best Set/2.30/netflix_tv_h2_2024_v2.30.xlsx'
df_tv = pd.read_excel(tv_path)
df_tv['imdb_id'] = df_tv['imdb_id'].astype(object)
df_tv['fc_uid'] = df_tv['fc_uid'].astype(object)

matched = 0
for idx, row in df_tv.iterrows():
    imdb_id, fc_uid = match_id(row['title'], row.get('season_number'), tv_lookup, is_tv=True)
    if imdb_id:
        df_tv.at[idx, 'imdb_id'] = imdb_id
        season = row.get('season_number')
        if fc_uid:
            df_tv.at[idx, 'fc_uid'] = fc_uid
        elif pd.notna(season):
            df_tv.at[idx, 'fc_uid'] = f"{imdb_id}_s{int(season):02d}"
        else:
            df_tv.at[idx, 'fc_uid'] = imdb_id
        matched += 1

print(f'Matched: {matched}/{len(df_tv)} ({100*matched/len(df_tv):.1f}%)')
df_tv.to_excel(tv_path, index=False, engine='openpyxl')
print(f'Saved: {tv_path}')

# ============================================================================
# SUMMARY
# ============================================================================
print('\n' + '='*100)
print('IMDB ID MATCHING COMPLETE')
print('='*100)

df_film_check = pd.read_excel(film_path)
df_tv_check = pd.read_excel(tv_path)

print(f'\nFinal Coverage:')
print(f'  Films H2 2024: {df_film_check["imdb_id"].notna().sum()}/{len(df_film_check)} ({100*df_film_check["imdb_id"].notna().mean():.1f}%)')
print(f'  TV H2 2024: {df_tv_check["imdb_id"].notna().sum()}/{len(df_tv_check)} ({100*df_tv_check["imdb_id"].notna().mean():.1f}%)')

print(f'\nfc_uid Coverage:')
print(f'  Films: {df_film_check["fc_uid"].notna().sum()}/{len(df_film_check)} ({100*df_film_check["fc_uid"].notna().mean():.1f}%)')
print(f'  TV: {df_tv_check["fc_uid"].notna().sum()}/{len(df_tv_check)} ({100*df_tv_check["fc_uid"].notna().mean():.1f}%)')

print(f'\n### Sample Films ###')
for _, row in df_film_check[df_film_check['imdb_id'].notna()].head(5).iterrows():
    print(f'  {row["title"][:40]:<40} | {row["imdb_id"]} | {row["fc_uid"]}')

print(f'\n### Sample TV ###')
for _, row in df_tv_check[df_tv_check['imdb_id'].notna()].head(5).iterrows():
    s = f"S{int(row['season_number'])}" if pd.notna(row['season_number']) else ''
    print(f'  {row["title"][:35]} {s:<5} | {row["imdb_id"]} | {row["fc_uid"]}')

# GPU memory status
mem_free, mem_total = cp.cuda.runtime.memGetInfo()
print(f'\nGPU Memory: {mem_free/1e9:.1f}GB free / {mem_total/1e9:.1f}GB total')
