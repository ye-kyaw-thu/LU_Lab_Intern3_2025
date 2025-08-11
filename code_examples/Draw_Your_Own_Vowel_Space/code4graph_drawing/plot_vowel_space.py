#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Burmese Vowel Space (F1 vs F2) using formant table
Written by Ye, LU Lab., Myanmar

- Supports stdin / --input CSV
- Labels vowels using IPA
- Rolling average (window=3) smoothing like the reference R script 
- Plots one smoothed path per vowel

Usage:
  python ./plot_vowel_space.py --input ye_formants_utf8.csv --output ./ye_vowel_space.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# IPA label map
IPA_MAP = {
    'i': '\u0069',
    'e': '\u0065',
    'ɛ': '\u025B',
    'a': '\u0061',
    'ɑ': '\u0251',
    'o': '\u006F',
    'u': '\u0075'
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=False)
    return parser.parse_args()

def main():
    args = parse_args()

    # Load file
    try:
        df = pd.read_csv(args.input, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(args.input, encoding='utf-16')

    # Filter to vowels we care about
    df = df[df['vowel'].isin(IPA_MAP)]
    df['IPA'] = df['vowel'].map(IPA_MAP)

    # --- Step 1: Aggregate by vowel + IPA + time_index (mean F1/F2 like in R) ---
    df_avg = df.groupby(['vowel', 'IPA', 'time_index'], as_index=False).agg({
        'F1': 'mean',
        'F2': 'mean',
        'F3': 'mean'
    })

    # --- Step 2: Apply rolling mean smoothing (window=3, like R's zoo::rollmean) ---
    df_avg['F1s'] = df_avg.groupby(['vowel', 'IPA'])['F1'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df_avg['F2s'] = df_avg.groupby(['vowel', 'IPA'])['F2'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    # --- Step 3: Endpoint data (time_index == 9) for labeling ---
    df_end = df_avg[df_avg['time_index'] == 9]

    # Slight manual offsets for overlapping vowel labels
    label_offsets = {
        'a': (-20, 25),    # move a slightly left-up
        'ɑ': (20, -20),    # move ɑ slightly right-down
    }

    # --- Step 4: Plotting ---
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title("Vowel Space of Ye Kyaw Thu (Male)")
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.invert_xaxis()
    ax.invert_yaxis()

    # One smooth line per vowel
    for vowel, group in df_avg.groupby('vowel'):
        ax.plot(group['F2s'], group['F1s'], label=vowel, linewidth=1.5)

    # One label per vowel at time_index=9
    # Draw IPA labels with optional nudges to avoid overlaps
    for _, row in df_end.iterrows():
        dx, dy = label_offsets.get(row['vowel'], (0, 0))  # default to no shift
        ax.text(row['F2s'] + dx, row['F1s'] + dy, row['IPA'],
                ha='center', va='center',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5))


    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.yaxis.set_major_locator(MultipleLocator(200))
    ax.grid(True)
    plt.tight_layout()

    # Save or show
    if args.output:
        plt.savefig(args.output, dpi=600)
    else:
        plt.show()

if __name__ == "__main__":
    main()

