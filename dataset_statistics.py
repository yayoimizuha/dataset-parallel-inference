#!/usr/bin/env python3
"""
Dataset Statistics Script for Hugging Face datasets

This script analyzes and outputs statistical information about the prompt lengths
in datasets. It automatically detects available subsets from the dataset metadata.

Usage:
    python dataset_statistics.py <dataset_name> [subset]
    
    If dataset_name is provided, analyzes all available subsets
    If subset is also provided, analyzes only that specific subset

Output includes:
- Count, mean, variance, standard deviation
- Percentile distribution (10%, 20%, ..., 90%, 100%)
- Min/max values
- Distribution analysis
"""

import json
import numpy as np
from datasets import load_dataset, get_dataset_config_names
from typing import Dict, List, Tuple, Optional
import sys
import argparse


def calculate_prompt_length(data: dict) -> int:
    """
    Calculate the length of input_json_str as done in task.py process() method.
    This replicates the exact logic used in the actual processing.
    """
    try:
        # "extra_info" は dict型
        input_json_str = json.dumps(data["extra_info"], ensure_ascii=False, separators=(",", ":"))
    except (KeyError, ValueError):
        # "extra_info" がデコードできない場合は "prompt" と "reward_model", "Rubrics" を取得
        try:
            input_json_str = json.dumps(
                {"prompt": data["prompt"], "reward_model": data["reward_model"], "rubrics": data["Rubrics"]}, 
                ensure_ascii=False, 
                separators=(",", ":")
            )
        except (KeyError, ValueError):
            # "Rubrics" がデコードできないか存在しない場合は "prompts" と "reward_model" を取得
            input_json_str = json.dumps(
                {"prompt": data["prompt"], "reward_model": data["reward_model"]}, 
                ensure_ascii=False, 
                separators=(",", ":")
            )
    
    return len(input_json_str)


def count_exceeding_threshold(lengths: List[int], threshold: int) -> Tuple[int, float]:
    """Count how many prompts exceed a given threshold."""
    count = sum(1 for length in lengths if length > threshold)
    percentage = (count / len(lengths)) * 100 if lengths else 0
    return count, percentage


def print_dataset_stats(config_name: str, lengths: List[int], stats: Dict):
    """Print comprehensive statistics for a dataset."""
    print(f"\n{'='*80}")
    print(f"Dataset: {config_name.upper()}")
    print(f"{'='*80}\n")
    
    # Basic statistics
    print("Basic Statistics:")
    print(f"  Total samples: {stats['count']:,}")
    print(f"  Mean length: {stats['mean']:,.2f} characters")
    print(f"  Median length: {stats['median']:,.2f} characters")
    print(f"  Variance: {stats['variance']:,.2f}")
    print(f"  Standard deviation: {stats['std_dev']:,.2f}")
    print(f"  Min length: {stats['min']:,} characters")
    print(f"  Max length: {stats['max']:,} characters")
    
    # Quartile information
    print(f"\nQuartile Information:")
    print(f"  Q1 (25th percentile): {stats['q1']:,} characters")
    print(f"  Q2 (50th percentile/Median): {stats['median']:,.0f} characters")
    print(f"  Q3 (75th percentile): {stats['q3']:,} characters")
    print(f"  IQR (Interquartile Range): {stats['iqr']:,} characters")
    
    # Percentile distribution
    print(f"\nPercentile Distribution:")
    for percentile, value in sorted(stats['percentiles'].items(), key=lambda x: int(x[0].rstrip('%'))):
        print(f"  {percentile:>4s}: {value:>10,} characters")
    
    # Threshold analysis
    print(f"\nThreshold Analysis:")
    thresholds = [2500, 5000, 10000, 15000, 20000, 30000]
    for threshold in thresholds:
        count, percentage = count_exceeding_threshold(lengths, threshold)
        print(f"  > {threshold:>6,} chars: {count:>6,} samples ({percentage:>6.2f}%)")
    
    # Distribution bins
    print(f"\nLength Distribution (bins):")
    bins = [0, 1000, 2500, 5000, 10000, 20000, 50000, float('inf')]
    bin_labels = ['0-1k', '1k-2.5k', '2.5k-5k', '5k-10k', '10k-20k', '20k-50k', '50k+']
    
    for i in range(len(bins) - 1):
        count = sum(1 for length in lengths if bins[i] < length <= bins[i+1])
        percentage = (count / len(lengths)) * 100 if lengths else 0
        print(f"  {bin_labels[i]:>10s}: {count:>6,} samples ({percentage:>6.2f}%)")


def print_summary_table(all_stats: Dict[str, Dict], all_lengths: Dict[str, List[int]], dataset_configs: List[str]):
    """Print a summary comparison table across all datasets."""
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    # Header
    print(f"{'Metric':<20}", end='')
    for config in dataset_configs:
        print(f"{config.upper():>12}", end='')
    print()
    print('-' * 80)
    
    # Metrics to compare
    metrics = [
        ('Count', 'count', ','),
        ('Mean', 'mean', ',.2f'),
        ('Median', 'median', ',.0f'),
        ('Std Dev', 'std_dev', ',.2f'),
        ('Min', 'min', ','),
        ('Max', 'max', ','),
        ('Q1', 'q1', ','),
        ('Q3', 'q3', ','),
        ('IQR', 'iqr', ','),
        ('90th %ile', 'percentiles', ','),
        ('95th %ile', 'percentiles', ','),
    ]
    
    for label, key, fmt in metrics:
        print(f"{label:<20}", end='')
        for config in dataset_configs:
            if key == 'percentiles':
                if label == '90th %ile':
                    value = all_stats[config]['percentiles']['90%']
                else:  # 95th percentile
                    value = np.percentile(all_lengths[config], 95)
                print(f"{value:>12{fmt}}", end='')
            else:
                value = all_stats[config][key]
                print(f"{value:>12{fmt}}", end='')
        print()
    
    # Add threshold comparison
    print('\n' + '-' * 80)
    print("Samples exceeding thresholds:")
    print('-' * 80)
    
    for threshold in [2500, 10000]:
        print(f"\n> {threshold:,} chars:")
        print(f"{'Dataset':<20}", end='')
        for config in dataset_configs:
            print(f"{config.upper():>12}", end='')
        print()
        
        print(f"{'Count':<20}", end='')
        for config in dataset_configs:
            lengths = all_lengths[config]
            count, _ = count_exceeding_threshold(lengths, threshold)
            print(f"{count:>12,}", end='')
        print()
        
        print(f"{'Percentage':<20}", end='')
        for config in dataset_configs:
            lengths = all_lengths[config]
            _, percentage = count_exceeding_threshold(lengths, threshold)
            print(f"{percentage:>11.2f}%", end='')
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze dataset statistics for Hugging Face datasets (automatically detects available subsets)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_statistics.py NovelHacja/RubricHub_v1_config           # Analyze all available subsets
  python dataset_statistics.py NovelHacja/RubricHub_v1_config chat      # Analyze only chat subset
  python dataset_statistics.py NovelHacja/RubricHub_v1_config writing   # Analyze only writing subset
        """
    )
    parser.add_argument('dataset_name', type=str, 
                        help='Dataset name (e.g., NovelHacja/RubricHub_v1_config)')
    parser.add_argument('subset', type=str, nargs='?', default=None,
                        help='Optional: specific subset to analyze. If not provided, all available subsets will be analyzed.')
    
    args = parser.parse_args()
    
    # Get available configs from the dataset metadata
    try:
        print(f"Retrieving available subsets for {args.dataset_name}...", file=sys.stderr)
        available_configs = get_dataset_config_names(args.dataset_name)
        print(f"Found {len(available_configs)} subsets: {', '.join(available_configs)}", file=sys.stderr)
    except Exception as e:
        print(f"Error retrieving dataset configs: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine which configs to analyze
    if args.subset:
        if args.subset not in available_configs:
            print(f"Error: Unknown subset '{args.subset}'", file=sys.stderr)
            print(f"Available subsets: {', '.join(available_configs)}", file=sys.stderr)
            sys.exit(1)
        dataset_configs = [args.subset]
    else:
        dataset_configs = available_configs
    
    all_stats = {}
    all_lengths = {}
    
    print("="*80, file=sys.stderr)
    print(f"Dataset Statistics Analysis for {args.dataset_name}", file=sys.stderr)
    if args.subset:
        print(f"Subset: {args.subset}", file=sys.stderr)
    else:
        print(f"Analyzing all subsets: {', '.join(dataset_configs)}", file=sys.stderr)
    print("="*80, file=sys.stderr)
    
    # Analyze each dataset
    for config in dataset_configs:
        try:
            print(f"Loading dataset: {config}...", file=sys.stderr)
            dataset = load_dataset(args.dataset_name, config, split="train", streaming=False)
            
            lengths = []
            for data in dataset:
                length = calculate_prompt_length(data)
                lengths.append(length)
            
            lengths_array = np.array(lengths)
            
            # Calculate statistics
            stats = {
                'count': len(lengths),
                'mean': float(np.mean(lengths_array)),
                'variance': float(np.var(lengths_array)),
                'std_dev': float(np.std(lengths_array)),
                'min': int(np.min(lengths_array)),
                'max': int(np.max(lengths_array)),
                'median': float(np.median(lengths_array)),
                'percentiles': {}
            }
            
            # Calculate percentiles
            percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            for p in percentiles:
                stats['percentiles'][f'{p}%'] = int(np.percentile(lengths_array, p))
            
            # Additional quartile information
            stats['q1'] = int(np.percentile(lengths_array, 25))
            stats['q3'] = int(np.percentile(lengths_array, 75))
            stats['iqr'] = stats['q3'] - stats['q1']
            
            all_lengths[config] = lengths
            all_stats[config] = stats
            print_dataset_stats(config, lengths, stats)
        except Exception as e:
            print(f"\nError analyzing {config}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            continue
    
    # Print summary comparison (only if analyzing multiple datasets)
    if all_stats and len(dataset_configs) > 1:
        print_summary_table(all_stats, all_lengths, dataset_configs)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")
