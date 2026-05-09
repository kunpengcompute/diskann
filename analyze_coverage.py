#!/usr/bin/env python3
"""
Analyze coverage_fast_diskann_only.info to find uncovered FAST_DISKANN code.
"""

import sys
import re

def analyze_coverage(info_file):
    """
    Parse .info file and report uncovered lines by file.
    """
    current_file = None
    covered_lines = set()
    uncovered_lines = set()

    with open(info_file, 'r') as f:
        for line in f:
            if line.startswith('SF:'):
                # New file
                if current_file:
                    # Report previous file
                    total = len(covered_lines) + len(uncovered_lines)
                    if total > 0:
                        coverage_pct = 100.0 * len(covered_lines) / total
                        if coverage_pct < 90:
                            print(f"\n{current_file}")
                            print(f"  Coverage: {coverage_pct:.1f}% ({len(covered_lines)}/{total})")
                            print(f"  Uncovered lines: {len(uncovered_lines)}")
                            if len(uncovered_lines) <= 20:
                                print(f"  Lines: {sorted(uncovered_lines)}")

                current_file = line[3:].strip()
                covered_lines = set()
                uncovered_lines = set()
            elif line.startswith('DA:'):
                # Line coverage: DA:line_num,hit_count
                parts = line[3:].split(',')
                if len(parts) >= 2:
                    line_num = int(parts[0])
                    hit_count = int(parts[1])
                    if hit_count > 0:
                        covered_lines.add(line_num)
                    else:
                        uncovered_lines.add(line_num)

        # Report last file
        if current_file:
            total = len(covered_lines) + len(uncovered_lines)
            if total > 0:
                coverage_pct = 100.0 * len(covered_lines) / total
                if coverage_pct < 90:
                    print(f"\n{current_file}")
                    print(f"  Coverage: {coverage_pct:.1f}% ({len(covered_lines)}/{total})")
                    print(f"  Uncovered lines: {len(uncovered_lines)}")
                    if len(uncovered_lines) <= 20:
                        print(f"  Lines: {sorted(uncovered_lines)}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <coverage.info>")
        sys.exit(1)

    analyze_coverage(sys.argv[1])
