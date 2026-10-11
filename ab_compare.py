"""
ab_compare.py — CLI tool to compare two evaluation run reports.

Usage:
    python ab_compare.py eval_history/run_a.json eval_history/run_b.json
    python ab_compare.py eval_history/run_a.json eval_history/run_b.json --name-a Baseline --name-b Candidate
"""

from reporter.ab_comparator import main

if __name__ == "__main__":
    main()
