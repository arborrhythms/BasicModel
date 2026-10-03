"""A gate receipt needs all four decoded rows, beyond the report's cache."""
import json
from pathlib import Path


def test_baseline_report_cache_is_not_a_four_row_reconstruction():
    path = Path(__file__).parent / 'before/candidate/measurements.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    grammar = [row for row in rows if row['kind'] == 'grammar']
    assert grammar
    assert all(len(row['reconstructions']) == 4 for row in grammar)
