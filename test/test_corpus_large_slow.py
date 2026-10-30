"""The large-file item 2.7 capacity search and prefix run, deferred to November.

One pipeline shares its measured size without relying on another test's order.
The central RUN_SLOW gate skips it before model construction in ordinary runs.
"""
from pathlib import Path
import os

import pytest


@pytest.mark.slow
def test_capacity_search_and_selected_corpus_prefix(tmp_path):
    from corpus_large_probe import run_pipeline
    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ.get('CORPUS_LARGE_OUTPUT', str(tmp_path / 'large-corpus')))
    report = run_pipeline(root, output)
    assert report['status'] == 'complete', report
