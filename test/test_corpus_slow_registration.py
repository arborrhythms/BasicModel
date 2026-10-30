"""Cheap contracts for the deferred large-corpus pipeline and its guard."""
import pytest


def test_large_corpus_has_an_isolated_bounded_weekly_tier(tmp_path, monkeypatch):
    import slow_weekly
    nodes = ['test/ordinary.py::case', slow_weekly.NATIVE_FILE + '::case',
             slow_weekly.CORPUS_FILE + '::test_capacity_search_and_selected_corpus_prefix']
    tiers = slow_weekly.selected_tiers(nodes)
    assert [node for _, group, _, _ in tiers for node in group] == nodes
    name, selected, ceiling, timeout = tiers[-1]
    assert (name, ceiling, timeout) == ('corpus', 28, 172800)
    calls = []

    def run_suite(**kwargs):
        calls.append(kwargs)
        return dict(workers=[dict(selected=selected)], exit_code=0, reason='exit')

    monkeypatch.setattr(slow_weekly.bounded, 'run_suite', run_suite)
    slow_weekly.run_tier(tmp_path, selected, tmp_path / 'result', ceiling, timeout=timeout)
    assert calls[0]['workers'] == 1
    assert calls[0]['memory_bytes'] == calls[0]['worker_memory_bytes'] == 28 * 1024**3
    assert calls[0]['timeout'] == timeout
    assert calls[0]['suite_timeout'] > timeout


def test_only_native_admission_exhaustion_can_select_a_smaller_prefix():
    from corpus_large_probe import selected_document_count
    assert selected_document_count(dict(status='complete'), dict(status='complete'), 100) == 100
    error = dict(status='failed', error_type='RuntimeError',
                 error_message='definition store capacity exhausted; no word was admitted',
                 largest_completed_max_docs=17)
    assert selected_document_count(error, dict(status='failed'), 100) == 17
    for message in ('tensor capacity shape mismatch', 'identity map requires finite coordinates in [0, 1]'):
        with pytest.raises(RuntimeError, match='outside a native inventory limit'):
            selected_document_count(dict(error, error_message=message), dict(status='failed'), 100)
    with pytest.raises(RuntimeError, match='outside a native inventory limit'):
        selected_document_count(error, dict(status='memory_guard'), 100)
    with pytest.raises(RuntimeError, match='no nonempty whole-document prefix'):
        selected_document_count(dict(error, largest_completed_max_docs=0), dict(status='failed'), 100)
