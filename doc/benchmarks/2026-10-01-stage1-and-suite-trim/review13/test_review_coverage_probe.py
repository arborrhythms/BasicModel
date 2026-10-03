"""Saved review probes for omitted coverage and orphaned prototype interfaces."""
import inspect
from pathlib import Path


def test_retired_guard_covers_both_grammars():
    import test_retired_names as guards
    source = Path(guards.__file__).read_text()
    assert 'complete.grammar' in source
    assert 'transitional_pos.grammar' in source


def test_orphaned_prototype_interfaces_are_retired():
    import Language, Layers, space_carrier
    assert not hasattr(Language, 'IdeaSubSpace')
    for name in ('emit', 'bind_marker', 'canonical_marker', 'bound_markers'):
        assert not hasattr(Layers.GrammarLayer, name)
    for name in ('mark_codebook_parameters_changed', 'mark_codebook_structure_changed'):
        assert not hasattr(space_carrier.SpaceCarrierMixin, name)


def test_compiled_k2_case_keeps_inductor():
    from test_compiled_word_chunk import test_real_aligned_loop_matches_prior_compiled_semantics_across_chunks
    source = inspect.getsource(test_real_aligned_loop_matches_prior_compiled_semantics_across_chunks)
    assert '"TheCompileBackend", "inductor"' in source


def test_shared_ladder_minimum_count_is_restored():
    from test_meronomy_ladder import test_utility_counts_accrue_once_per_presentation
    source = inspect.getsource(test_utility_counts_accrue_once_per_presentation)
    assert 'cs.utility_min_count =' not in source
    assert 'monkeypatch.setattr(cs, "utility_min_count"' in source


def test_unused_mixing_bank_does_not_stage(monkeypatch):
    from types import SimpleNamespace
    from Models import BasicModel
    touched = []
    class Input:
        def __getattr__(self, name):
            touched.append(name)
            return None
    model = SimpleNamespace(serial=True, reconstruct_in_loop=False,
        inputSpace=Input(), _aligned_serial_word_mode=lambda: False)
    BasicModel._stage_mixing_reconstruction_bank(model)
    assert touched == []
