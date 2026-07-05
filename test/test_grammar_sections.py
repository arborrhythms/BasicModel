"""The grammar's canonical section vocabulary (single name per concept -- no
aliases / no namespace pollution):

  <PartSpace> { <Synthesize>, <Analyze> }  -- mereological parts<->whole
  <Symbolic>  { <compose>, <generate> }    -- the symbolic rules
  <Queries>                                -- top-level introspection ops

Every .grammar file uses these names; the parser reads ONLY these (the old
<WholeSpace> / PartSpace-<compose> section aliases and the normalization layer
were removed). Inline config <grammar> blocks remain the legacy flat
<compose>/<generate> form.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import Language as L


def _fresh(grammar_file):
    g = L.Grammar()
    g.configure(L.load_grammar(grammar_file))
    return g


def test_complete_grammar_canonical_sections():
    d = L.load_grammar('complete.grammar')
    assert 'Symbolic' in d and 'WholeSpace' not in d
    assert {'Synthesize', 'Analyze'} <= set(d['PartSpace'].keys())
    assert {'compose', 'generate'} <= set(d['Symbolic'].keys())
    assert 'Queries' in d


def test_all_shipped_grammars_load_with_rules():
    for g in ['complete.grammar', 'default.grammar',
              'shamatha.grammar', 'xor.grammar']:
        G = _fresh(g)
        assert len(G.rules_upward) > 0, g
        assert len(G.rules_downward) > 0, g
        # stop/boundary/uniform structural cover (>= 3); complete.grammar adds
        # the additive 'chunk' (the ex-"union" PartSpace sum).
        expected = 4 if g == 'complete.grammar' else 3
        assert len(G.ps_rules_upward) == expected, g
        assert len(G.ps_rules_downward) == expected, g


def test_query_ops_from_top_level_queries():
    G = _fresh('complete.grammar')
    # The truth family are now is-prefixed BOOLEAN-PREDICATE queries (isTrue /
    # isEqual / isPart / isWhole); the bare part/whole/equal are compositional
    # RELATIONS in <compose>, not queries. The non-is queries return a value
    # (parts/wholes retrieve, quantize/arma transform/predict, query dispatches).
    assert {'isTrue(X)', 'isEqual(X, Y)', 'isPart(X, Y)', 'isWhole(X, Y)',
            'query(X, Y)', 'quantize(X)', 'wholes(X)', 'parts(X)',
            'arma(X)'} == set(G.query_ops)
    assert not any('isomorph' in q for q in G.query_ops)


def test_symbolic_is_the_canonical_section_name():
    G = L.Grammar()
    G.configure({'Symbolic': {'compose': {'rule': ['S = not.forward(S)']}}})
    assert any(getattr(r, 'method_name', None) == 'not'
               for r in G.rules_upward)


def test_old_wholespace_section_no_longer_loads_symbolic_rules():
    # <WholeSpace> is no longer a recognized grammar section, so its rules are
    # not loaded as symbolic compose rules (the aliases were removed).
    G = L.Grammar()
    G.configure({'WholeSpace': {'compose': {'rule': ['S = not.forward(S)']}}})
    assert not any(getattr(r, 'method_name', None) == 'not'
                   for r in G.rules_upward)


def test_normalization_layer_removed():
    assert not hasattr(L, '_normalize_grammar_dict')


def test_legacy_flat_form_still_works():
    # Inline config <grammar> blocks use the flat top-level <compose>/<generate>.
    G = L.Grammar()
    G.configure({'compose': {'rule': ['S = not.forward(S)']}})
    assert len(G.rules_upward) == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
