"""Fixed compiled banks and a trial/commit reference-order certificate."""
import torch


def test_distinct_word_counts_share_one_fixed_capacity_compiled_graph():
    from FormCentroids import BasicCandidateWholes
    from test_item20_round5_percepts import percepts
    basic, _, _ = percepts()
    a = basic.reserve((0,), (0,))
    b = basic.reserve((1,), (1,))
    counts = torch.nn.functional.one_hot(torch.full((2, 4), 97), 256).float()
    forms = torch.eye(4)[None].expand(2, -1, -1)
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    read = torch.compile(lambda bank: bank.read_presence(bank.forms),
                         backend=backend, fullgraph=True, dynamic=False)
    with torch._dynamo.config.patch(error_on_recompile=True):
        for rows in (torch.tensor([[a, a, a, -1], [a, a, a, -1]]),
                     torch.tensor([[a, b, a, -1], [b, a, b, -1]])):
            weights, codes = basic.word_weights(counts, rows)
            assert weights.shape == (2, 4) and codes.shape == (2, 4, basic.width)
            assert not weights[:, 3].any() and not codes[:, 3].any()
            bank = BasicCandidateWholes(forms, weights, rows >= 0, codes)
            actual = read(bank)
            expected = bank.read_presence(forms)
            for value, wanted in zip(actual, expected):
                torch.testing.assert_close(value, wanted)
            torch.testing.assert_close(bank.for_stream(1).read_presence(forms[1:]),
                                       tuple(value[1:] for value in expected))
    assert len(graphs) == 1


def test_unnamed_order_one_site_binds_the_trial_selected_order_zero_path(monkeypatch):
    from dataclasses import replace
    from MemoryLink import Occurrence
    from test_reference_close import fixture, known_concept
    from test_medium_term_memory import mention
    from ReferenceContext import SituationFrame
    from chunk_fixtures import context
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    index = store.append_idea(atoms[0], order=0, when=False)
    address = store.occurrence_of(index)
    path = owner.link(Occurrence(address, order=0, selected=None), {},
                      value=atoms[0], where=None, when=0, stream=None)
    assert owner.mtm.in_window(path, None)
    def selected(choices, logits, **options):
        assert path in choices
        return choices.index(path)
    monkeypatch.setattr(resolver, 'select', selected)
    site = mention(atoms[0], mode='bind')
    plan = resolver.preview(site, context(resolver,
        (SituationFrame.from_row(store.row(index)),)))
    higher = known_concept(owner, atoms[0], order=1)
    # The use's own idea carries a different native order. Its unnamed
    # reference decision reads the selected path, not that containing idea.
    row = store.append_meaning(replace(site.meaning, bindings={'_identity': ('sym', higher)}))
    use = store.occurrence_of(row)
    assert store.order_of_row(row) == 1
    assert plan['items'][0]['member_order'] == 0
    resolver._bind_with_capture(path, use, stream=None, document='certificate',
        position=1, kind=None, value=atoms[0], when=1, order=plan['items'][0]['member_order'])
    assert owner.mtm.token_of(use) == path
    assert owner.mtm.entries[path].refinement['member_order'] == 0
