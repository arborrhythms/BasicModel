"""Direct certificates for failures exposed by the end-of-item measurements."""
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
import torch
from nomination_fixtures import recurrent_word


@pytest.mark.parametrize('source', ['members', 'occurrences', 'candidate'])
def test_memory_centroids_preserve_common_rails_with_fractional_evidence(source):
    from MemoryConsolidation import _fold
    from FamiliarTier import centroid
    weights = torch.tensor([3., 2.999999999301692, 2., 1., .9999999997671821,
                            .9999999997671778, .9999999997671775], dtype=torch.float64)
    codes = torch.cat((torch.ones(7, 176, dtype=torch.float64),
                       torch.zeros(7, 176, dtype=torch.float64)), dim=1)
    bank = NS(W=codes, lookup_rows=lambda rows: codes[rows])
    owner = NS(similarity_codebook=bank, _row_order=lambda row: 0)
    members = dict(enumerate(weights.tolist()))
    if source == 'candidate':
        result = centroid(owner, members)
    else:
        store = NS(slots=codes, trust=weights)
        result = _fold(owner, store, list(members), members=members if source == 'members' else None)
    assert torch.equal(result, codes[0])
    assert ((result >= 0) & (result <= 1)).all()


@pytest.mark.parametrize('flag', ['sign_leaf', 'no_op_leaf'])
@pytest.mark.parametrize('mode', ['mint', 'bind'])
def test_surface_only_introduction_uses_its_written_order_for_later_aliases(flag, mode):
    from test_reference_close import fixture, known_concept
    from test_medium_term_memory import mention
    from chunk_fixtures import context
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    word = recurrent_word(owner.interpret, [120], (), form='surface-only')
    concept = known_concept(owner, atoms[0], order=1)
    owner.interpret.define(word, concept)
    identity = dict(handle=1, origins=torch.zeros(1, 3, dtype=torch.long))
    point = torch.zeros_like(atoms[0])
    first = Clause(ConceptualMeaning.from_description(point), point=point, order=0,
        subject_word_id=word, composition=Composition(point, order=0, word_id=word),
        identity=identity, reference_request=dict(mode=mode, leaves=(0,),
            **{flag: True}, addressed_introduction=True,
            **({} if mode=='mint' else {'compose_bind':0})))
    alias = mention(atoms[0], mode='bind')
    alias = replace(alias, identity=identity,
        reference_request=dict(alias.reference_request, compose_bind=1))
    root = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0],
        children=(first,alias), refs=(('clause',0),-1,('clause',1)))
    plan = resolver.preview(root, context(resolver, ()))
    assert owner._concept_source_order(owner.interpret.referent(word)) == 1
    assert [item['selected'] for item in plan['items']] == [('new',0),('new',0)]
    assert [item['member_order'] for item in plan['items']] == [0,0]
    kept=resolver.prepare(root,plan)
    written={}
    row=store.write_clause(kept,written_rows=written)
    resolver.commit(plan,row,written,document='surface-only',position=0,training=False)
    references=tuple(plan['mention_occurrences'].values())
    assert len(references)==2
    key=owner.mtm.token_of(references[0])
    assert owner.mtm.token_of(references[1])==key
    assert owner.mtm.entries[key].refinement['member_order']==0
    assert all(store.order_of_row(store.location(ref))==0 for ref in references)
