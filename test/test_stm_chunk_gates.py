"""Step 3's field, lifetime and recall certificates, with explicit vectors."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import STMChunks as chunks
from PrependedSymbols import SymbolEntries, stage, reconcile
from Meaning import ConceptualMeaning
from QueryWork import QueryWorkBudget


def test_prepend_preserves_content_without_a_separate_meaning_chart():
    from MediumTermMemory import MediumTermMemory
    from PrependedSymbols import symbolize
    pool = MediumTermMemory(2,n_vectors=90)
    key = ('ltm','fixture',1)
    value = torch.arange(1,17,dtype=torch.float32)/17
    pool.open(key,kind=None,when=1,value=value,pending=False)
    owner = SimpleNamespace(mtm=pool,similarity_codebook=SimpleNamespace(
        mereology=SimpleNamespace(meaning_pairs=0,percept_width=8,percept_event_width=16)))
    store = SimpleNamespace(location=lambda _:0,slots=torch.zeros(1,16))
    actual = symbolize(owner,store,key,pool.entries[key])
    torch.testing.assert_close(actual,value,rtol=0,atol=0)
    assert actual.data_ptr() != value.data_ptr()


@pytest.mark.parametrize('order',(1,2))
def test_bind_menu_at_each_order_and_kept_mention_commit(monkeypatch,order):
    from test_clause_acceptance import SentenceFixture
    from chunk_fixtures import prepend, record
    from ReferenceContext import prepare_operands
    from ClauseJournal import finish_clause
    f = SentenceFixture(monkeypatch)
    keys = []
    for i in (0,1):
        row = f.store.append_meaning(ConceptualMeaning.from_description(torch.eye(8)[i]),
            sentence_index=i,document_key='antecedents',order=1)
        key = f.store.occurrence_of(row)
        f.cs.mtm.open(key,kind=None,when=i,value=f.store.slots[row],pending=False)
        keys.append(key)
    state = prepend(f,(),1)
    mention = torch.eye(8)[2].reshape(1,1,8)
    state,_ = chunks.deposit(state,mention[:,0],torch.tensor([True]),stack=True,order=order)
    age = state[chunks.READ].clone()
    bind = next(rule for rule in f.binary if rule.method_name == 'bind')
    menu = prepare_operands(mention.expand(1,2,8),torch.full((1,2),-1),torch.zeros(1,2,dtype=torch.long),
        torch.tensor([[0,0]]),orders=torch.full((1,2),order),rules=(bind,),unary_rules=(),
        active=torch.tensor([[True,False]]),chunk_state=state,bank=None,live=None)
    valid = menu['binary_valid']
    assert int(valid.sum()) == 3
    ids = menu['binary_refs'][...,1][valid].tolist()
    assert set(ids) == {0,*(f.cs.mtm.entries[key].transient for key in keys)}
    torch.testing.assert_close(age,state[chunks.READ],rtol=0,atol=0)
    entry,_,_ = record(f,('bind','subject'),leaf_orders=[order])
    selected = int(entry.identity_actions[-1,4])-1
    chosen = ('ltm',f.store.address_domain,int(entry.identity_origins[selected,1]))
    clause = finish_clause(f.language,entry,registry=f.registry)
    resolver = f.cs.components.reference_close
    plan = resolver.preview(clause,chunks.empty(1,8,8))
    prepared = resolver.prepare(clause,plan)
    written = {}
    row = f.store.write_clause(prepared,written_rows=written,document_key='mention',sentence_index=0)
    before = f.cs.mtm.entries[chosen].count
    resolver.commit(plan,row,written,document='mention',position=0,training=False)
    assert len(plan['items']) == 1
    item = plan['items'][0]
    occurrence = written.get(('occurrence',id(item['written_node'])))
    assert occurrence is not None and f.cs.mtm.token_of(occurrence) == chosen
    assert f.cs.mtm.entries[chosen].count == before+1
    assert item['written_node'].order == order


def test_chain_query_reenters_shared_chunks_with_stored_stamp_and_identities():
    from test_layout_removal import pending_chain
    from Layers import ShortTermMemory
    from Queries import ThoughtLTMCapability
    store,rows,_,_,_,chain = pending_chain()
    owner = store._concept_owner()
    owner.stm = ShortTermMemory(batch=1,capacity=8,concept_dim=16)
    for key in tuple(owner.mtm.entries):owner.mtm.forget(key)
    for number in range(1,9):
        state,_ = chunks.deposit(owner.stm.state,torch.full((1,16),number/10),torch.tensor([True]))
        owner.stm.set_state(state)
    old = owner.stm.state[chunks.READ].clone()
    reader = ThoughtLTMCapability(store=lambda:store,equal=lambda *_:0.,tau_id=.9)
    when = store.entry_when(chain)
    result = reader.query(when=when,max_records=32,work=QueryWorkBudget(64))
    assert result['frames'] and not result['incomplete']
    state = owner.stm.state
    assert state[chunks.REFS][0,0,0].tolist() == [chunks.RELATION,chain[1]]
    torch.testing.assert_close(state[chunks.WHEN][0,0],when,rtol=0,atol=0)
    torch.testing.assert_close(state[0][0,0,store.when_start:store.when_start+4],when,rtol=0,atol=0)
    assert state[chunks.READ][0,0] > old.max()
    assert all(owner.mtm.token_of(store.occurrence_of(row)) is not None for row in rows)
    state,index = chunks.deposit(state,torch.full((1,16),.9),torch.tensor([True]),stack=True)
    assert index.tolist() == [1]


def test_expired_and_promoted_transient_ids_cannot_alias_resident_chunks():
    from MediumTermMemory import MediumTermMemory
    pool = MediumTermMemory(2,n_vectors=90)
    store = SimpleNamespace(address_domain='fixture')
    first,second = ('ltm','fixture',1),('ltm','fixture',2)
    pool.open(first,kind=None,when=1,value=torch.ones(4),pending=False)
    state,_ = chunks.deposit(chunks.empty(1,8,4),torch.ones(1,4),torch.tensor([True]),identity=90,
        refs=torch.tensor([[[chunks.IDENTITY_REF,90],[chunks.OCCURRENCE,1]]]))
    pool.forget(first)
    pool.open(second,kind=None,when=2,value=torch.full((4,),.2),pending=False)
    assert pool.entries[second].transient == 90
    clean = reconcile(state,pool,store)
    assert clean[chunks.IDENTITY][0,0] == -1
    assert chunks.bind_menu(clean,torch.ones(1,1,4))[-1].sum() == 1
    state,_ = chunks.deposit(clean,torch.full((1,4),.2),torch.tensor([True]),identity=90,
        refs=torch.tensor([[[chunks.IDENTITY_REF,90],[chunks.OCCURRENCE,2]]]))
    pool.promote(second,7)
    promoted = reconcile(state,pool,store)
    selected = promoted[chunks.IDENTITY] == 7
    assert selected.sum() == 1
    assert promoted[chunks.REFS][selected][0,0].tolist() == [chunks.SYMBOL,7]


def test_unpublished_timeline_cannot_survive_as_a_recent_identity(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    row = f.store.append_meaning(ConceptualMeaning.from_description(torch.eye(8)[0]),order=1)
    key = f.store.occurrence_of(row)
    f.cs.mtm.open(key,kind=None,when=1,value=f.store.slots[row],pending=False)
    identity = f.cs.mtm.entries[key].transient
    state = chunks.empty(1,8,8)
    refs = torch.tensor([[[chunks.IDENTITY_REF,identity],[chunks.OCCURRENCE,key[-1]]]])
    for stack in (False,True):
        state,_ = chunks.deposit(state,torch.ones(1,8),torch.tensor([True]),
            identity=identity,timeline=1,stack=stack,refs=refs)
    result = f.cs.components.reference_close.committed_chunks(state,dict(items=()),{},stream=0)
    assert result[chunks.IDENTITY].eq(-1).all()
    assert result[chunks.TIMELINE].eq(0).all()
    assert result[chunks.READ][0,0] == 0
    assert result[1].item() == 1 and result[0][0,1].eq(1).all()


def test_prepend_transients_do_not_prime_or_place_and_promoted_symbol_registers(monkeypatch):
    from test_placement_surface import lesson
    from ClauseRow import attach_clause_index
    from Layers import LTM0
    from Models import BasicModel
    import weakref
    cs,derived,rows,reading = lesson(monkeypatch)
    store = LTM0(cs.similarity_codebook.W.shape[-1],capacity=32,percept_width=derived.percept_event_width)
    object.__setattr__(cs,'_clause_store_ref',weakref.ref(store))
    attach_clause_index(SimpleNamespace(),store,cs)
    entries = SymbolEntries(1,2,8,store.nDim)
    for name in ('concept_ids','object_ids'):
        entries.stage_words(name,torch.tensor([[cs.concept_id_at_row(rows[0])]]))
    for name in ('concept_rows','object_rows','lookup_rows'):
        entries.stage_words(name,torch.tensor([[rows[0]]]))
    entries.stage_words('lookup_atoms',cs.similarity_codebook.W[rows[:1]][None])
    entries.stage_words('lookup_sentence_ids',torch.zeros(1,1,dtype=torch.long))
    isp = SimpleNamespace(symbol_entries=entries,_packed_sentence_slot_mask=torch.ones(1,1,dtype=torch.bool))
    class Model:
        _prime_sentence_symbols = BasicModel._prime_sentence_symbols
        reconstruction_basis_limit = 2
        _open_sentence_slot = 0
        inputSpace = isp
        perceptualSpace = None
        languageSpace = SimpleNamespace(_compose_binary_rules=())
        wholeSpace = None
        symbolSpace = SimpleNamespace(ltm_store=store)
        def _concept_owner(self):return cs
    model = Model()
    row = store.append_meaning(ConceptualMeaning.from_description(cs.similarity_codebook.W[rows[0]].detach()),order=1)
    key = store.occurrence_of(row)
    cs.mtm.open(key,kind=None,when=1,value=store.slots[row],pending=False)
    state = stage(model,chunks.empty(1,8,store.nDim),torch.tensor([True]))
    identity = cs.mtm.entries[key].transient
    assert cs._csw_row_of(identity) is None
    assert entries.concept_ids[0,0] == identity and entries.concept_rows[0,0] == -1
    witnesses = derived.placement_witnesses.clone()
    isp._ar_concept_lookup_rows = entries.read('lookup_rows')
    isp._ar_concept_lookup_sentence_ids = entries.read('lookup_sentence_ids')
    model._prime_sentence_symbols(0)
    torch.testing.assert_close(derived.placement_witnesses,witnesses,rtol=0,atol=0)
    assert torch.all(cs._sentence_seen_rows[cs._sentence_seen_rows >= 0] != identity)
    symbol = cs.concept_id_at_row(rows[1])
    cs.mtm.promote(key,symbol)
    state = stage(model,state,torch.tensor([True]))
    assert entries.concept_ids[0,0] == symbol and entries.concept_rows[0,0] == rows[1]
    model._prime_sentence_symbols(0)
    assert rows[1] in cs._sentence_seen_rows[0].tolist()
    before = int(derived.placement_witnesses[rows[1]])
    derived.observe_close(reading)
    assert int(derived.placement_witnesses[rows[1]]) == before+1


def test_retired_modules_and_transient_checkpoint_version():
    from MediumTermMemory import MediumTermMemory
    for name in ('MTMOverlay','TimelineOverlay'):
        with pytest.raises(ModuleNotFoundError):__import__(name)
    root = Path(__file__).resolve().parents[1]/'bin'
    assert not any(('import MTMOverlay' in file.read_text() or 'import TimelineOverlay' in file.read_text())
                   for file in root.glob('*.py'))
    pool = MediumTermMemory(2,n_vectors=90)
    state = pool.state()
    state.pop('transient_ids')
    with pytest.raises(ValueError,match='transient'):
        pool.restore(state)


def test_restaging_reads_live_values_once_without_reading_the_menu(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    row = f.store.append_meaning(ConceptualMeaning.from_description(torch.eye(8)[0]),order=1)
    key = f.store.occurrence_of(row)
    f.cs.mtm.open(key,kind=None,when=1,value=f.store.slots[row],pending=False)
    entries = SymbolEntries(1,1,8,8)
    model = SimpleNamespace(_concept_owner=lambda:f.cs,inputSpace=SimpleNamespace(symbol_entries=entries),
        symbolSpace=SimpleNamespace(ltm_store=f.store),_open_sentence_slot=0)
    state = stage(model,chunks.empty(1,8,8),torch.tensor([True]))
    index = int((state[chunks.IDENTITY][0] >= 0).nonzero()[0])
    for i in range(1,8):
        state,_ = chunks.deposit(state,torch.full((1,8),i/10),torch.tensor([True]))
    previous = state[chunks.READ].clone()
    state = stage(model,state,torch.tensor([True]))
    assert state[chunks.READ][0,index] > previous.max()
    torch.testing.assert_close(state[chunks.READ][0,1:],previous[0,1:],rtol=0,atol=0)
    age = state[chunks.READ].clone()
    chunks.bind_menu(state,torch.ones(1,1,8))
    torch.testing.assert_close(state[chunks.READ],age,rtol=0,atol=0)
    state,arrived = chunks.deposit(state,torch.ones(1,8),torch.tensor([True]),stack=True)
    assert arrived.tolist() == [1]
