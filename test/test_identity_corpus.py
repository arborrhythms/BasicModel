"""The lesson/measurement boundary cannot teach or hide the grading identity."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json

import pytest

from identity_corpus import KINDS, PROPERTIES, VERBS, DEST, build, write
from identity_measurement import control_observation, score_document, summarize


def test_saved_corpus_matches_generator_and_keeps_labels_out_of_text(tmp_path):
    manifest=write(tmp_path)
    assert manifest==json.loads((DEST/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((DEST/name).read_bytes()).hexdigest()==digest
        assert (DEST/name).read_bytes()==(tmp_path/name).read_bytes()
    pairs=build()
    assert len({p['id'] for p,_ in pairs})==len(pairs)
    for public,label in pairs:
        assert set(public)=={'id','sentences'}
        assert public['id']==label['id']
        for mention in label['mentions']:
            assert public['sentences'][mention['sentence']].split()[mention['word']]==mention['form']
    train={tuple(p['sentences']) for p,g in pairs if g['split']=='train'}
    held={tuple(p['sentences']) for p,g in pairs if g['split']=='eval'}
    assert train.isdisjoint(held)


def test_factorial_lessons_and_support_rehearsal_do_not_confuse_missing_factors():
    labels=[g for _,g in build()]
    factorial=[g for g in labels if g['family']=='factorial_support']
    assert [g['tags']['object_count'] for g in factorial]==sorted(g['tags']['object_count'] for g in factorial)
    for support in (1,2,3):
        rows=[g for g in factorial if g['tags']['object_count']==support]
        pairs={(kind,prop,g['tags']['verb']) for g in rows
               for kind,prop in zip(g['tags']['kinds'],g['tags']['properties'])}
        assert pairs=={(kind,prop,verb) for kind in KINDS for prop in PROPERTIES for verb in VERBS}
        recurring={atom for g in labels if g['family']=='support_rehearsal'
                   and g['tags']['object_count']==support for atom in g['tags']['atoms']}
        assert recurring==set((*KINDS,*PROPERTIES,*VERBS))
    confounded=[g for g in labels if g['family']=='confounded']
    for kind in KINDS:
        assert len({g['tags']['properties'][0] for g in confounded if g['tags']['kinds']==[kind]})==1


@pytest.mark.parametrize('split',['train','eval'])
def test_two_candidate_pronouns_cross_refresh_recency_with_role_and_report_position_confound(split):
    labels=[g for _,g in build() if g['split']==split and g['family']=='pronoun']
    cells=Counter((g['probe']['recency_rank'],g['probe']['role'],g['probe']['first_mention_rank']) for g in labels)
    assert len(cells)==4 and set(cells.values())=={8}
    assert all(g['tags']['introduction_role_position_confounded'] for g in labels)
    assert all((role, position) in (('subject', 0), ('object', 1)) for _, role, position in cells)
    by_kind=Counter()
    for g in labels:
        mentions={m['id']:m for m in g['mentions']}
        target=mentions[g['probe']['target']]['entity']
        by_kind[target,g['probe']['recency_rank'],g['probe']['role']]+=1
    assert set(by_kind.values())=={2}


def test_counterbalancing_preserves_grammatical_roles_instead_of_passive_agent_labels():
    for public,label in build():
        if label['family'] not in ('pronoun','neither_candidate','recency_biased','reversed_recency'):
            continue
        words=public['sentences'][0].split()
        mentions={m['role']:m for m in label['mentions'] if m['sentence']==0}
        subject,object_=mentions['subject'],mentions['object']
        # No unverified fronting or passive voice in the measured lesson.
        assert words[subject['word']+1]=='saw'
        assert 'by' not in words and 'was' not in words
        assert object_['word']>subject['word'] and ',' not in words


def test_same_kind_later_properties_use_the_original_verb_forms():
    for public, label in build():
        if label['family'] != 'same_kind':
            continue
        words = [s.split() for s in public['sentences']]
        assert words[0][-2] == words[2][-2] == 'runs'
        assert words[1][-2] == words[3][-2] == 'sleeps'
        assert not {'running', 'sleeping'} & set(sum(words, []))


def test_imperfect_determiners_and_content_conflicts_have_separate_evidence():
    training=[g for _,g in build() if g['split']=='train' and g['family']=='determiner']
    assert sum(g['tags']['cue_correct'] for g in training)/len(training)==.9
    for expected in (1,2):
        rows=[g for g in training if g['row_pair']['expected']==expected]
        assert sum(g['tags']['cue_correct'] for g in rows)/len(rows)==.9
    conflicts=[(p,g) for p,g in build() if g['family']=='determiner_conflict']
    assert len(conflicts)==4
    for p,g in conflicts:
        assert 'black' in p['sentences'][0] and 'white' in p['sentences'][1]
        assert p['sentences'][1].startswith('the ') and g['row_pair']['expected']==2


def test_neither_candidate_control_has_identical_text_for_both_hidden_answers():
    by_text=defaultdict(list)
    for public,label in build():
        if label['family']=='neither_candidate':by_text[tuple(public['sentences'])].append(label)
    assert by_text
    for labels in by_text.values():
        assert len(labels)==4  # two repeat witnesses of each hidden answer
        assert {g['probe']['recency_rank'] for g in labels}=={0,1}
        assert not any(g['probe']['informative'] for g in labels)
    rows=[score_document(g,control_observation(g,'recent')) for labels in by_text.values() for g in labels]
    assert summarize(rows)['groups']['all']['accuracy']==.5


def test_reversed_recency_control_exposes_position_shortcut():
    rows=build()
    for split,expected in [('biased_train',1.),('reversed_eval',0.)]:
        scored=[score_document(g,control_observation(g,'recent')) for _,g in rows if g['split']==split]
        assert summarize(scored)['groups']['all']['accuracy']==expected


def test_missing_and_colliding_candidates_cannot_be_counted_as_correct():
    label=next(g for _,g in build() if g['split']=='eval' and g['family']=='pronoun')
    ideal=control_observation(label,'oracle')
    assert score_document(label,ideal)['binding']['correct']
    missing=deepcopy(ideal); missing['references'][label['probe']['candidates'][1]]=None
    collision=deepcopy(ideal)
    for mid in label['probe']['candidates']+[label['probe']['mention']]:collision['references'][mid]=-72
    for obs in (missing,collision,{'references':{}}):
        assert not score_document(label,obs)['binding']['correct']
    summary=summarize([score_document(label,obs) for obs in (ideal,missing,collision)])['groups']['all']
    assert summary['accuracy']==pytest.approx(1/3) and summary['conditional_accuracy']==1.
    assert summary['coverage']==pytest.approx(1/3) and summary['collisions']==1


@pytest.mark.parametrize('expected',[1,2])
def test_one_vs_two_measures_semantic_addresses_including_negative_occurrences(expected):
    label=next(g for _,g in build() if g['family']=='same_kind' and g['row_pair']['expected']==expected)
    first,second=label['row_pair']['mentions']
    good={'references':{first:-72,second:-72 if expected==1 else -84}}
    assert score_document(label,good)['rows']['correct']
    good['references'][second]=-1
    result=score_document(label,good)['rows']
    assert not result['correct'] and result['actual'] is None


def test_document_context_changes_referent_with_the_same_probe_words():
    rows=[(p,g) for p,g in build() if g['split']=='eval' and g['family']=='document_context']
    targets=defaultdict(set)
    for public,label in rows:
        query=label['probe']; mentions={m['id']:m for m in label['mentions']}
        targets[public['sentences'][-1]].add(mentions[query['target']]['entity'])
    assert targets and all(len(values)==2 for values in targets.values())


def test_native_observer_uses_owned_individual_not_enclosing_sentence(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    import torch
    from ClauseJournal import finish_clause
    from identity_measurement import native_observer
    from reading_fixtures import record_reading, resolve_reading_references
    from test_clause_acceptance import SentenceFixture
    fixture=SentenceFixture(monkeypatch)
    entry=fixture.program(('lift',('lower','a','cat'),'runs'))
    entry=replace(entry,leaf_orders=torch.tensor([0,2,0]))
    entry=record_reading(fixture.language,resolve_reading_references(
        fixture.language,entry,forced={1:-1}))
    clause=finish_clause(fixture.language,entry,registry=fixture.registry)
    model=SimpleNamespace(languageSpace=fixture.language,
        symbolSpace=SimpleNamespace(ltm_store=fixture.store),
        inputSpace=SimpleNamespace(_word_active_mask=torch.ones(1,3,dtype=torch.bool),
            _packed_sentence_ids=torch.zeros(1,3,dtype=torch.long)),
        _sentence_observation=lambda *a,**k:dict(entries=(entry,)),
        _derivation_program=lambda *a:(torch.arange(3)[None],))
    with native_observer(model) as captured:
        model._sentence_observation(None,0,None,admit=True)
        token=captured[0][0]['tokens'][1]
        assert token['requested'] and token['reference'] is None
        assert token['selected_reference']==-1
        assert token['missing_reason']=='mint_awaits_row_write'
        written={}
        root=fixture.store.write_clause(clause,stream=0,written_rows=written)
        child=written[id(clause.children[0])]
        assert root != child
        assert clause.order == 0 and clause.children[0].order == 1
        assert token['reference']==int(fixture.store.row_ids[child])
        assert token['reference']!=int(fixture.store.row_ids[root])
        assert token['missing_reason'] is None

        # A later selected bind must expose the same address to the grader.
        # This is an explicit grammar/writer mechanism fixture, not a lesson.
        from ReferenceContext import SituationFrame
        from identity_measurement import grade_capture, score_document
        address=token['reference']
        bound=fixture.program(('lift',('lower','the','cat'),'runs'))
        bound=replace(bound,leaf_orders=torch.tensor([0,2,0]))
        local=next(i for i,r in enumerate(fixture.binary) if r.determiner_mode=='bind')
        bound.actions[2,1]=local
        bound=record_reading(fixture.language,resolve_reading_references(
            fixture.language,bound,frames=(SituationFrame.from_row(fixture.store.row(child)),),
            forced={1:address}))
        entry=bound
        model.inputSpace._packed_sentence_ids.fill_(1)
        model._sentence_observation(None,1,None,admit=True)
        assert captured[1][0]['tokens'][1]['reference']==address
        label=dict(id='mint-bind',family='determiner',probe=None,
            mentions=[dict(id='m0',sentence=0,word=1),dict(id='m1',sentence=1,word=1)],
            row_pair=dict(mentions=['m0','m1'],expected=1))
        graded=score_document(label,grade_capture(label,captured,0))
        assert graded['rows']['correct'] and graded['rows']['resolved']


def test_observer_reports_discarded_mint_and_unadmitted_leaf_by_reason(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    import torch
    from ClauseJournal import finish_clause
    from identity_measurement import native_observer, grade_capture
    from reading_fixtures import record_reading, resolve_reading_references
    from test_clause_acceptance import SentenceFixture
    fixture=SentenceFixture(monkeypatch)
    # The second projection discards the cat individual; only the final
    # word becomes the head of the written row. Never assign that row to cat.
    entry=fixture.program(('lower',('lower','a','cat'),'.'))
    entry=replace(entry,leaf_orders=torch.tensor([0,2,2]))
    entry=record_reading(fixture.language,resolve_reading_references(
        fixture.language,entry,forced={1:-1,2:-1}))
    clause=finish_clause(fixture.language,entry,registry=fixture.registry)
    model=SimpleNamespace(languageSpace=fixture.language,
        symbolSpace=SimpleNamespace(ltm_store=fixture.store),
        inputSpace=SimpleNamespace(_word_active_mask=torch.ones(1,3,dtype=torch.bool),
            _packed_sentence_ids=torch.zeros(1,3,dtype=torch.long)),
        _sentence_observation=lambda *a,**k:dict(entries=(entry,)),
        _derivation_program=lambda *a:(torch.arange(3)[None],))
    with native_observer(model) as captured:
        model._sentence_observation(None,0,None,admit=True)
        fixture.store.write_clause(clause,stream=0,written_rows={})
        label=dict(mentions=[dict(id='cat',sentence=0,word=1),
            dict(id='unread',sentence=0,word=7),dict(id='absent',sentence=1,word=0)])
        result=grade_capture(label,captured,0)
    assert all(value is None for value in result['references'].values())
    assert result['provenance']['cat']['missing_reason']=='mint_not_retained_in_written_clause'
    assert result['provenance']['unread']['missing_reason']=='leaf_not_admitted'
    assert result['provenance']['absent']['missing_reason']=='sentence_not_observed'
