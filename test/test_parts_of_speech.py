"""Default certificates for the occupation categorizer, independent of learning gates."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from PartsOfSpeech import PartsOfSpeech, formal_arguments, update_heads
from Language import OperationSelectionLayer


def rule(name, arity=2, outputs=1, head=1):
    return NS(method_name=name, surface_name=name, rhs_symbols=list(range(arity)),
              lhs=','.join(f'{name}_argout{i+1}' for i in range(outputs)), head_role=head)


class Pair(nn.Module):
    def forward(self, left, right):
        return left + right


class Unary(nn.Module):
    def forward(self, value):
        return value


@pytest.fixture
def chooser():
    binary = [rule('lower', head=2), rule('intersection')]
    unary = [rule('surface_unary', 1)]
    arguments = formal_arguments(binary+unary)
    layer = OperationSelectionLayer(d_model=4, ops=[Pair(), Pair()], unary_ops=[Unary()], reduce_pressure=0.)
    layer.configure_arguments(arguments, binary, unary)
    return layer, PartsOfSpeech(arguments)


def test_kept_readings_alone_supply_occupation_counts(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    from What import What
    from AttentionLesson import prompt_need
    import GrammarLessons
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=2, word_buckets='16')
    model._tensor_peer_while_eager = True
    model.languageSpace.language_layer.operation_layer.set_stage_menu(('STOP', 'lower', 'definite', 'intersection'))
    model.inputSpace.data.has_supervised_outputs = True
    model.inputSpace.data.train_input = ['a red cat', 'the blue dog']
    model.inputSpace.data.train_output = [torch.zeros(1),torch.zeros(1)]
    model.inputSpace.data.text_answers = {'train':['cat','dog']}
    model.inputSpace.data.grammar_lessons = {'train': [
        dict(text='a red cat', tree=['lower',0,['intersection',1,2]], waits=True),
        dict(text='the blue dog', tree=['definite',0,['intersection',1,2]], waits=True)]}
    parts = model.languageSpace.parts_of_speech
    events, lessons, initial = [], [], {}
    stage = model._stage_reconstruction_teacher
    def seeded_stage():
        # Distinct, arbitrary prior classes expose an incorrect head gather.
        # This is fixture initialization, before either reading/annotation.
        if not initial:
            words = model.inputSpace._ar_word_concept_ids
            for i, word in enumerate(sorted(set(words[words>=0].flatten().tolist()))):
                parts.add(word,parts.arguments[i])
            initial.update({word:dict(counts) for word,counts in parts.counts.items()})
        stage()
    monkeypatch.setattr(model,'_stage_reconstruction_teacher',seeded_stage)
    observe, lesson = parts.observe_kept, GrammarLessons.compose_loss
    def observation(program, binary, unary):
        events.append((program, binary, unary))
        observe(program, binary, unary)
    def annotation(*args, **kwargs):
        before = {key: dict(value) for key,value in parts.counts.items()}
        result = lesson(*args, **kwargs)
        assert parts.counts == before
        lessons.append(args[1])
        return result
    monkeypatch.setattr(parts, 'observe_kept', observation)
    monkeypatch.setattr(GrammarLessons, 'compose_loss', annotation)
    commit = model._commit_sentence
    def committed(state, sid, active, observations, predictions, wins):
        before = len(events)
        result = commit(state,sid,active,observations,predictions,wins)
        admitted = events[before:]
        rows = active.nonzero().flatten().tolist()
        assert len(admitted)==len(rows)
        for (program,_,_), b in zip(admitted,rows,strict=True):
            winner = observations[int(bool(wins[b]))]['entries'][b]
            states = lessons[int(bool(wins[b])) * len(active) + b][0].student_states
            torch.testing.assert_close(program.actions,winner.actions,rtol=0,atol=0)
            torch.testing.assert_close(program.word_ids,winner.word_ids,rtol=0,atol=0)
            # Independent host replay compares every kept decision's input
            # distributions with its actual program's lexical heads.
            source = model.inputSpace._ar_word_concept_ids[b].tolist()
            words = winner.word_ids.tolist()
            places = sorted(source.index(word) for word in words)
            positions = {word:places.index(source.index(word)) for word in words}
            stack, heads = [], {}
            for kind, op, position in winner.actions.tolist():
                if kind == 0:
                    node = ((positions[words[position]],), words[position])
                    stack.append(node); heads[node[0]]=node[1]
                elif kind == 1:
                    right,left=stack.pop(),stack.pop()
                    rule=model.languageSpace._compose_binary_rules[op]
                    role=getattr(rule,'head_role',0)
                    node=(left[0]+right[0],(left,right)[role-1][1] if role else -1)
                    stack.append(node);heads[node[0]]=node[1]
                elif kind == 3:
                    stack[-2:]=reversed(stack[-2:])
            for state in states:
                for i,span in enumerate(state['spans']):
                    word=heads[span]
                    counts=initial.get(word,{})
                    expected_probability=torch.tensor([counts.get(arg,0)/max(1,sum(counts.values())) for arg in parts.arguments])
                    torch.testing.assert_close(state['kwargs']['occupation'][b,i].cpu(),expected_probability,rtol=0,atol=0)
        return result
    monkeypatch.setattr(model,'_commit_sentence',committed)
    with prompt_need(model,['what is red ?','what is blue ?']):
        inputs = model.inputSpace.prepPackedInput([['a red cat'], ['the blue dog']])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=2, split='train',
            batch_override=(inputs, torch.zeros(2,1)), source_rows=[[0],[1]],
            questions=(What.supervised(0,split='train',prompt='what is red ?'),
                       What.supervised(1,split='train',prompt='what is blue ?')))
    assert len(events) == 2 and len(lessons) == 4
    expected = PartsOfSpeech(parts.arguments)
    expected.counts={word:Counter(counts) for word,counts in initial.items()}
    for program, binary, unary in events:
        expected.observe_kept(program, binary, unary)
    assert parts.counts == expected.counts and parts.counts
    # Independently pin every input and output, phrase head and pair rewrite.
    binary = [rule('intersection', head=2), rule('lower',head=2), rule('transpose',outputs=2)]
    unary = [rule('surface_unary',1)]
    measured = PartsOfSpeech(formal_arguments(binary+unary))
    program = NS(word_ids=torch.tensor([11,12,13]), actions=torch.tensor([
        [0,-1,0],[0,-1,1],[0,-1,2],[1,0,0],[2,0,0],[3,2,0],[3,2,0],[1,1,0]]))
    measured.observe_kept(program,binary,unary)
    assert measured.counts[12] == Counter(intersection_argin1=1)
    assert measured.counts[13]['intersection_argout1'] == 1
    assert measured.counts[13]['lower_argin2'] == measured.counts[13]['lower_argout1'] == 1
    assert measured.counts[13]['surface_unary_argin1'] == measured.counts[13]['surface_unary_argout1'] == 1
    assert sum(sum(row.values()) for row in measured.counts.values()) == 3+2+4+4+3
    measured.frozen = True
    before = {word:dict(counts) for word,counts in measured.counts.items()}
    measured.observe_kept(program,binary,unary)
    assert measured.counts == before
    measured.frozen = False
    malformed = NS(word_ids=torch.tensor([11]),actions=torch.tensor([[0,-1,0],[2,0,2]]))
    with pytest.raises(ValueError,match='unary position'):
        measured.observe_kept(malformed,binary,unary)
    # A set-valued result with no declared head does not make either word
    # its lexical head. In particular the adjective cannot acquire the
    # enclosing determiner's argument by an invented first-leaf convention.
    binary = [rule('intersection',head=0),rule('lower',head=2)]
    unheaded = PartsOfSpeech(formal_arguments(binary))
    program = NS(word_ids=torch.tensor([11,12,13]), actions=torch.tensor([
        [0,-1,0],[0,-1,1],[0,-1,2],[1,0,0],[1,1,0]]))
    unheaded.observe_kept(program,binary,())
    assert unheaded.counts == {11:Counter(lower_argin1=1),
        12:Counter(intersection_argin1=1),13:Counter(intersection_argin2=1)}
    model.End()


def test_one_stored_part_of_speech_is_the_recomputed_argmax(chooser):
    _, parts = chooser
    a, b = parts.arguments[:2]
    assert parts.part_of_speech(7) == a
    assert sum(parts.distribution(7)) == pytest.approx(1.)
    parts.add(7,b)
    assert parts.part_of_speech(7) == b
    parts.add(7,a); parts.add(7,a)
    assert parts.part_of_speech(7) == a and parts.labels == {7:a}
    assert parts.distribution(7)[parts.index[a]] == pytest.approx(2/3)
    # Cosines report shared occupants without changing either argument.
    before = parts.get_extra_state().clone()
    assert next(row for row in parts.collapse() if row['left']==a and row['right']==b)['cosine'] == 1.
    assert torch.equal(before, parts.get_extra_state())


def test_chooser_adds_input_probabilities_directly_and_carries_heads(chooser):
    layer, parts = chooser
    x = torch.randn(1,2,4)
    probabilities = torch.zeros(1,2,len(parts.arguments))
    probabilities[0,0,parts.index['lower_argin1']] = .25
    probabilities[0,1,parts.index['lower_argin2']] = .75
    probabilities[0,0,parts.index['surface_unary_argin1']] = .4
    probabilities[0,1,parts.index['lower_argout1']] = .9
    base = layer(x,slots=2)[2]['logits']
    logits = layer(x,slots=2,occupation=probabilities)[2]['logits']
    torch.testing.assert_close(logits-base, torch.tensor([[1.,0.,.4,0.,0.]]))
    heads = torch.arange(12.).reshape(1,3,4)
    choice = NS(applied=torch.tensor([True]),kind=torch.tensor([1]),local_op=torch.tensor([0]))
    assert torch.equal(update_heads(heads,choice,(2,1))[:,0],heads[:,0])
    assert torch.equal(update_heads(heads,choice,(0,1))[:,0],torch.zeros_like(heads[:,0]))
    choice.kind.fill_(3)
    assert torch.equal(update_heads(heads,choice,(2,1))[:,:2],heads[:,:2].flip(1))
    compiled = torch.compile(lambda p:layer(x,slots=2,occupation=p)[2]['logits'],backend='eager',fullgraph=True)
    torch.testing.assert_close(compiled(probabilities),logits,rtol=0,atol=0)


def test_stage_menu_masks_every_undeclared_operation(chooser):
    layer, parts = chooser
    layer.set_stage_menu(('STOP','lower'))
    logits = layer(torch.randn(1,2,4),slots=2,occupation=torch.ones(1,2,len(parts.arguments)))[2]['logits']
    assert torch.isfinite(logits).tolist() == [[True,False,False,False,True]]
    from np_corpus import corpus, load
    rows = corpus()
    assert len(rows)==56 and sum(row[0]=='train' for row in rows)==42
    assert sum(row[0]=='eval' and row[3]['family']=='np' for row in rows)==10
    assert len({row[1]['text'] for row in rows}) == 56
    for split in ('train','eval'):
        text, labels = load(split)
        assert text == [row[1] for row in rows if row[0]==split]
        assert labels == {row[1]['id']:row[3] for row in rows if row[0]==split}
    from np_experiment import batches,first_lesson,lesson_stage,STAGES
    training,_ = load('train')
    scheduled=list(batches(training))
    assert len(scheduled)==11 and sorted(index for batch in scheduled for index in batch)==list(range(42))
    assert all(len(training[i]['text'].split())==2 and training[i]['id'].split(':')[1]=='np'
               for batch in scheduled[:2] for i in batch)
    assert all(len(training[i]['text'].split())==3 for batch in scheduled[2:8] for i in batch)
    assert all(training[i]['id'].split(':')[1]=='description' for batch in scheduled[8:] for i in batch)
    first=[row for row in training if first_lesson(row)]
    heldout,_=load('eval')
    assert [row['text'] for row in first]==['a cat','a dog','a bird']
    assert [row['text'] for row in heldout if first_lesson(row)]==['a horse']
    assert list(batches(first))==[[0,1,2]]
    partitions=[[row for row in training if lesson_stage(row)==stage] for stage in STAGES]
    assert [len(part) for part in partitions]==[3,3,12,12,12]
    assert sum(4*len(list(batches(part))) for part in partitions)==44
    assert [sum(lesson_stage(row)==stage for row in heldout) for stage in STAGES]==[1,1,4,4,4]


def test_artifact_restoration_retains_weights_and_occupation_state_exactly(tmp_path, monkeypatch):
    from Models import BaseModel
    from identity_experiment import make_model,restore_warmup
    from IndependentComponents import _unpack_state
    restore = BaseModel.load_weights
    model = make_model(tmp_path, monkeypatch)
    archive = Path(__file__).resolve().parents[1]/'output/stage4e-student-order-20261017/order.ckpt'
    restore_warmup(model,archive,restore)  # Asserts every retained tensor, with declared migration.
    parts = model.languageSpace.parts_of_speech
    assert parts.counts == {}
    for word in range(123,143):
        parts.add(word,parts.arguments[0]);parts.add(word,parts.arguments[-1])
    expected = _unpack_state(parts.get_extra_state())
    path = tmp_path/'occupation.ckpt'
    model.save_weights(str(path))
    saved_size=parts.get_extra_state().numel()
    model.End()
    fresh=tmp_path/'fresh';fresh.mkdir()
    model=make_model(fresh,monkeypatch)
    parts=model.languageSpace.parts_of_speech
    assert parts.counts=={} and parts.get_extra_state().numel()!=saved_size
    restore_warmup(model,path,restore)
    assert _unpack_state(parts.get_extra_state()) == expected
    assert parts.labels == {word:parts.arguments[0] for word in range(123,143)}
    from GrammarCheckpoint import align,catalog
    damaged = model.state_dict()
    key = next(key for key in damaged if key.endswith('.parts_of_speech._extra_state'))
    del damaged[key]
    with pytest.raises(ValueError,match='missing its occupation state'):
        align(model,damaged,catalog(model))
    migrated,_,_ = align(model,damaged,None)
    assert _unpack_state(migrated[key])['counts'] == {}
    model.End()
