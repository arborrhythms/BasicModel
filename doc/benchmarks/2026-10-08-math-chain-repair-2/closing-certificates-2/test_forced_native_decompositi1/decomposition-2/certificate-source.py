"""FORCED §14.11 development certificate; not a learning measurement.

Only grammar choices and checked thought requests are forced. The ordinary
driver, optimizer, observer, fork, masked departure, costs and writer run.
The runtime must justify each filling from the rows its executors read.
"""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test'), str(HERE)]
from math_chain_ordinary import ordinary_model, train_documents
import closing_episode_state as footprint
from forced_decomposition_grammar import ForcedDecompositionGrammar
from math_chain_corpus import ChainDocument
from ThoughtReferences import bindings, question, open_slots
from ThoughtStream import query_pattern
from Queries import ThoughtOperationCandidate
import ThoughtStream


def run(folder, successors=1):
    folder.mkdir(exist_ok=False)
    shutil.copy2(__file__, folder/'certificate-source.py')
    shutil.copy2(HERE/'forced_decomposition_grammar.py', folder/'grammar.py')
    torch.set_num_threads(1)
    question_budget = 512 if successors == 1 else 768
    model = ordinary_model(folder, budget=question_budget)
    if successors == 1:
        sentences = ('x is three.', 'y is x plus one.', 'three plus one is four.', 'what is y ?')
        assignments = {'y': 1, 'x': 0}
        licenses = {'y': 2}
        steps = (('three', 'one', 'four'),)
    else:
        sentences = ('x is three.', 'z is x plus one.', 'y is z plus one.',
                     'three plus one is four.', 'four plus one is five.', 'what is y ?')
        assignments = {'y': 2, 'z': 1, 'x': 0}
        licenses = {'z': 3, 'y': 4}
        steps = (('three', 'one', 'four'), ('four', 'one', 'five'))
    documents = [ChainDocument(str(i), sentences, question=len(sentences)-1,
        answer=steps[-1][-1], pair=(3, successors), steps=steps) for i in range(2)]
    sources, episodes, menus = {}, [], []
    original_menu = ThoughtStream.candidates
    chooser = model._selected_thought_chooser(None)
    original_logits = chooser.thought_logits

    def preferred_references(row, labels):
        for name, index in assignments.items():
            source = sources.get((row, index))
            if name in labels and source is not None:
                return source.role_refs[0][-1]
        for word, index in [('three', assignments['x']), ('four', min(licenses.values()))]:
            source = sources.get((row, index))
            if word in labels and source is not None:
                return source.role_refs[2][-1]
        return None

    def query_request(registry, root, source):
        pattern = query_pattern(root, 'equal', source.role_refs[0], None, source.roles[0])
        pattern = replace(pattern, bindings=dict(bindings(pattern), _query_components=True))
        return registry.form('query', pattern, bindings={'_forced_decomposition': True})

    def candidates(registry, root, active, current, records, descriptions=()):
        actions = original_menu(registry, root, active, current, records, descriptions)
        target = root.role_refs[2]
        selected = None
        env = next(((doc, name) for (doc, index), source in sources.items()
            for name, sentence in assignments.items() if index == sentence and source.role_refs[0] == target), None)
        if env is not None:
            doc, name = env
            reads = [record for record in records if record.kind == 'thought'
                     and record.result is not None and record.operation == 'query']
            returned = [record for record in records if record.kind == 'return']
            if not reads:
                selected = query_request(registry, root, sources[doc, assignments[name]])
            elif name in licenses and not returned:
                region = current.role_refs[0]
                components = reads[0].result.evidence.get('components', {})
                compound = components.get(region)
                if compound and compound['operands']:
                    subject = compound['operands'][0]
                    atom = components.get(subject)
                    if atom is not None:
                        roles = root.roles.clone()
                        roles[2] = atom['meaning'].roles[0]
                        child = question(replace(root, roles=roles,
                            role_refs=(None, root.role_refs[1], subject),
                            bindings={'_equality': True}, constituents=()), (('referent', 0),))
                        selected = registry.form('ask', child, bindings={'_forced_decomposition': True})
            elif name in licenses and len(reads) == 1:
                selected = query_request(registry, root, sources[doc, licenses[name]])
        if selected is not None:
            actions = (*actions, ThoughtOperationCandidate(registry.operation_spec(
                registry.signature_for(selected).operation.semantic_id), selected, ()))
        menus.append(dict(target=target, env=env, current=current.role_refs,
            chosen=None if selected is None else registry.signature_for(selected).operation.semantic_id))
        with (folder/'menus.jsonl').open('a') as stream:
            stream.write(json.dumps(menus[-1])+'\n')
        return actions

    def logits(active, requests):
        value = original_logits(active, requests)
        return value + value.new_tensor([100. if request is not None and
            bindings(request).get('_forced_decomposition') else 50. if request is None else 0.
            for request in requests])

    def after(model, split, rows, result, observed, observer):
        store = model.symbolSpace.ltm_store
        for row, source in enumerate(rows):
            address = model.inputSpace.data.source_addresses[split][source]
            field = model._sentence_fields[0][row]
            sources[address['document'], address['sentence']] = store.meaning_of(store.index_of_row(field.row_id))
        (folder/'progress.json').write_text(json.dumps(dict(rows=observed,episodes=episodes,menus=menus),indent=2)+'\n')

    def episode(original, model, meaning, **kwargs):
        before = footprint.snapshot(model)
        store = model.symbolSpace.ltm_store
        old = [footprint.value_digest(store.meaning_of(i)) for i in range(len(store))]
        start = len(store)
        result = original(model, meaning, **kwargs)
        changes = footprint.difference(before, footprint.snapshot(model))
        trail = {'.'+name for name in ('_last_thought_comparison', '_last_thought_score_function',
                 '_thought_ordinals', '_walk_audit', '_walk_observations', '_walk_previous')}
        trail.update('what_memory.'+name for name in (
            '_address_sources', '_episode_live', '_thought_next_id', '_what_slots', '_what_closure_pressure'))
        unexpected = [key for key in changes if '._ltm_store.' not in key and key not in trail]
        changed_rows = [i for i, previous in enumerate(old) if previous != footprint.value_digest(store.meaning_of(i))]
        source_index = store._index_occurrences.get(bindings(meaning).get('_query_occurrence'))
        assert not unexpected, unexpected
        assert set(changed_rows) <= {source_index}, changed_rows
        added = [store.KINDS[int(store.record_kind[i])] for i in range(start, len(store))]
        episodes.append(dict(row=kwargs['row'], before=meaning.role_refs, after=result.meaning.role_refs,
            open=open_slots(result.meaning), metadata=bindings(result.meaning), work=result.work.spent,
            state_diff=dict(changed=changes, unexpected=unexpected, changed_old_rows=changed_rows,
                            source_row=source_index, added_row_kinds=added),
            trace=[dict(kind=r.kind, operation=r.operation, level=r.level,
                        refs=None if r.meaning is None else r.meaning.role_refs,
                        frames=[] if r.result is None else [dict(occurrence=f['occurrence'],
                            refs=f['meaning'].role_refs) for f in r.result.evidence.get('frames', ())])
                   for r in result.records]))
        return result

    try:
        with ForcedDecompositionGrammar(model, open_names=('what',), question_budget=question_budget,
                preferred_references=preferred_references) as grammar, \
                patch.object(ThoughtStream, 'candidates', candidates), \
                patch.object(chooser, 'thought_logits', logits):
            rows = train_documents(model, documents, folder, after=after, episode=episode)
        report = dict(forced=True, successors=successors, attention_budget=question_budget,
            premise_budget=0,
            episodes=episodes, menus=menus, rows=rows, grammar_choices=grammar.records)
        (folder/'certificate.json').write_text(json.dumps(report, indent=2)+'\n')
        print((folder/'questions.jsonl').read_text())
        return report
    finally:
        model.End()


if __name__ == '__main__':
    run(HERE/sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
