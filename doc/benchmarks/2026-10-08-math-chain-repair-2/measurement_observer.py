"""Read-only reporting beside the frozen math observer.

No sampling, policy overrides, answer scoring, or extra model execution. The
greedy closing is read from the existing first compose trial before commit.
Episode work is the kept episode's native meter, not the sum of both trials.
"""
from collections import defaultdict
from contextlib import ExitStack
import json
import time
from unittest.mock import patch


class Measurements:
    def __init__(self, folder):
        self.folder = folder
        self.context = None
        self.episodes = {}
        self.greedy = {}
        self.batches = 0
        self.totals = defaultdict(lambda: dict(sentences=0, episodes=0, work=0,
            references_left_open=0, sentences_left_open=0, seconds=0.))
        self.what = []
        self.model = None

    def __enter__(self):
        from Models import BasicModel
        from ThoughtReferences import bindings, open_slots
        original_batch = BasicModel.runBatch
        original_episode = BasicModel.run_selected_thought
        original_commit = BasicModel._commit_sentence

        def commit(model, state, sid, active, observations, predictions, wins):
            if self.context is not None:
                greedy = observations[0]
                for row, info in enumerate(self.context['rows']):
                    if not info['question'] or not bool(active[row]):
                        continue
                    meaning, program = greedy['meanings'][row], greedy['entries'][row]
                    if meaning is None or program is None:
                        self.greedy[row] = dict(what_identified=False, what_open=None,
                            open_references=None)
                        continue
                    def form(value):
                        return value.decode('utf-8') if isinstance(value, bytes) else value
                    identities = {int(program.concept_ids[index])
                        for index, value in enumerate(program.lexical_forms)
                        if form(value) == 'what'}
                    # The ordinary component reader can omit lexical_forms;
                    # its native word IDs still identify the presented leaf.
                    known = set(model._concept_owner().word_concepts('what'))
                    if program.word_ids is not None:
                        identities.update(int(program.concept_ids[index])
                            for index, value in enumerate(program.word_ids.tolist())
                            if value in known)
                    identities.update(int(value) for value in program.concept_ids.tolist()
                                      if value in known)
                    pending, seen, what_roles = [meaning], set(), []
                    while pending:
                        value = pending.pop()
                        if id(value) in seen:
                            continue
                        seen.add(id(value))
                        opened = set(open_slots(value))
                        what_roles.extend(role for role, identity in
                            bindings(value).get('_forward_references', ())
                            if identity in identities and ('referent', role) in opened)
                        pending.extend(value.constituents)
                    self.greedy[row] = dict(what_identified=bool(identities),
                        what_open=bool(what_roles) if identities else None,
                        what_open_roles=what_roles, open_references=list(open_slots(meaning)),
                        kept_trial='explore' if bool(wins[row]) else 'greedy')
                    if not identities:
                        self.greedy[row]['unidentified_leaf_metadata'] = dict(
                            forms=list(map(repr, program.lexical_forms)),
                            concepts=program.concept_ids.tolist(),
                            words=None if program.word_ids is None else program.word_ids.tolist(),
                            known=sorted(known))
            return original_commit(model, state, sid, active, observations, predictions, wins)

        def episode(model, meaning, **kwargs):
            if self.context is None:
                return original_episode(model, meaning, **kwargs)
            row = kwargs.get('row', 0)
            event = dict(references_open=len(open_slots(meaning)), completed=False)
            self.episodes[row] = event
            started = time.perf_counter()
            try:
                result = original_episode(model, meaning, **kwargs)
                event.update(completed=True, work=result.work.spent,
                    steps=sum(record.kind == 'thought' for record in result.records))
                return result
            finally:
                event['seconds'] = time.perf_counter()-started

        def batch(model, *args, **kwargs):
            self.model = model
            data = model.inputSpace.data
            split = kwargs.get('split', 'train')
            sources = kwargs.get('source_rows', ())
            rows = []
            for source in sources:
                address = data.source_addresses[split][source]
                doc = data.math_chain_documents[split][address['document']]
                text = doc.sentences[address['sentence']]
                question = address['sentence'] == doc.question
                kind = 'question' if question else 'answer_line' if text.startswith('the answer is ') else 'plain'
                rows.append(dict(source=source, document=doc.key, sentence=address['sentence'],
                    epoch=address['document_key'][2]+1, question=question, kind=kind, text=text))
            self.context = dict(split=split, phase='train' if kwargs.get('train') else 'evaluation', rows=rows)
            self.episodes, self.greedy = {}, {}
            started, completed, failure = time.perf_counter(), False, None
            try:
                result = original_batch(model, *args, **kwargs)
                completed = True
                return result
            except BaseException as error:
                failure = repr(error)
                raise
            finally:
                elapsed = time.perf_counter()-started
                self.record_batch(elapsed, completed, failure, len(model.symbolSpace.ltm_store))
                self.context = None

        self.stack = ExitStack()
        self.stack.enter_context(patch.object(BasicModel, 'runBatch', batch))
        self.stack.enter_context(patch.object(BasicModel, 'run_selected_thought', episode))
        self.stack.enter_context(patch.object(BasicModel, '_commit_sentence', commit))
        return self

    def record_batch(self, seconds, completed, failure, occupancy):
        self.batches += 1
        base = max(0., seconds-sum(event['seconds'] for event in self.episodes.values()))
        rows = self.context['rows']
        base /= max(1, len(rows))
        sentences = []
        for row, info in enumerate(rows):
            event, greedy = self.episodes.get(row), self.greedy.get(row)
            item = dict(info, episode=event, greedy=greedy,
                seconds=base+(0. if event is None else event['seconds']))
            sentences.append(item)
            if completed:
                key = (info['epoch'], self.context['phase'], self.context['split'], info['kind'])
                totals = self.totals[key]
                totals['sentences'] += 1
                totals['episodes'] += event is not None
                totals['work'] += 0 if event is None else event['work']
                totals['sentences_left_open'] += event is not None and event['references_open'] > 0
                totals['references_left_open'] += 0 if event is None else event['references_open']
                totals['seconds'] += item['seconds']
            if greedy is not None:
                self.what.append(dict(epoch=info['epoch'], phase=self.context['phase'],
                    split=self.context['split'], document=info['document'],
                    completed_batch=completed, **greedy))
        record = dict(batch=self.batches, phase=self.context['phase'], split=self.context['split'],
            completed=completed, failure=failure, seconds=seconds, occupancy=occupancy,
            sentences=sentences)
        with (self.folder/'episode-batches.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        self.write_summaries()

    def write_summaries(self):
        rows = [dict(epoch=key[0],phase=key[1],split=key[2],kind=key[3],**value,
            mean_work_per_episode=value['work']/value['episodes'] if value['episodes'] else None,
            episode_share=value['episodes']/value['sentences'],
            seconds_per_sentence=value['seconds']/value['sentences'])
            for key,value in sorted(self.totals.items())]
        first = [row for row in self.what if row['epoch']==1 and row['phase']=='train']
        report = dict(definition='Existing greedy compose closing, before the keep decision is committed; free referent attributed to the lexical what by the native forward identity. No extra forward pass.',
            first_question=first[0] if first else None,
            first_epoch_questions_observed=len(first),
            first_epoch_what_opens=sum(row['what_open'] is True for row in first),
            first_epoch_what_identified=sum(row['what_identified'] for row in first),
            detail_file='episode-batches.jsonl')
        for name,value in (('episodes-by-epoch-kind.json',rows),('greedy-what.json',report)):
            (self.folder/name).write_text(json.dumps(value,indent=2)+'\n')

    def __exit__(self, *exception):
        self.stack.close()
        if not self.folder.exists():
            return
        self.write_summaries()
        log = self.folder/'questions.jsonl'
        questions = [] if not log.exists() else [json.loads(line) for line in log.read_text().splitlines()]
        questions = [row for row in questions if row['epoch']==1 and row['phase']=='train']
        totals = self.totals.get((1,'train','train','question'),{})
        first = dict(questions_observed=totals.get('sentences',0),
            questions_left_open_at_closing=totals.get('sentences_left_open',0),
            references_left_open_at_closing=totals.get('references_left_open',0),
            episodes_opened=totals.get('episodes',0),
            bindings_correct=sum(row['bound_correct'] for row in questions),
            frozen_observer_questions=len(questions),
            definition='Completed batches only; openings before the episode; correctness from the unchanged frozen observer on committed rows. Partial failed batch observations are retained in episode-batches.jsonl.')
        (self.folder/'first-epoch-questions.json').write_text(json.dumps(first,indent=2)+'\n')
        if self.model is not None and (self.folder/'initial-chooser.pt').exists():
            import torch
            before = torch.load(self.folder/'initial-chooser.pt',map_location='cpu',weights_only=True)
            movement = {name:float((value.detach().cpu()-before[name]).norm())
                for name,value in self.model._selected_thought_chooser(None).named_parameters() if name in before}
            (self.folder/'final-chooser-movement.json').write_text(json.dumps(dict(
                completed=exception[0] is None, movement=movement),indent=2)+'\n')
