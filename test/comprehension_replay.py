"""Replay a fixed reading's numerical inputs, independent of the old source.

The recorded calls are dictionary/MTM inputs, not operator outputs. Operators,
interpretation, extension, held-value refresh and clause assembly run natively.
The fixture stops at clause assembly; the native-store certificates cover its
subsequent staging and admission separately.
"""
from types import SimpleNamespace

import torch


def key(value):
    if torch.is_tensor(value):
        return ('tensor', str(value.dtype), tuple(value.shape), tuple(value.flatten().tolist()))
    if isinstance(value, dict):
        return tuple(sorted((k, key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(key(v) for v in value)
    return value


# A method renamed since the recording reads the calls recorded under its
# old name; the recorded evidence is not rewritten.
RECORDED_AS = {'_concept_row_of': '_csw_row_of'}


class Reads:
    def __init__(self, calls, prefix):
        self.calls, self.prefix = calls, prefix

    def __getattr__(self, name):
        def read(*args, **kwargs):
            # The recording used a scalar long tensor for a single bank row.
            # Address the same recorded call when the reader already has its integer.
            if self.prefix == 'bank' and name == 'lookup_rows' and type(args[0]) is int:
                args = (torch.tensor(args[0], dtype=torch.long), *args[1:])
            return self.calls[self.prefix + '.' + RECORDED_AS.get(name, name)][key((args, kwargs))]
        return read


def grammar(value):
    from Language import TheGrammar
    if isinstance(value, dict) and 'grammar_type' in value:
        return getattr(TheGrammar, value['grammar_type'])(**{
            k: grammar(v) for k, v in value['fields'].items()})
    if isinstance(value, (tuple, list)):
        return type(value)(grammar(v) for v in value)
    return value


def replay(record, monkeypatch):
    from Understanding import Reading
    from Comprehension import Evaluation
    from Extension import reading, with_extensions
    from SentenceStrata import SentenceStrata
    from Interpret import InterpretLayer
    from RowTime import encoder
    import Models

    inputs = record['inputs']
    owner = Reads(record['reads'], 'owner')
    owner.definitions = SimpleNamespace(word_ids=inputs['word_ids'])
    owner._concept_allocator = Reads(record['reads'], 'allocator')
    owner.similarity_codebook = Reads(record['reads'], 'bank')
    owner.similarity_codebook.mereology = SimpleNamespace(percept_event_width=inputs['percept_width'])
    interpreter = InterpretLayer(conceptualSpace=owner)
    for name in ('referent', 'preferred', 'denotations', 'no_object'):
        object.__setattr__(interpreter, name, getattr(Reads(record['reads'], 'interpret'), name))
    owner.interpret = interpreter
    store = StoreReads(record['reads'], 'store')
    store.when_start = inputs['when_start']
    store.count = inputs['store_count']
    owner._closed_clause_store = lambda: store
    owner._clause_store_ref = lambda: store
    registry = SimpleNamespace(space=owner)
    monkeypatch.setattr(Models, '_boundary_registry', lambda model: registry)
    binary, unary = grammar(inputs['binary']), grammar(inputs['unary'])
    operations = {int(index): value for index, value in inputs['operators'].items()}
    language = SimpleNamespace(_compose_binary_rules=binary, _compose_unary_rules=unary,
        _tree_layer=lambda arity: SimpleNamespace(ops=operations, unary_ops={}),
        chunk_journal=SimpleNamespace(percept_band=inputs['percept_band']),
        parts_of_speech=SimpleNamespace(seats=inputs['seats']))
    model = SimpleNamespace(_concept_owner=lambda: owner, languageSpace=language,
        symbolSpace=SimpleNamespace(ltm_store=store), when_time=inputs['when_time'], when_encoding=encoder())
    __import__('clock_fixtures').install(model, int(inputs['when_time']))
    program = Reading(**inputs['program'])
    strata = SentenceStrata(**inputs['strata'])
    strata.capture(program, language, 0, after=inputs['after'])

    class ClauseEvaluation(Evaluation):
        def finish(self, *, roots=None):
            closing, self.closing = self.closing, None
            result = super().finish(roots=roots)
            self.closing = closing
            result.clause = with_extensions(closing.finish(), roots)
            return result

    evaluation = ClauseEvaluation(model, program, 0)
    forest = reading(model, program, inputs['field'], 0, inputs['positions'], evaluation=evaluation)
    return evaluation, forest


class StoreReads(Reads):
    def __len__(self):
        return self.count
