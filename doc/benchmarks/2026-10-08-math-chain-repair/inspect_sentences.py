"""Development construction trace; not a declared training or gate attempt."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]
import torch
torch.set_num_threads(1)
from test_math_chain import build_model
from MathChainTraining import present
from ThoughtReferences import bindings, open_slots
import ClauseJournal

config = Path(__file__).with_name('construction.xml')
config.write_text((ROOT/'data/MM_math_chain.xml').read_text().replace(
    '<ltmCapacity>1048576</ltmCapacity>', '<ltmCapacity>8192</ltmCapacity>'))
model = build_model(config)
data = model.inputSpace.data
texts = ['y is x plus two.', 'x is three.', 'what is y ?', 'the answer is five.']
data.train_input, data.train_output = texts, [torch.zeros(1) for _ in texts]
data.text_answers['train'] = [None] * len(texts)
data.source_addresses['train'] = [dict(document=0, sentence=i, split='train') for i in range(len(texts))]
finish = ClauseJournal.finish_clause
def trace(language, program, **kw):
    value = finish(language, program, **kw)
    print('WORDS', program.lexical_forms, flush=True)
    print('ACTIONS', [(kind, language._compose_binary_rules[op].method_name if kind == 1 else
        language._compose_unary_rules[op].method_name if kind == 2 else leaf,
        program.operation_refs[i].tolist()) for i, (kind, op, leaf) in enumerate(program.actions.tolist()) if kind >= 0], flush=True)
    def show(clause, indent=''):
        print(indent, clause.relation, clause.refs, bindings(clause.meaning), flush=True)
        for child in clause.children: show(child, indent+'  ')
    show(value)
    return value
ClauseJournal.finish_clause = trace
def after(model, split, rows, result):
    for fields in model._sentence_fields.values():
        for field in fields:
            if field is not None:
                print('FIELD', rows, 'OPEN', None if field.query is None else open_slots(field.query),
                      'MEANING', field.meaning.role_refs, bindings(field.meaning), flush=True)
try:
    present(model, split='train', after_batch=after)
finally:
    model.End()
