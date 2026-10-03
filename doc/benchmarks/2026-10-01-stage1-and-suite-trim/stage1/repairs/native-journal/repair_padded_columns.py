"""Apply the saved padding probe's correction only after workers finish."""
from pathlib import Path
import ast,difflib,json
h=Path(__file__).resolve().parent
root=h.parents[5]
p=root/'bin/Models.py';before=p.read_text()
a='''        first = torch.cat((torch.ones(B, 1, device=active.device, dtype=torch.bool),
                           sentence_ids[:, 1:] != sentence_ids[:, :-1]), dim=1)
        starts = torch.where(active & first, positions, 0).cummax(dim=1).values
        local = torch.where(active, positions - starts, 0)
'''
b='''        # Count actual words, so an inactive column inside a sentence
        # neither consumes a frame nor makes its next word alias the first.
        ordinal = active.long().cumsum(dim=1) - 1
        owners = sentence_ids.clamp(0, W - 1)
        starts = torch.full_like(positions, W).scatter_reduce(
            1, owners, torch.where(active, ordinal, W), reduce='amin')
        local = torch.where(active, ordinal - starts.gather(1, owners), 0)
'''
assert before.count(a)==1
p.write_text(before.replace(a,b))
after=p.read_text()
(h/'padded-columns-before.py.txt').write_text(before)
(h/'padded-columns-after.py.txt').write_text(after)
(h/'padded-columns.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='before',tofile='after')))
def body(s):
    cls=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name=='BasicModel')
    n=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_sentence_journal_layout')
    return '\n'.join(s.splitlines()[n.lineno-2:n.end_lineno])+'\n'
(h/'padded-columns-old-new.json').write_text(json.dumps(dict(old=body(before),new=body(after)),indent=2)+'\n')
p=root/'test/test_sentence_journal.py'
p.write_text(p.read_text()+'''

def test_inactive_columns_neither_split_a_sentence_nor_alias_its_active_words():
    active = torch.tensor([[True, False, True, True]])
    ids = torch.tensor([[0, -1, 0, 1]])
    columns, width = BasicModel._sentence_journal_layout(active, ids, 4, 28)
    assert width == 3 * 2 + 4
    assert columns[0, :3].tolist() == [0, 1, 2]
    assert columns[0, 6:9].tolist() == [3, 4, 5]
    assert columns[0, 9:12].tolist() == [0, 1, 2]
''')
