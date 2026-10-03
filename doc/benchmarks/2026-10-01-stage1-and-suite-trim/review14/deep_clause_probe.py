"""Saved reproduction of LM_5M's completed-clause host recursion limit."""
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'bin'))
import torch
from ClauseJournal import finish_clause
from Understanding import AnswerProgram

def check(unary=False, words=1200):
    rows=[(0,-1,0)]
    if unary:
        rows.extend((2,0,-1) for _ in range(words))
        rows.extend(((0,-1,1),(1,0,-1)))
        words=2
    else:
        for w in range(1,words):rows.extend(((0,-1,w),(1,0,-1)))
    actions=torch.tensor(rows)
    frames=torch.ones(len(rows),3,2)
    frames[-1]=torch.tensor([[4.,5.],[6.,7.],[8.,9.]])
    end=torch.tensor([[8.,9.],[0.,0.],[0.,0.]],requires_grad=True)
    program=AnswerProgram(rows=torch.arange(words),word_rows=torch.arange(words),
        leaves=torch.ones(words,2),activations=torch.ones(words),actions=actions,
        concept_ids=torch.arange(1,words+1),targets=torch.zeros(len(rows),dtype=torch.long),
        end_state=end,operation_values=frames)
    language=SimpleNamespace(_compose_binary_rules=[SimpleNamespace(method_name='sum')],
        _compose_unary_rules=[SimpleNamespace(method_name='not')])
    result=finish_clause(language,program)
    torch.testing.assert_close(result.point,end[0],rtol=0,atol=0)
    torch.testing.assert_close(result.meaning.roles[:2],frames[-1,:2],rtol=0,atol=0)
    assert result.children==() and result.relation is None
    assert result.refs==(-1,words,-1)
    result.point.sum().backward()
    torch.testing.assert_close(end.grad,end.new_tensor([[1.,1.],[0.,0.],[0.,0.]]))

if __name__=='__main__':
    check('--unary' in sys.argv)
    print('PASS: 1,200 operations; exact ended value, roles, references and gradient')
