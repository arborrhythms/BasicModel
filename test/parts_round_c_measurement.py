"""Declared-place POS measurements; fixed 32 rounds, no acquisition bar.

Rule 6.7 permits these forced places. The four-word geometry fixture, its
capacity and random stream are unchanged. This script also runs on the A/B
checkpoint: it observes whichever kept-close implementation that source has.
"""
import json
import time
from types import SimpleNamespace as NS

import torch
from argument_fixtures import argument_inventory
from ConceptHierarchy import direction
from PartsOfSpeech import formal_arguments
from test_parts_of_speech import rule


def fixture(names):
    rules = tuple(rule(name, head=2 if name in ('preposition','surface') else 1) for name in names)
    owner, parts, words = argument_inventory(formal_arguments(rules))
    return owner, parts, words, rules


def close(parts, words, rules, item):
    operation, left, right, *opening = item
    actions = [[0,-1,0]] + ([[1,opening[0],0]] if opening else []) + [[0,-1,1],[1,operation,0]]
    program = NS(word_ids=torch.tensor([words[left], words[right]]),
        actions=torch.tensor(actions))
    place_close(parts, program, rules)


def place_close(parts, program, rules):
    """The two kept-close placement steps, in their production order."""
    owner = parts.owner
    owner._model = NS(languageSpace=NS(parts_of_speech=parts,
        _compose_binary_rules=rules, _compose_unary_rules=()))
    rows = [owner._concept_row_of(int(cid)) for cid in program.word_ids if int(cid) >= 0]
    owner._observed_knowing.zero_()
    owner._observed_knowing[0, rows, 0] = 1.
    program.word_rows = torch.tensor(rows)
    placements = []
    parts.observe_kept(program, rules, (), placements=placements)
    owner.similarity_codebook.mereology.observe_close(program)
    for words, occupied in placements: parts._place_groups(words, occupied)


def train(parts, words, rules, schedule):
    start = time.perf_counter()
    for _ in range(32):
        for item in schedule:
            close(parts, words, rules, item)
    return dict(closes=32*len(schedule),seconds=time.perf_counter()-start)


def cosine(owner, left, right):
    return float(direction(owner,left) @ direction(owner,right))


def measure():
    out = dict(protocol='32 fixed rounds per case; four existing fixture words; declared argument places')
    owner, parts, words, rules = fixture(('verb','modifier','preposition'))
    names = ('verb_argin1','verb_argin2','modifier_argin1','preposition_argin2')
    def merge():
        ids = [parts.concepts[name] for name in names]
        return dict(noun_pairs=[cosine(owner,a,b) for i,a in enumerate(ids) for b in ids[i+1:]],
            modifier_pairs=[cosine(owner,a,parts.concepts['modifier_argin2']) for a in ids])
    before = merge()
    timing = train(parts,words,rules,[(op,a,b) for noun in (0,1)
        for op,a,b in ((0,noun,3),(0,3,noun),(1,noun,2),(2,2,noun))])
    out['2_merge'] = dict(before=before,after=merge(),collapse=parts.collapse(),**timing)

    owner, parts, words, rules = fixture(('verb','modifier','surface','bind'))
    def split():
        return dict(words=cosine(owner,words[0],words[1]),
            subject_fits=[float(parts.similarities(word)[parts.index['verb_argin1']]) for word in words[:2]])
    before = split()
    timing = train(parts,words,rules,[(0,0,3,3),(0,1,3),(1,1,2),(2,2,1)])
    out['3_split'] = dict(before=before,after=split(),collapse=parts.collapse(),**timing)

    owner, parts, words, rules = fixture(('modifier','verb'))
    def halves():
        groups = [[parts.concepts[name] for name in parts.arguments if name.startswith(op+'_')]
                  for op in ('modifier','verb')]
        return dict(word_within=[cosine(owner,words[0],words[1]),cosine(owner,words[2],words[3])],
            word_between=[cosine(owner,a,b) for a in words[:2] for b in words[2:]],
            argument_within=[cosine(owner,a,b) for group in groups for i,a in enumerate(group) for b in group[i+1:]],
            argument_between=[cosine(owner,a,b) for a in groups[0] for b in groups[1]])
    before = halves()
    timing = train(parts,words,rules,[(0,0,1),(0,1,0),(1,2,3),(1,3,2)])
    out['4a_both_halves'] = dict(before=before,after=halves(),collapse=parts.collapse(),**timing)

    owner, parts, words, rules = fixture(('verb','surface'))
    train(parts,words,rules,[(0,0,3),(0,3,0),(1,2,0)])
    def novel():
        i = parts.index['verb_argin2']
        return dict(known=float(parts.similarities(words[0])[i]),novel=float(parts.similarities(words[1])[i]))
    before = novel()
    program = NS(word_ids=torch.tensor([words[2],words[1],words[3]]),
        actions=torch.tensor([[0,-1,0],[0,-1,1],[1,1,0],[0,-1,2],[1,0,0]]))
    place_close(parts, program, rules)
    out['6_new_word'] = dict(before=before,after=novel(),object_members=parts.members('verb_argin2'),
                             novel_was_object=words[1] in parts.members('verb_argin2'),collapse=parts.collapse())
    from Expectation import SymbolRead
    argument_rows = torch.tensor([owner._concept_row_of(parts.concepts[name]) for name in parts.arguments])
    expected = torch.zeros(1, len(argument_rows), 2)
    i = parts.index['verb_argin2']
    expected[0, i, 0] = .7
    read = SymbolRead(torch.zeros(1,len(argument_rows),4), expected, 1.,1e-6,'clip',argument_rows)
    prime = parts.expected_arguments(read, expected[:,0])
    fit = parts.similarities(words[1])[i]
    out['5_priming'] = dict(protocol='declared argument expectation, mechanism measurement',
        new_word=words[1], argument='verb_argin2', expected=float(prime[0,i]),
        without=float(fit), with_expected=float(fit+prime[0,i]),
        delta=float(fit+prime[0,i]-fit))
    return out


if __name__ == '__main__':
    print(json.dumps(measure(),indent=2))
