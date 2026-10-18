"""The declared NP factorial, separate text, questions, lessons and labels."""
from itertools import product
import json

from constituent_corpus import spans, load as load_constituents
from identity_corpus import ROOT, KINDS, PROPERTIES

DEST = ROOT / 'data/noun_phrases'


def corpus():
    rows = []
    for determiner, adjective, noun in product(range(2), range(-1, 4), range(4)):
        words = [('a', 'the')[determiner]]
        if adjective >= 0:
            words.append(PROPERTIES[adjective])
        words.append(KINDS[noun])
        description = ['intersection', 1, 2] if adjective >= 0 else 1
        tree = [('lower', 'definite')[determiner], 0, description]
        heldout = ((adjective-noun) % 4 == 3 if adjective >= 0 else (noun+determiner) % 4 == 3)
        key = f'np:np:{determiner}:{adjective}:{noun}'
        question = (dict(prompt=f'what is {PROPERTIES[adjective]} ?', answer=KINDS[noun])
                    if adjective >= 0 else None)
        rows.append(('eval' if heldout else 'train', dict(id=key, text=' '.join(words)),
            dict(text=' '.join(words), tree=tree, waits=True),
            dict(id=key, family='np', words=words, brackets=spans(tree),
                 nps=[dict(span=list(range(len(words))), description=list(range(1,len(words))), singular=True)],
                 unlowered=[]), question))
    for adjective, noun in product(range(4), repeat=2):
        words = [PROPERTIES[adjective], KINDS[noun]]
        key = f'np:description:{adjective}:{noun}'
        tree = ['intersection', 0, 1]
        rows.append(('eval' if (adjective-noun) % 4 == 3 else 'train', dict(id=key, text=' '.join(words)),
            dict(text=' '.join(words), tree=tree, waits=True),
            dict(id=key, family='description', words=words, brackets=spans(tree), nps=[], unlowered=[[0,1]]),
            dict(prompt=f'what is {PROPERTIES[adjective]} ?', answer=KINDS[noun])))
    return rows


def write():
    DEST.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'eval'):
        rows = [row for row in corpus() if row[0] == split]
        for index, name in ((1, 'text'), (3, 'labels')):
            (DEST/f'{split}.{name}.jsonl').write_text(''.join(json.dumps(row[index])+'\n' for row in rows))
        # Held-out answers remain with the evaluator, never a model call.
        (DEST/f'{split}.questions.json').write_text(json.dumps({row[1]['id']: row[4] for row in rows}, indent=2)+'\n')
        if split == 'train':
            (DEST/'train.lessons.json').write_text(json.dumps([row[2] for row in rows], indent=2)+'\n')


def load(split):
    return load_constituents(split, DEST)


if __name__ == '__main__':
    write()
