"""Stage 4e text, teacher annotations and held-out labels; no runtime reader."""
import argparse
from itertools import product
import json
from pathlib import Path

from identity_corpus import KINDS, PROPERTIES, ROOT

DEST = ROOT / 'data/constituents'


def spans(tree):
    """All nontrivial brackets, including the root; labels are not POS guesses."""
    found = []
    def visit(node):
        if type(node) is int:
            return [node]
        leaves = sum((visit(child) for child in node[1:]), [])
        if len(leaves) > 1:
            found.append(leaves)
        return leaves
    visit(tree)
    return sorted(found)


def corpus():
    rows = []
    for family in ('np', 'intransitive', 'transitive', 'preposition'):
        for kind, distance, definite in product(range(4), (1, 2, 3), (False, True)):
            words, nps = [], []
            def word(form):
                words.append(form)
                return len(words)-1
            def np(k, d, adjective=None):
                start = len(words)
                cue = word('the' if d else 'a')
                if adjective is None:
                    description = word(KINDS[k])
                else:
                    description = ['compound', word(PROPERTIES[adjective]), word(KINDS[k])]
                nps.append(dict(span=list(range(start, len(words))),
                    description=list(range(start+1, len(words))), singular=True))
                return ['definite' if d else 'lower', cue, description]
            other = (kind+distance) % 4
            left = np(kind, definite, other if family in ('np', 'intransitive') else None)
            tree = left
            if family == 'intransitive':
                tree = ['lift', left, word(('runs', 'sleeps')[(kind+distance+definite) % 2])]
            elif family == 'transitive':
                verb = word(('chases', 'sees')[(kind+definite) % 2])
                tree = ['lift', left, ['verb', verb, np(other, not definite)]]
            elif family == 'preposition':
                copula, prep = word('is'), word(('on', 'near')[(kind+definite) % 2])
                tree = ['lift', left, ['surface', copula,
                    ['preposition', prep, np(other, not definite)]]]
            split = 'eval' if distance == 3 else 'train'
            key = f'4e:{family}:{kind}:{distance}:{int(definite)}'
            rows.append((split, dict(id=key, text=' '.join(words)),
                dict(text=' '.join(words), tree=tree, waits=True),
                dict(id=key, family=family, words=words, brackets=spans(tree), nps=nps,
                     unlowered=[])))
    for kind, prop in product(range(4), repeat=2):
        words = [PROPERTIES[prop], KINDS[kind]]
        tree = ['compound', 0, 1]
        split = 'train' if (kind+prop) % 2 == 0 else 'eval'
        key = f'4e:description:{kind}:{prop}'
        rows.append((split, dict(id=key, text=' '.join(words)),
            dict(text=' '.join(words), tree=tree, waits=True),
            dict(id=key, family='description', words=words, brackets=spans(tree),
                 nps=[], unlowered=[[0, 1]])))
    return rows


def write(directory=DEST):
    directory.mkdir(parents=True, exist_ok=True)
    rows = corpus()
    for split in ('train', 'eval'):
        selected = [row for row in rows if row[0] == split]
        for column, name in ((1, 'text'), (3, 'labels')):
            (directory/f'{split}.{name}.jsonl').write_text(''.join(
                json.dumps(row[column])+'\n' for row in selected))
        if split == 'train':
            (directory/'train.lessons.json').write_text(json.dumps(
                [row[2] for row in selected], indent=2)+'\n')


def load(split, directory=DEST):
    texts = [json.loads(line) for line in (directory/f'{split}.text.jsonl').read_text().splitlines()]
    labels = {row['id']: row for line in (directory/f'{split}.labels.jsonl').read_text().splitlines()
              for row in [json.loads(line)]}
    if set(labels) != {row['id'] for row in texts}:
        raise ValueError('constituent texts and grading labels must have matching IDs')
    return texts, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEST)
    write(parser.parse_args().out)
