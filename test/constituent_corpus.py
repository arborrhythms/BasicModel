"""Stage 4e text, teacher annotations and held-out labels; no runtime reader."""
import argparse
from itertools import product
import json
from pathlib import Path

from identity_corpus import KINDS, PROPERTIES, ROOT

DEST = ROOT / 'data/constituents'
STAGE3_ARTIFACT = ROOT / 'output/stage4e-student-order-20261017/order.ckpt'
FAMILY_ORDER = ('description', 'np', 'intransitive', 'transitive', 'preposition')
CASE_VARIANTS = ('verb_form', 'separate_leaf', 'noun_form',
                 'one_leaf_two_selections', 'two_leaves', 'unmarked')


def case_corpus(variant):
    """Six surface placements; labels stay with the evaluator, never a lesson.

    The same three case answers and held-out factorial apply to every
    placement. The unmarked form supplies no phase evidence. The extra
    selection is a second location restriction on the same head.
    """
    if variant not in CASE_VARIANTS:
        raise ValueError('unknown case-corpus placement')
    rows = []
    for i, subject in enumerate(('ava', 'oro', 'emi', 'ulu')):
        for j, head in enumerate(('dak', 'fep')):
            for phase, marker in enumerate(('za', 'zi', 'zu')):
                if variant == 'verb_form':
                    phrase = head+marker
                elif variant == 'separate_leaf':
                    phrase = marker+' '+head
                elif variant == 'noun_form':
                    phrase = ('nav', 'vem')[j]+marker
                elif variant == 'one_leaf_two_selections':
                    phrase = marker+'lo '+head
                elif variant == 'two_leaves':
                    phrase = marker+' lo '+head
                else:
                    phrase = head
                rows.append(dict(id=f'{variant}:{i}:{j}:{phase}', subject=subject,
                    sentences=[subject+' '+('bem', 'tek')[j], subject+' '+phrase],
                    prompt='qi '+subject+' ?',
                    answer='un' if variant == 'unmarked' else ('pa', 'pc', 'pz')[phase],
                    heldout=(i+j)%4 == 3,
                    labels=dict(case=phase, head='noun' if variant == 'noun_form' else 'verb',
                        selections=0 if variant == 'unmarked' else
                                   2 if variant in ('one_leaf_two_selections', 'two_leaves') else 1,
                        nested=variant in ('noun_form', 'verb_form'))))
    return rows


def curriculum_order(texts, rng):
    """The declared family progression, counterbalanced cases within each."""
    families = [row['id'].split(':')[1] for row in texts]
    if set(families) != set(FAMILY_ORDER):
        raise ValueError('stage 4e requires all five declared families')
    return [int(index) for family in FAMILY_ORDER
            for index in rng.permutation([i for i, name in enumerate(families) if name == family])]


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
                    description = ['intersection', word(PROPERTIES[adjective]), word(KINDS[k])]
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
        tree = ['intersection', 0, 1]
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
