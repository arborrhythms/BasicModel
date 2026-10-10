"""Item 6 part 2: text lessons and separate, evaluator-only identity labels.

Generate with ``PYTHONPATH=test python test/identity_corpus.py``. No identity,
role, mention offset, candidate ordering or target is passed to the learner.
The small finite world has four kinds and exclusive colour/size properties;
its assumptions describe teaching data, never production grammar rules.
"""
from collections import Counter
from itertools import permutations, product
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/identity_from_data'
KINDS = ('cat', 'dog', 'bird', 'horse')
PROPERTIES = ('black', 'white', 'small', 'large')
VERBS = ('runs', 'sleeps', 'waits', 'moves')
FOLLOW = dict(zip(KINDS, ('purred', 'barked', 'chirped', 'neighed')))


class Document:
    def __init__(self, split, family, stage, **tags):
        self.split, self.family, self.stage = split, family, stage
        self.tags, self.sentences, self.mentions = tags, [], []
        self.probe = None
        self.row_pair = None

    def add(self, text, mentions=()):
        sid = len(self.sentences)
        self.sentences.append(text)
        ids = []
        for word, entity, role in mentions:
            mid = f'm{len(self.mentions)}'
            self.mentions.append(dict(id=mid, sentence=sid, word=word,
                form=text.split()[word], entity=entity, role=role))
            ids.append(mid)
        return ids

    def query(self, text, target, *, informative=True, role_by_entity=None):
        prior = list(self.mentions)
        entities = list(dict.fromkeys(m['entity'] for m in prior))
        last = {entity: next(m for m in reversed(prior) if m['entity'] == entity)
                for entity in entities}
        recent = sorted(entities, key=lambda e: (last[e]['sentence'], last[e]['word']), reverse=True)
        query, = self.add(text, [(0, target, 'subject')])
        self.probe = dict(mention=query, candidates=[last[e]['id'] for e in entities],
            target=last[target]['id'], informative=informative,
            candidate_count=len(entities), recency_rank=recent.index(target),
            role=(role_by_entity or {}).get(target, last[target]['role']),
            first_mention_rank=entities.index(target))

    def finish(self, number):
        # IDs are opaque, document-scoped addresses, not semantic features.
        key = hashlib.sha256(f'identity-v2:{number}'.encode()).hexdigest()[:20]
        public = dict(id=key, sentences=self.sentences)
        gold = dict(id=key, split=self.split, family=self.family, stage=self.stage,
                    tags=self.tags, mentions=self.mentions, probe=self.probe, row_pair=self.row_pair)
        return public, gold


def build():
    docs = []
    # Independently recurring singleton kinds, properties and predicates.
    for repeat, atom in product(range(4), (*KINDS, *PROPERTIES, *VERBS, *FOLLOW.values())):
        d = Document('train', 'singleton', 1, repeat=repeat, atoms=[atom], object_count=int(atom in KINDS))
        d.add(atom + ' .'); docs.append(d)
    # One object, then two, then three. Keep all kinds/properties/verbs alive
    # in every stage. Latin rotations cover every object/property/verb pair.
    for count in (1, 2, 3):
        for start, prop, verb in product(range(4), range(4), VERBS):
            kinds = [KINDS[(start+i) % 4] for i in range(count)]
            props = [PROPERTIES[(prop+i) % 4] for i in range(count)]
            phrases = [f'a {p} {k}' for p, k in zip(props, kinds)]
            text = (' and '.join(phrases) + ' ' + verb + ' .')
            d = Document('train', 'factorial_support', 1, object_count=count,
                atoms=kinds+props+[verb], kinds=kinds, properties=props, verb=verb)
            d.add(text, [(4*i+2, f'e{i}', 'subject') for i in range(count)])
            docs.append(d)
        # Singletons remain present when the maximum support grows.
        for atom in (*KINDS, *PROPERTIES, *VERBS):
            d = Document('train', 'support_rehearsal', 1, object_count=count, atoms=[atom])
            d.add(atom + ' .'); docs.append(d)
    # Deliberately confounded comparison stream; never mixed into main train.
    for repeat, kind, verb in product(range(4), KINDS, VERBS):
        prop = PROPERTIES[KINDS.index(kind)]
        d = Document('confounded_train', 'confounded', 1, object_count=1,
            atoms=[kind, prop, verb], kinds=[kind], properties=[prop], verb=verb)
        d.add(f'a {prop} {kind} {verb} .', [(2, 'e0', 'subject')]); docs.append(d)
    # Known predictive kind/predicate associations, with independently varied
    # properties. Generic verbs above do not become exclusive kind markers.
    for kind, prop in product(KINDS, PROPERTIES):
        d = Document('train', 'prediction_vocabulary', 5)
        d.add(f'a {prop} {kind} {FOLLOW[kind]} .', [(2, kind, 'subject')]); docs.append(d)

    def pronouns(split, pairs, family='pronoun', biased=False, neither=False):
        for pair in pairs:
            for subject, object_first, refreshed, target in product(range(2), range(2), range(2), range(2)):
                if biased and target != refreshed:
                    continue
                if family == 'reversed_recency' and target == refreshed:
                    continue
                a, b = pair[subject], pair[1-subject]
                roles = {a:'subject', b:'object'}
                d = Document(split, family, 5, word_order='OSV' if object_first else 'SVO')
                if object_first:
                    # Object topicalization changes surface order without
                    # turning the object into a passive grammatical subject.
                    d.add(f'a {b} , a {a} saw .', [(1,b,'object'),(4,a,'subject')])
                else:
                    d.add(f'a {a} saw a {b} .', [(1,a,'subject'),(4,b,'object')])
                # Refresh each candidate equally; grammatical role in the
                # introduction and last-mention recency are independent.
                d.add(f'the {pair[refreshed]} waited .', [(1,pair[refreshed],'subject')])
                pred = 'glittered' if neither else FOLLOW[pair[target]]
                d.query(f'it {pred} .', pair[target], informative=not neither, role_by_entity=roles)
                docs.append(d)
    train_pairs = (('cat','dog'),('bird','horse'))
    held_pairs = (('cat','bird'),('dog','horse'))
    pronouns('train', train_pairs)
    pronouns('eval', held_pairs)
    pronouns('eval', held_pairs, 'neither_candidate', neither=True)
    pronouns('biased_train', train_pairs, 'recency_biased', biased=True)
    pronouns('reversed_eval', held_pairs, 'reversed_recency')
    for split, groups in (('train', (('cat','dog','bird'),)), ('eval', (('cat','dog','horse'),))):
        for order in permutations(groups[0]):
            for recent, target in product(range(3), range(3)):
                d = Document(split, 'pronoun_three', 5)
                roles = {order[0]:'subject', order[1]:'object', order[2]:'subject'}
                d.add(f'a {order[0]} saw a {order[1]} .', [(1,order[0],'subject'),(4,order[1],'object')])
                d.add(f'a {order[2]} waited .', [(1,order[2],'subject')])
                d.add(f'the {order[recent]} moved .', [(1,order[recent],'subject')])
                d.query(f'it {FOLLOW[order[target]]} .', order[target], role_by_entity=roles)
                docs.append(d)
        for kind in KINDS:
            d = Document(split, 'pronoun_one', 5)
            # Distinct training/evaluation content, same seen vocabulary.
            verb = 'runs' if split=='train' else 'sleeps'
            d.add(f'a {kind} {verb} .', [(1,kind,'subject')])
            d.query(f'it {FOLLOW[kind]} .', kind); docs.append(d)
    # Determiners are 90% reliable, including both first and later mentions.
    # Labels describe continuity in the little world; subsequent properties
    # reveal the identity. No identity label is supplied as a training target.
    for repeat, kind, same in product(range(10), KINDS, (False, True)):
        first_det = 'the' if repeat==9 else 'a'
        det = ('a' if same else 'the') if repeat==9 else ('the' if same else 'a')
        color = 'black' if same else 'white'
        d = Document('train', 'determiner', 5, cue_correct=repeat!=9, expected_rows=1 if same else 2)
        first, = d.add(f'{first_det} black {kind} runs .', [(2,'e0','subject')])
        second, = d.add(f'{det} {color} {kind} sleeps .', [(2,'e0' if same else 'e1','subject')])
        d.add(f'it is {color} .', [(0,'e0' if same else 'e1','subject')])
        d.row_pair = dict(mentions=[first,second], expected=1 if same else 2)
        docs.append(d)
    for kind, same, conflict in product(KINDS, (False,True), (False,True)):
        if same and conflict:
            # Identical content cannot prove that an indefinite repeats the
            # same individual. Do not fabricate an identifiable conflict.
            continue
        det = ('a' if same else 'the') if conflict else ('the' if same else 'a')
        color = 'black' if same else 'white'
        d = Document('eval', 'determiner_conflict' if conflict else 'determiner', 5,
                     cue_correct=not conflict, expected_rows=1 if same else 2)
        first, = d.add(f'a black {kind} waits .', [(2,'e0','subject')])
        second, = d.add(f'{det} {color} {kind} moves .', [(2,'e0' if same else 'e1','subject')])
        d.add(f'it is {color} .', [(0,'e0' if same else 'e1','subject')])
        d.row_pair = dict(mentions=[first,second], expected=1 if same else 2); docs.append(d)
    for split, kind, same in product(('train','eval'), KINDS, (False,True)):
        d = Document(split, 'same_kind', 5, expected_rows=1 if same else 2)
        first, = d.add(f'a {kind} runs .', [(1,'e0','subject')])
        second, = d.add(f'{"the" if same else "a"} {kind} sleeps .', [(1,'e0' if same else 'e1','subject')])
        # Later distinguishing evidence, independent of the earlier verb.
        color = 'black' if split=='train' else 'white'
        d.add(f'the running {kind} is {color} .', [(2,'e0','subject')])
        d.add(f'the sleeping {kind} is {color if same else ("white" if color=="black" else "black")} .',
              [(2,'e0' if same else 'e1','subject')])
        d.row_pair = dict(mentions=[first,second], expected=1 if same else 2); docs.append(d)
    for split, pair in (('train',train_pairs[0]),('eval',held_pairs[0])):
        for order, dark, target_color in product((pair,pair[::-1]), range(2), ('black','white')):
            d = Document(split, 'document_context', 7, dark_kind=pair[dark])
            for kind in order:
                color = 'black' if kind==pair[dark] else 'white'
                d.add(f'a {kind} is {color} .', [(1,kind,'subject')])
            target = pair[dark if target_color=='black' else 1-dark]
            d.query(f'it is {target_color} .', target); docs.append(d)
    return [d.finish(i) for i,d in enumerate(docs)]


def write(destination=DEST):
    destination.mkdir(parents=True, exist_ok=True)
    rows = build()
    paths = {}
    for split in sorted({g['split'] for _,g in rows}):
        selected = [(p,g) for p,g in rows if g['split']==split]
        for suffix, index in (('text',0),('labels',1)):
            path=destination/f'{split}.{suffix}.jsonl'
            path.write_text(''.join(json.dumps(pair[index],sort_keys=True)+'\n' for pair in selected))
            paths[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=dict(schema=1, revision=2, files=paths, documents=len(rows),
        by_split=dict(Counter(g['split'] for _,g in rows)),
        by_family=dict(Counter(g['family'] for _,g in rows)))
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,default=DEST)
    print(json.dumps(write(parser.parse_args().out),indent=2))
