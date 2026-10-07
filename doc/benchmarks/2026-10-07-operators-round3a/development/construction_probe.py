"""Pre-implementation check of the hand-off's literal sparse construction.

This is a specification diagnostic, not a model training or a gate run.
The dictionary sample is the identity toy's unchanged sample. Atom codes
use private CPU generators seeded by SHA-256 of the atom's tagged bytes.
No seed is set on any global generator and no alternative seed is tried.
"""
import hashlib
import json
import platform
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

D, S = 64, 3


def pieces(word, n):
    text = '#' + word + '#'
    return tuple(text[i:i+n] for i in range(len(text)-n+1))


def atoms(word):
    return {b'pair\0' + p.encode() for p in pieces(word, 2)} | {
        b'len\0' + n.to_bytes(8, 'big') for n in range(1, len(word)+1)}


@lru_cache(None)
def code(atom):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int.from_bytes(hashlib.sha256(atom).digest()[:8], 'little'))
    positions = torch.randperm(D, generator=generator)[:S].tolist()
    return sum(1 << i for i in positions)


def join(parts):
    value = 0
    for atom in parts:
        value |= code(atom)
    return value


def main():
    before = torch.get_rng_state().clone()
    words = [w.strip().lower() for w in open('/usr/share/dict/words') if w.strip().isalpha()]
    words = [w for w in words if 3 <= len(w) <= 12]
    # Reproduce the pre-existing toy sample, not a gate training seed.
    sample = list(dict.fromkeys(np.random.default_rng(0).choice(words, 20000, replace=False).tolist()))
    groups = {}
    for word in sample:
        groups.setdefault(join(atoms(word)), []).append(word)
    collisions = [v for v in groups.values() if len(v) > 1]
    triple_checks = []
    for group in collisions:
        for word in group[1:]:
            first = group[0]
            differing = next(((a, b) for a, b in zip(pieces(first, 3), pieces(word, 3)) if a != b), None)
            after = None if differing is None else [
                join(atoms(w) | {b'triple\0' + t.encode()})
                for w, t in zip((first, word), differing)]
            triple_checks.append(dict(words=[first, word], first_differing_triples=differing,
                                      resolved=after is not None and after[0] != after[1]))
    native = ['hello', 'world', 'loving', 'there']
    witnesses = ['an', 'and', 'ant', 'bana', 'banana', 'circus', 'cursic', 'cirrus', 'calaba', 'cabala']
    repeat = {}
    saturation = None
    for n in range(1, 257):
        word = 'a'*n
        form = join(atoms(word))
        if form in repeat:
            saturation = dict(lengths=[repeat[form], n], same_triple_set=set(pieces('a'*repeat[form],3)) == set(pieces(word,3)),
                              active_coordinates=form.bit_count())
            break
        repeat[form] = n
    repeat_words = ['a'*n for n in saturation['lengths']]
    extra_length = b'len\0' + saturation['lengths'][1].to_bytes(8, 'big')
    triple_pair = [('calaba', 'cal'), ('cabala', 'cab')]
    positional = next((a, b) for a, b in zip(*(pieces(w, 3) for w in repeat_words)) if a != b)
    repeat_repaired = [join(atoms(w) | {b'triple\0'+t.encode()})
                       for w, t in zip(repeat_words, positional)]
    proofs = dict(
        length=dict(words=repeat_words,
                    forms=[hex(join(atoms(w))) for w in repeat_words],
                    extra_atom=extra_length.hex(),
                    extra_code=hex(code(extra_length)),
                    extra_bit_positions=[i for i in range(D) if code(extra_length) >> i & 1],
                    new_bits=hex(code(extra_length) & ~join(atoms(repeat_words[0]))),
                    triple_sets=[sorted(set(pieces(w, 3))) for w in repeat_words],
                    first_differing_positional_triples=positional,
                    positional_repair_forms=[hex(v) for v in repeat_repaired],
                    positional_repair_resolves=repeat_repaired[0] != repeat_repaired[1]),
        triples=[dict(word=w, triple=t, before=hex(join(atoms(w))),
                      atom_code=hex(code(b'triple\0'+t.encode())),
                      after=hex(join(atoms(w) | {b'triple\0'+t.encode()})))
                 for w, t in triple_pair])
    assert proofs['length']['forms'][0] == proofs['length']['forms'][1]
    assert proofs['length']['triple_sets'][0] == proofs['length']['triple_sets'][1]
    assert proofs['triples'][0]['after'] == proofs['triples'][1]['after']
    assert atoms('an') - atoms('and') == {b'pair\0n#'}
    assert atoms('bana') < atoms('banana')
    assert all(join(atoms(a)) & ~join(atoms(b)) == 0
               for a in witnesses for b in witnesses if atoms(a) <= atoms(b))
    result = dict(status='specification diagnostic; no model/gate training', D=D, s=S,
                  atom_seed='little-endian first 8 SHA256 bytes; private torch CPU generator; randperm first s',
                  environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dictionary_sha256=hashlib.sha256(Path('/usr/share/dict/words').read_bytes()).hexdigest(),
                  global_rng_unchanged=torch.equal(before, torch.get_rng_state()),
                  dictionary_words=len(sample), collisions=collisions, triple_checks=triple_checks,
                  gate_forms={w:hex(join(atoms(w))) for w in native},
                  witness_forms={w:hex(join(atoms(w))) for w in witnesses},
                  gate_strict_part_containment=[(a,b) for a in native for b in native if atoms(a) < atoms(b)],
                  witness_strict_part_containment=[(a,b) for a in witnesses for b in witnesses if atoms(a) < atoms(b)],
                  repeated_letter_first_collision=saturation, counterexample_proofs=proofs)
    path = Path(__file__).with_suffix('.json')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(dictionary_words=len(sample), collision_groups=len(collisions),
                          unresolved_first_triple=sum(not r['resolved'] for r in triple_checks),
                          gate_distinct=len(set(result['gate_forms'].values())),
                          gate_containment=result['gate_strict_part_containment'],
                          witness_containment=result['witness_strict_part_containment'],
                          repeated_letter_first_collision=saturation,
                          global_rng_unchanged=result['global_rng_unchanged']), indent=2))


if __name__ == '__main__':
    main()
