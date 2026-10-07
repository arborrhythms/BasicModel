"""Static identity audits; no model training or global RNG seed.

Native vocabularies and witness fixtures are separate banks. The dictionary
sample is exactly the existing identity toy's declared sample, not a gate.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'bin'))
import numpy as np
import torch
from Layers import RadixLayer
from WordIdentity import WordIdentity, base_atoms
from util import XMLConfig

WITNESSES = ['bana','banana','cat','concat','aba','ababa','an','and','ant',
             'circus','cursic','cirrus','calaba','cabala','aaaaaa','aaaaaaa']
PAIRS = [('bana','banana'),('cat','concat'),('aba','ababa')]
GATE = ['hello','world','loving','there']


def make_bank(configuration, capacity):
    cfg = XMLConfig(str(ROOT/'data'/configuration), str(ROOT/'data/model.xml'))
    ps = cfg.data['PartSpace']
    d, s, length, dense = [int(ps[k]) for k in
        ('identityPairDim','identityOnes','identityLengthDim','identityBindingDim')]
    store = RadixLayer(d+length, initial_cap=capacity)
    store.identity = WordIdentity(store, d,s,length,dense)
    return store.identity


def census(configuration, vocabulary, scope):
    bank = make_bank(configuration, max(512, len(vocabulary)*3))
    rng = torch.get_rng_state().clone()
    initial = {}
    for word in vocabulary:
        raw = word.encode()
        initial.setdefault(bank.key(bank.form(raw)).hex(), []).append(word)
        bank.admit(raw)
    result = bank.audit()
    result.update(configuration=configuration, vocabulary_scope=scope,
                  before_mint_collisions=[g for g in initial.values() if len(g)>1],
                  atom_admission_global_rng_unchanged=torch.equal(rng,torch.get_rng_state()),
                  projection_sha256=hashlib.sha256(bank.projection.cpu().numpy().tobytes()).hexdigest())
    assert not result['collisions'] and not result['containment_violations'] and not result['reconstruction_errors']
    assert result['atom_admission_global_rng_unchanged']
    return result, bank


def configured_basic_vocabulary():
    """Census the configured local corpus with the existing sentence/word lexer."""
    from data import iter_documents, _file_fingerprint
    from util import parse
    from Meronomy import word_spans
    cfg = XMLConfig(str(ROOT/'data/BasicModel.xml'), str(ROOT/'data/model.xml'))
    settings = cfg.data['architecture']['data']
    paths = sorted((ROOT/settings['shardDir']).glob('shard_*.parquet'))[:int(settings['numShards'])]
    assert len(paths) == int(settings['numShards']), 'configured corpus must be present; do not download or substitute'
    words, admitted, excluded = set(), 0, 0
    for doc in iter_documents([str(p) for p in paths], max_docs=int(settings['maxDocs'])):
        for sentence, _ in parse(doc, lex='sentences'):
            raw = sentence.encode('ascii', errors='replace')
            spans = word_spans(raw)
            if len(spans) > int(settings['maxSentenceWords']):
                excluded += 1
                continue
            admitted += bool(sentence.strip())
            words.update(raw[a:b].decode('ascii') for a,b in spans)
    return sorted(words), dict(shards=[_file_fingerprint(str(p)) for p in paths],
        max_docs=int(settings['maxDocs']), admitted_sentences=admitted, excluded_sentences=excluded,
        lexer='existing sentence splitter and ASCII letter runs, all document splits',
        configured_part_capacity=int(cfg.data['PartSpace']['nVectors']))


def main():
    torch.set_num_threads(1)
    result = dict(status='static, no trainings', configurations=[])
    basic_words, corpus = configured_basic_vocabulary()
    vocabularies = [('XOR_grammar.xml',GATE,'four XOR gate words; sum control shares this vocabulary'),
                    ('MM_grammar.xml',GATE,'four grammar fixture words'),
                    ('BasicModel.xml',basic_words,
                     'configured first 2,000 local FineWeb documents, all admitted letter words, all splits; static census only')]
    for config, words, scope in vocabularies:
        native, bank = census(config, words, scope)
        witnesses, witness_bank = census(config,WITNESSES,'separate required witness bank; not appended to gate inputs')
        witnesses['required_order'] = [dict(words=[a,b],parts=set(base_atoms(a.encode())) < set(base_atoms(b.encode())),
            form=bool((witness_bank.form(a.encode()) <= witness_bank.form(b.encode())).all())) for a,b in PAIRS]
        witnesses['not_comparable'] = [dict(words=['an',b],parts=False,missing_atom='n#') for b in ('and','ant')]
        assert all(x['parts'] and x['form'] for x in witnesses['required_order'])
        assert torch.equal(bank.projection,witness_bank.projection)
        witnesses['lengths_1_through_32'] = []
        for length in range(1,33):
            word = b'a'*length
            witness_bank.admit(word)
            witnesses['lengths_1_through_32'].append(dict(length=length,
                exact=witness_bank.form(word)[64:].tolist()==[1.]*length+[0.]*(32-length)))
        assert all(row['exact'] for row in witnesses['lengths_1_through_32'])
        native['native_atom_and_word_rows'] = len(bank.store)
        if config == 'BasicModel.xml':
            native['corpus'] = corpus
            native['fits_declared_part_capacity'] = len(bank.store) <= corpus['configured_part_capacity']
        elif words == GATE:
            from Language import ConjunctionLayer, DisjunctionLayer
            projected = {w: bank.binding(bank.form(w.encode())) for w in GATE}
            pairs = [('hello','world'),('hello','there'),('loving','world'),('loving','there')]
            native['projected_gate_roots'] = {}
            for name, kernel in [('conjunction',ConjunctionLayer()),('disjunction',DisjunctionLayer())]:
                roots = torch.stack([kernel.compose(projected[a],projected[b]) for a,b in pairs])
                design = torch.cat((roots.double(),torch.ones(4,1,dtype=torch.float64)),-1)
                target = torch.tensor([0.,1.,1.,0.],dtype=torch.float64)
                fit = design @ (torch.linalg.pinv(design) @ target)
                native['projected_gate_roots'][name] = dict(norms=roots.norm(dim=-1).tolist(),
                    affine_rank=int(torch.linalg.matrix_rank(design)),
                    optimal_affine_mse=float((fit-target).square().mean()))
                assert (roots.norm(dim=-1)>0).all()
        result['configurations'].append(dict(native=native,witnesses=witnesses))
    result['numeric_MM_xor'] = dict(vocabulary=[], applicable=False, reason='numeric input has no letters')
    words = [w.strip().lower() for w in open('/usr/share/dict/words') if w.strip().isalpha()]
    words = [w for w in words if 3 <= len(w) <= 12]
    sample = list(dict.fromkeys(np.random.default_rng(0).choice(words,20000,replace=False).tolist()))
    stress,_ = census('BasicModel.xml',sample,'identity toy dictionary sample, 19,986 unique words from 20,000 draws')
    result['dictionary_stress'] = stress
    result['source_sha256'] = hashlib.sha256((ROOT/'bin/WordIdentity.py').read_bytes()).hexdigest()
    (HERE/'form-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(configurations=len(result['configurations']), dictionary_words=len(sample),
        dictionary_before=len(stress['before_mint_collisions']), dictionary_after=stress['collisions'],
        dictionary_mints=stress['mints']),indent=2),flush=True)


if __name__ == '__main__':
    main()
