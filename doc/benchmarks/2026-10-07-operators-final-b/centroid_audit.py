"""Exhaustive centroid audit on configured vocabularies and occurrence adjacency."""
import json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'bin')]
import torch
from form_audit import make_bank, GATE
from SymbolCentroid import place
from Occurrence import document_digest, address_key, sentence_key
from util import XMLConfig, parse
from data import iter_documents, _file_fingerprint
from Meronomy import word_spans


def audit(configuration, documents):
    sentences=[]
    for document, contents in documents:
        for position, words in enumerate(contents,1):
            if words:
                sentences.append((address_key(document_digest(document),position,sentence_key(words)),words))
    words=sorted({w for _,s in sentences for w in s})
    bank=make_bank(configuration,max(512,len(words)*3))
    for word in words:bank.admit(word)
    forms={w:bank.form(w) for w in words}
    parts={w:set(bank.words[w]) for w in words}
    adjacency={}
    for address, sentence in sentences:
        for position,pair in enumerate(zip(sentence,sentence[1:])):
            adjacency.setdefault(pair,set()).add((address,position))
    rng=torch.get_rng_state().clone()
    placed,report=place(forms,parts,adjacency)
    report.update(configuration=configuration, distinct_occurrences=len({a for a,_ in sentences}),
        part_weight='distinct native part atoms, including length atoms and collision mints',
        extent='all configured words and all comparable pairs; no sampled cosines',
        rng_unchanged=torch.equal(rng,torch.get_rng_state()))
    assert report['identity_recovered'] and report['after']['violating_pairs']==report['below_lower']==0
    if len(words)<10:report['words_detail']={w.decode():dict(lower=forms[w].tolist(),centroid=placed[w].tolist()) for w in words}
    return report


def main():
    torch.set_num_threads(1)
    before=torch.get_rng_state().clone()
    gate=[(f'xor:{i}',[sentence.encode().split()]) for i,sentence in enumerate(('hello world','hello there','loving world','loving there'))]
    results={c:audit(c,gate) for c in ('XOR_grammar.xml','MM_grammar.xml')}
    cfg=XMLConfig(str(ROOT/'data/BasicModel.xml'),str(ROOT/'data/model.xml'))
    settings=cfg.data['architecture']['data']
    paths=sorted((ROOT/settings['shardDir']).glob('shard_*.parquet'))[:int(settings['numShards'])]
    documents=[]
    for i,doc in enumerate(iter_documents([str(p) for p in paths],max_docs=int(settings['maxDocs']))):
        sentences=[]
        for text,_ in parse(doc,lex='sentences'):
            raw=text.encode('ascii',errors='replace');spans=word_spans(raw)
            sentences.append([raw[a:b] for a,b in spans] if len(spans)<=int(settings['maxSentenceWords']) else [])
        documents.append((('configured-corpus-document',i),sentences))
    results['BasicModel.xml']=audit('BasicModel.xml',documents)
    # RadixLayer construction initializes its storage, while all identity and
    # centroid algorithms use fixed or local codes. Production certificate is
    # independently checked by focused RNG fixtures.
    results['BasicModel.xml']['corpus']=[_file_fingerprint(str(p)) for p in paths]
    (HERE/'centroid-audit.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({c:{k:r[k] for k in ('words','moved','share_moved','cosine_before','cosine_after','before','after','distinct_occurrences')} for c,r in results.items()}),flush=True)

if __name__=='__main__':main()
