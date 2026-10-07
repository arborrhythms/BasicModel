"""Static, exhaustive semantic certificate. No training, sampling or global seed."""
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'bin'))
import numpy as np
from MeaningCodes import sentence_key, identity_bits, certificate
from util import XMLConfig, parse
from Meronomy import word_spans


def audit(sentences, pairs, ones):
    vocabulary=sorted({word for words in sentences for word in words})
    index={word:i for i,word in enumerate(vocabulary)}
    codes=np.zeros((len(sentences),pairs),dtype=np.float32)
    positive=np.zeros((len(vocabulary),pairs),dtype=np.float32)
    weights=np.zeros(len(vocabulary),dtype=np.float32)
    extents=[set() for _ in vocabulary]
    keys=[]
    for row,words in enumerate(sentences):
        key=sentence_key(words); keys.append(key)
        bits=list(identity_bits(key,pairs,ones)); codes[row,bits]=1
        ids=np.array(sorted({index[word] for word in words}),dtype=np.int32)
        weight=1/(len(sentences)-row)
        positive[ids[:,None],bits]+=weight
        weights[ids]+=weight
        for word in ids:extents[word].add(row)
    positive/=np.maximum(weights[:,None],np.finfo(np.float32).tiny)
    means=np.concatenate((positive,np.zeros_like(positive)),1)
    report=certificate(means,codes,extents)
    report.update(ones=ones,distinct_sentence_keys=len(set(keys)),
        sentence_keys_sha256=hashlib.sha256(b''.join(keys)).hexdigest(),
        postings=sum(map(len,extents)), recency='1/(1+age); one stored occurrence per admitted sentence',
        vocabulary_sha256=hashlib.sha256(b'\0'.join(vocabulary)).hexdigest())
    if len(vocabulary)<10:
        report.update(vocabulary=[w.decode() for w in vocabulary],meanings=means.tolist(),codes=codes.tolist())
    return report


def main():
    reports={}
    gate=[s.encode().split() for s in ['hello world','hello there','loving world','loving there']]
    for name in ['XOR_grammar.xml','MM_grammar.xml']:
        cfg=XMLConfig(str(ROOT/'data'/name),str(ROOT/'data/model.xml'))
        pairs=(int(cfg.data['ConceptualSpace']['nDim'])-int(cfg.data['PartSpace']['nDim']))//2
        report=audit(gate,pairs,int(cfg.data['ConceptualSpace']['meaningOnes']))
        assert report['exact'] and report['distinct_meanings']==4
        reports[name]=report
    (HERE/'semantic-gates.json').write_text(json.dumps(reports,indent=2)+'\n')
    print('Grammar certificates exact; 4 distinct meanings each',flush=True)
    from data import iter_documents, _file_fingerprint
    cfg=XMLConfig(str(ROOT/'data/BasicModel.xml'),str(ROOT/'data/model.xml'))
    settings=cfg.data['architecture']['data']
    paths=sorted((ROOT/settings['shardDir']).glob('shard_*.parquet'))[:int(settings['numShards'])]
    assert len(paths)==int(settings['numShards'])
    sentences=[]; excluded=0
    for doc in iter_documents([str(p) for p in paths],max_docs=int(settings['maxDocs'])):
        for sentence,_ in parse(doc,lex='sentences'):
            raw=sentence.encode('ascii',errors='replace');spans=word_spans(raw)
            if len(spans)>int(settings['maxSentenceWords']):excluded+=1;continue
            if sentence.strip():sentences.append([raw[a:b] for a,b in spans])
    print(f'Corpus: {len(sentences)} rows; running exhaustive pair counts',flush=True)
    start=time.monotonic()
    report=audit(sentences,448,int(cfg.data['ConceptualSpace']['meaningOnes']))
    report.update(seconds=time.monotonic()-start,diagnostic_only=True,
        corpus=dict(shards=[_file_fingerprint(str(p)) for p in paths],max_docs=int(settings['maxDocs']),
        excluded_sentences=excluded,lexer='existing sentence splitter and ASCII letter runs, all document splits'),
        scope='static content-addressed corpus census, not a resident production store or a training run')
    (HERE/'semantic-basicmodel.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['words','sentences','checks_per_case','errors','false_membership_rates','seconds']}),flush=True)

if __name__=='__main__':main()
