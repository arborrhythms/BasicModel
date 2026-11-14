"""Freeze study tasks and controls from authored frames; no model or network calls."""
import hashlib
import itertools
import json
from pathlib import Path
import random
import re

from game_corpus import render


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path,value):
    text=json.dumps(value,sort_keys=True,indent=1)+'\n'
    path.write_text(text)
    return sha(text.encode())


def freeze(recipe_path,output,*,seed):
    recipe_path,output=Path(recipe_path),Path(output)
    recipe=json.loads(recipe_path.read_text())
    output.mkdir(parents=True,exist_ok=True)
    rng=random.Random(seed)
    train,heldout=[],[]
    for family,frame in recipe['families'].items():
        choices={split:[] for split in ('train','test')}
        slots=list(frame['slots'])
        for values in itertools.product(*(recipe['fillers'][frame['slots'][slot]] for slot in slots)):
            assignment=dict(zip(slots,values))
            if any(len({json.dumps(assignment[slot],sort_keys=True) for slot in group})<len(group)
                   for group in frame.get('distinct',())):
                continue
            texts=[render(text,assignment) for text in frame['sentences']]
            key=sha('\n'.join(texts).encode())
            # A first context cannot enter both training and held-out controls.
            split='train' if int(sha(' '.join(texts[0].lower().split()).encode()),16)%10<8 else 'test'
            choices[split].append(dict(key=family+'-'+key[:16],family=family,
                sentences=texts, question=frame.get('question'),
                answer=None if frame.get('answer') is None else render(frame['answer'],assignment),
                expectations=frame.get('expectations',{})))
        for split,destination in (('train',train),('test',heldout)):
            unique={row['key']:row for row in choices[split]}
            candidates=sorted(unique.values(),key=lambda row:row['key'])
            count=recipe['counts'][split]
            if len(candidates)<count:
                raise ValueError(f'{family}/{split}: {len(candidates)} candidates, {count} required')
            destination.extend(sorted(rng.sample(candidates,count),key=lambda row:row['key']))
    words=sorted({word for row in train+heldout for text in row['sentences']+[row['answer'] or '']
                  for word in re.findall(r'\w+',text)})
    alphabet=recipe['rename_alphabet']
    def opaque(index):
        value=''
        while True:
            value=alphabet[index%len(alphabet)]+value
            index=index//len(alphabet)-1
            if index<0:return recipe['rename_prefix']+value
    renaming={word:opaque(i) for i,word in enumerate(words)}
    def replace(text,mapping):
        return re.sub(r'\w+',lambda m:mapping.get(m.group(0),m.group(0)),text)
    for row in train+heldout:
        shuffled=[]
        determiners=[]
        for text in row['sentences']:
            tokens=re.findall(r'\w+',text);rng.shuffle(tokens);iterator=iter(tokens)
            shuffled.append(re.sub(r'\w+',lambda _:next(iterator),text))
            markers=[word for word in re.findall(r'\w+',text) if word in recipe['determiners']]
            rng.shuffle(markers);iterator=iter(markers)
            determiners.append(re.sub(r'\w+',lambda m:next(iterator) if m.group(0) in recipe['determiners'] else m.group(0),text))
        row['controls']=dict(shuffled_order=shuffled,
            renamed_vocabulary=[replace(text,renaming) for text in row['sentences']],
            renamed_answer=None if row['answer'] is None else replace(row['answer'],renaming),
            shuffled_determiners=determiners)
    files={}
    for name,rows in [('training',train),('learning-heldout',heldout),
                      ('causal-heldout',[row for row in heldout if row['family'] in recipe['causal_families']])]:
        path=output/(name+'.json')
        files[name]=dict(path=path.name,sha256=write_json(path,dict(schema=1,rows=rows)),rows=len(rows))
    manifest=dict(schema=1,seed=seed,recipe_sha256=sha(recipe_path.read_bytes()),
        generator='bin/study_corpus.py',command=f'python bin/study_corpus.py --recipe data/eval/studies/recipe.json --output data/eval/studies --seed {seed}',
        split_rule='first normalized context sha256 modulo ten; 0 to 7 train, 8 to 9 held out',
        files=files,renaming=renaming,learning_seeds=[0,1,2],causal_tasks=256)
    if files['causal-heldout']['rows']!=256:
        raise ValueError('causal study requires exactly 256 frozen tasks')
    write_json(output/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',required=True,type=int)
    args=parser.parse_args()
    print(json.dumps(freeze(args.recipe,args.output,seed=args.seed)['files']))
