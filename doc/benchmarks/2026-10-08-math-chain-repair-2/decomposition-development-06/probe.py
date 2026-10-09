"""Development only: inspect forced ordinary grammar without changing results."""
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test'),str(HERE)]
from math_chain_ordinary import ordinary_model, train_documents
from forced_decomposition_grammar import ForcedDecompositionGrammar
from math_chain_corpus import ChainDocument
from ThoughtReferences import bindings, open_slots


def main():
    folder=HERE/sys.argv[1]
    folder.mkdir(exist_ok=False)
    shutil.copy2(HERE/'forced_decomposition_grammar.py',folder/'grammar.py')
    shutil.copy2(__file__,folder/'probe.py')
    model=ordinary_model(folder,budget=0)
    documents=[ChainDocument(str(i),('y is x plus one.','x is three.',
        'three plus one is four.','what is y ?'),question=3,answer='four',pair=(3,1),
        steps=(('three','one','four'),)) for i in range(2)]
    reports=[]
    def after(model,split,sources,result,observed,observer):
        store=model.symbolSpace.ltm_store
        fields=model._sentence_fields[0]
        report=dict(sources=sources,source_indices=[store.index_of_row(field.row_id) for field in fields],
            words={w:list(model._concept_owner().word_concepts(w)) for w in ('x','y','three','one','four')},rows=[])
        for index in range(len(store)):
            item=store.row(index)
            value=item['meaning']
            report['rows'].append(dict(index=index,kind=item['kind'],row_id=item['row_id'],
                occurrence=item['occurrence'],refs=value.role_refs,native_refs=item['refs'].tolist(),
                role_mask=value.role_mask.tolist(),bindings=bindings(value),pair=item['evidence']))
        reports.append(report)
    try:
        with ForcedDecompositionGrammar(model,open_names=('what',),generic_subjects=('x','three')) as forced:
            train_documents(model,documents,folder,after=after)
        (folder/'native-rows.json').write_text(json.dumps(reports,indent=2)+'\n')
        (folder/'forced-choices.json').write_text(json.dumps(forced.records,indent=2)+'\n')
        print(json.dumps(dict(status='development_only',sentences=8,forced=True)))
    finally:
        model.End()


if __name__=='__main__': main()
