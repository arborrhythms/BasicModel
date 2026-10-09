"""FORCED diagnostic, not a passing decomposition certificate or learning run.

Force the requested first query using native operands from the ordinary question.
The actual paired episode, scorer, writer, observer and explore suffix run intact.
No result, binding, write, evidence or keep decision is patched.
"""
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test'),str(HERE)]
from math_chain_ordinary import ordinary_model, train_documents, episode_observer
from forced_decomposition_grammar import ForcedDecompositionGrammar
from math_chain_corpus import ChainDocument
from ThoughtReferences import bindings, open_slots
from ThoughtStream import query_pattern
from Queries import ThoughtOperationCandidate
from BindingAnswers import matches
import ThoughtStream


def main():
    folder=HERE/'decomposition-first-query-diagnostic'
    folder.mkdir(exist_ok=False)
    shutil.copy2(__file__,folder/'diagnostic.py')
    shutil.copy2(HERE/'forced_decomposition_grammar.py',folder/'grammar.py')
    model=ordinary_model(folder,budget=256)
    documents=[ChainDocument(str(i),('x is three.','three plus one is four.',
        'y is x plus one.','what is y ?'),question=3,answer='four',pair=(3,1),
        steps=(('three','one','four'),)) for i in range(2)]
    menu=ThoughtStream.candidates
    chooser=model._selected_thought_chooser(None)
    logits=chooser.thought_logits
    episodes=[]
    def forced_operands(registry,root,active,current,records,descriptions=()):
        actions=menu(registry,root,active,current,records,descriptions)
        if ('referent',0) not in open_slots(root):
            return actions
        pattern=query_pattern(root,'part',root.role_refs[2],None,values=root.roles[2])
        request=registry.form('query',pattern)
        return (*actions,ThoughtOperationCandidate(registry.operation_spec('query'),request,()))
    def preferred(active,requests):
        value=logits(active,requests)
        def bonus(request):
            if request is None: return 100.
            if len(request.constituents)==1:
                data=bindings(request.constituents[0])
                if data.get('_query_relation')=='part': return 50.
            return 0.
        return value+value.new_tensor([bonus(request) for request in requests])
    def episode(original,model,meaning,**kwargs):
        result=original(model,meaning,**kwargs)
        episodes.append(dict(row=kwargs['row'],before=meaning.role_refs,
            after=result.meaning.role_refs,before_open=open_slots(meaning),
            after_open=open_slots(result.meaning),bound_roles=bindings(result.meaning).get('_bound_roles'),
            answers={word:matches(model,result.meaning,word) for word in ('x','y','three','four')},
            operations=[record.operation for record in result.records if record.kind=='thought'],
            kinds=[record.kind for record in result.records],
            query_results=[dict(request=record.result.request.role_refs,
                frames=[dict(occurrence=frame['occurrence'],refs=frame['meaning'].role_refs)
                        for frame in record.result.evidence.get('frames',())])
                for record in result.records if record.result is not None and record.result.semantic_id=='query']))
        return result
    try:
        with ForcedDecompositionGrammar(model,open_names=('what',)) as forced, \
                patch.object(ThoughtStream,'candidates',forced_operands), \
                patch.object(chooser,'thought_logits',preferred):
            train_documents(model,documents,folder,episode=episode)
        (folder/'diagnostic.json').write_text(json.dumps(dict(forced=True,
            passing_certificate=False,source_learner_unchanged=True,episodes=episodes,
            forcing='Grammar choices and query operand/operation preference only; live masked alternatives and keep costs.'),indent=2)+'\n')
        print(json.dumps(episodes))
    finally:
        model.End()


if __name__=='__main__': main()
