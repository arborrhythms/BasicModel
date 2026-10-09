"""Development reserve-size probe, not an ordinary-path certificate."""
from pathlib import Path
import gc,json,sys,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import torch
from test_item6_2_thinking import world
from Layers import TernaryTruthStore
from QueryWork import QueryWorkBudget
from ThoughtReferences import question

def main():
    torch.set_num_threads(1)
    model,registry,_,(a,b,c)=world()
    facts=[registry.form('part',left,right,mode='assertive') for left,right in ((a,b),(b,c))]
    goal=question(registry.form('isPart',a,c))
    reports=[]
    for capacity in (8192,32768,1048576):
        store=TernaryTruthStore(goal.roles.shape[-1],capacity=capacity)
        model.symbolSpace.ltm_store=store
        for fact in facts:store.append_meaning(fact,kind='fact',rel_type=store.REL_PARTOF,evidence=(1.,0.),trust=1.)
        samples={key:[] for key in ('menu','score','query')}
        work_counts=[]
        with torch.no_grad(),model._query_boundary_scope((0,)):
            for _ in range(25):
                start=time.perf_counter();menu=registry.controller_candidates(goal,goal,goal)
                samples['menu'].append(time.perf_counter()-start)
                start=time.perf_counter();logits=model.shared_grammar.thought_logits(goal,tuple(item.request for item in menu))
                samples['score'].append(time.perf_counter()-start)
                work=QueryWorkBudget(32)
                context=model._thought_grammar_context(goal,row=0,work=work,continuation=None)
                start=time.perf_counter();value=context.ltm.relation_evidence('part',a,b,max_records=32,work=work)
                samples['query'].append(time.perf_counter()-start)
                work_counts.append(dict(work.counts))
        reports.append(dict(capacity=capacity,occupied=len(store),menu_size=len(menu),
            seconds_per_call={key:sum(values)/len(values) for key,values in samples.items()},
            samples_seconds=samples,query_work_counts=work_counts))
        del store
        gc.collect()
    result=dict(kind='development_microprofile',seed=None,reads=25,results=reports,
        scope='Same constructed two-row world and chooser parameters, three reserved LTM capacities; allocation excluded. This supplements the ordinary development episode profiles, not their certificates.')
    (HERE/'capacity-profile.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{key:value for key,value in row.items() if key in ('capacity','occupied','menu_size','seconds_per_call')} for row in reports]))
if __name__=='__main__':main()
