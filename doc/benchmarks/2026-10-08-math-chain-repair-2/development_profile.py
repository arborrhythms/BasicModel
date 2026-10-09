"""A single development batch profile, never an evaluation or learning attempt."""
import cProfile,json,runpy,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import MathChainTraining
original=MathChainTraining.present
class Finished(Exception): pass
def present(*args,**kwargs):
    after=kwargs.get('after_batch')
    def stop(*a):
        if after: after(*a)
        raise Finished()
    kwargs['after_batch']=stop
    return original(*args,**kwargs)
MathChainTraining.present=present
profiler=cProfile.Profile()
sys.argv=[str(HERE/'development_documents.py'),'development-profile-01']
try:
    profiler.enable()
    runpy.run_path(sys.argv[0],run_name='__main__')
except Finished:
    pass
finally:
    profiler.disable()
    profiler.dump_stats(str(HERE/'development-profile-01.pstats'))
    import pstats
    pstats.Stats(profiler).strip_dirs().sort_stats('cumulative').print_stats(45)
