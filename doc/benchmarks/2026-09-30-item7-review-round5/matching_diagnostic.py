"""The guarded worker environment, with only its memory guard removed."""
import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'test'))
from bounded_tests import ProcessTree,worker_environment

def diagnostic(selector,directory):
 directory.mkdir()
 started=time.monotonic()
 with (directory/'pytest.log').open('w') as log:
  proc=subprocess.Popen([sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',selector],cwd=ROOT,env=worker_environment(ROOT),stdout=log,stderr=log,start_new_session=True)
  tree,peak,reason=ProcessTree(proc.pid),0,'completed'
  while proc.poll() is None:
   peak=max(peak,tree.sample())
   if time.monotonic()-started>1800:
    tree.terminate(proc,.5);reason='timeout';break
   time.sleep(.1)
 result=dict(selector=selector,diagnostic_only=True,memory_guard=None,exit_code=proc.returncode,reason=reason,peak_memory_bytes=peak,elapsed_seconds=time.monotonic()-started,log=str(directory/'pytest.log'))
 (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
