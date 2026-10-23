import os,time
from pathlib import Path
def test_wait():
 Path('active.pid').write_text(str(os.getpid()))
 time.sleep(60)
