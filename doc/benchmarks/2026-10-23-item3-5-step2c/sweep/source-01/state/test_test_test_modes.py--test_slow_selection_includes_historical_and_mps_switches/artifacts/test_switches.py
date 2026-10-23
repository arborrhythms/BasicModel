import os,pytest,unittest
_RUN_SLOW = os.getenv("RUN_SLOW") == "1"
@pytest.mark.skipif(not _RUN_SLOW, reason="old opt-in")
def test_old_pytest(): pass
@unittest.skipIf(not _RUN_SLOW, "old unittest opt-in")
class TestOld(unittest.TestCase):
 def test_class(self): pass
@pytest.mark.skipif(os.getenv("RUN_MPS_SLOW") != "1", reason="MPS opt-in")
@pytest.mark.parametrize("width", [16,64])
def test_mps(width): pass
def test_ordinary(): pass
