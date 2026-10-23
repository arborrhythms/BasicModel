import pytest
from pathlib import Path
@pytest.fixture
def expensive():
 Path('expensive-ran').write_text('yes')
def test_quick(): pass
@pytest.mark.slow
def test_training(expensive): pass
