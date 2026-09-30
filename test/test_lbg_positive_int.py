"""WholeSpace no longer has the word/META LBG mutation path."""
import pytest
from Spaces import WholeSpace

@pytest.mark.parametrize('name', ['_record_lbg_pull', '_maybe_split_lbg', 'insert_meta'])
def test_wholespace_word_taxonomy_mutators_are_deleted(name):
    assert not hasattr(WholeSpace, name)
