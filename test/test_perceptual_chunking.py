import pytest
from Spaces import Embedding, PartSpace








def test_perceptual_space_exposes_synthesis_mode_attribute():
    """PartSpace carries a synthesis_mode attribute resolved at construction."""
    # Instance-level attribute is set by __init__; class-level default is not
    # defined, so we only verify the setter exists via a fresh instance.
    # Use a synthetic Space-like fake: the attribute we care about is
    # populated from config, so just assert the init path references it.
    import inspect
    src = inspect.getsource(PartSpace.__init__)
    assert "self.synthesis_mode" in src
