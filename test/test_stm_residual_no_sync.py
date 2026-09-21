"""The retired per-word prediction injection has no callable path."""
from Language import SymbolSubSpace


def test_no_comprehension_prediction_api():
    for name in ("stm_residual", "stm_residual_microbatch", "arm_stm"):
        assert not hasattr(SymbolSubSpace, name)
