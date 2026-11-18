"""Quality is evaluated after substantial training, not on random tiny models."""
import statistics
import pytest
from eval_fineweb_learning import load_model, read_validation
