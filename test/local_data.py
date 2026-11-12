"""Required external data is optional in an offline checkout."""
from pathlib import Path
import pytest
from prepare_data import require_local


def require_mnist():
    from util import ProjectPaths
    for name in ('mnist_train.csv', 'mnist_test.csv'):
        try:
            require_local(Path(ProjectPaths.DATA_DIR) / name)
        except FileNotFoundError as error:
            pytest.skip(str(error))
