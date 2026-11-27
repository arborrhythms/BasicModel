"""The first Pi's column index, made when it is read, names the rows that
read each key, as the rows themselves do (item 1.75)."""
import random
from OrderZero import SparsePiLayer3


def test_rows_reading_each_key_after_every_append():
    rng = random.Random(0)
    pi = SparsePiLayer3(64, 512)
    keys = [('byte', k) for k in range(20)]
    for _ in range(40):
        held = rng.sample(keys, rng.randint(1, 6))
        pi.append(held)
        for key in keys:
            expected = [row for row in range(pi.n_rows) if key in pi.row_members(row)]
            assert pi.rows_reading(key) == expected
    pi.clear()
    assert pi.rows_reading(keys[0]) == []
