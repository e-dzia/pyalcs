import pytest

from lcs.representations.visualization import visualize, _scale


@pytest.mark.parametrize("_interval, _range, n, _visualization", [
    ((3, 3), (0, 7), 10, '...O....'),
    ((0, 2), (0, 7), 10, 'OOO.....'),
    ((2, 4), (0, 7), 10, '..OOO...'),
    ((0, 7), (0, 7), 10, 'OOOOOOOO'),
    ((0, 1), (0, 3), 10, 'OO..'),
    ((3, 3), (0, 3), 10, '...O'),
    ((0, 7), (0, 15), 10, 'OOOOO.....'),
    ((0, 15), (0, 15), 10, 'OOOOOOOOOO'),
    ((0, 1), (0, 31), 10, 'O.........'),
])
def test_visualize(_interval, _range, n, _visualization):
    assert visualize(_interval, _range) == _visualization


@pytest.mark.parametrize("_val, _init_n, _n, _result", [
    (2, 4, 10, 5),
    (2, 8, 10, 2),
    (4, 8, 10, 5),
])
def test_scale(_val, _init_n, _n, _result):
    assert _scale(_val, _init_n, _n) == _result
