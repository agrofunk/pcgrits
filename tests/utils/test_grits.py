from pandas._libs.testing import assert_almost_equal

from utils.grits import get_min_and_max_values


def test_get_min_and_max_values(xarray_dataset):
    # um xarray_dataset qualquer
    # quando eu buscar pelo min/max dos quantis de referência
    min_max_lim_map = get_min_and_max_values(xarray_dataset, ["temperature", "precipitation"])

    # espero ter o valor dos min e max corretamente
    assert_almost_equal(min_max_lim_map["precipitation"], [0.2580834,  9.72785955])
    assert_almost_equal(min_max_lim_map["temperature"], [7.90858715, 32.59861061])