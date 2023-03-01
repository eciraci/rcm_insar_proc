import sys
import numpy as np
sys.path.insert(0, '../st_release')
from st_release.fill_nodata import fill_nodata


def test_fill_nodata():
    # Generate test data
    data_arr = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
    mask_arr = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    max_search_dist = 100
    smth_iter = 10

    # Expected output
    expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Actual output
    result = fill_nodata(data_arr, mask_arr, max_search_dist, smth_iter)

    # Compare expected and actual output
    np.testing.assert_array_equal(result, expected)