import sys
sys.path.insert(0, '../st_release')
import numpy as np
from st_release.median_filter_off import median_filter_off


def test_median_filter_off():
    # Generate test data
    off = np.array([1+2j, 3+4j, 5+6j, 7+16j, 15+10j])
    size = 3
    thre = 2

    # Expected output
    expected = np.array([False, False, False, True, True])

    # Actual output
    result = median_filter_off(off, size, thre)

    # Compare expected and actual output
    print(result)
    print(expected)
    np.testing.assert_array_equal(result, expected)
