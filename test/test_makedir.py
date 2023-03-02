from tempfile import TemporaryDirectory
import sys
import os
import pytest
from pathlib import Path
from utils.make_dir import make_dir
sys.path.insert(0, '../utils')

@pytest.fixture
def temp_dir():
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_make_dir(temp_dir):
    dir_name = "test_dir"
    dir_path = temp_dir / dir_name

    # Ensure directory is not initially present
    assert not dir_path.is_dir()

    # Create directory
    make_dir(temp_dir, dir_name)

    # Ensure directory is created
    assert dir_path.is_dir()

    # Ensure the function returns the correct directory path
    assert make_dir(temp_dir, dir_name) == dir_path

    # Clean up the directory after the test
    os.rmdir(str(dir_path))