"""
Enrico Ciraci 03/2022
Create Directory at the selected location.
"""
import os
from pathlib import Path


def make_dir(abs_path: Path, dir_name: str) -> Path:
    """
    Create directory
    :param abs_path: absolute path to the output directory
    :param dir_name: new directory name
    :return: absolute path to the new directory
    """
    dir_to_create = abs_path / dir_name
    if not dir_to_create.is_dir():
        os.mkdir(str(dir_to_create))
    return dir_to_create
