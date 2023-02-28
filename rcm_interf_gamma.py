#!/usr/bin/env python
u"""
Temporal Doc Stirg
"""
# - Python Dependencies
from __future__ import print_function
import os
import argparse
import datetime
import shutil
import numpy as np
import yaml
from pathlib import Path
# - GAMMA's Python integration with the py_gamma module
import py_gamma as pg
import py_gamma2019 as pg9
# - ST_Release dependencies
from scipy.signal import medfilt
from astropy.convolution import convolve, Box2DKernel
from st_release.madian_filter_off import median_filter_off
from st_release.fill_nodata import fill_nodata
from utils.make_dir import make_dir


def create_isp_par(data_dir: str, ref: str, sec: str,
                   algorithm: int = 1, rlks: int = 1,
                   azlks: int = 1, iflg: int = 0):
    """
    Generate a new parameter file ISP offset and interferogram parameter files
    :param data_dir: absolute path to data directory
    :param ref: reference SLC
    :param sec: secondary SLC
    :param algorithm: offset estimation algorithm
    :param rlks: number of interferogram range looks
    :param azlks: number of interferogram azimuth looks
    :param iflg: interactive mode flag [0, 1]
    :return: None
    """
    # - Create and update ISP offset and interferogram parameter files
    pg.create_offset(
        os.path.join(data_dir, f'{ref}.par'),
        os.path.join(data_dir, f'{sec}.par'),
        os.path.join(data_dir, f'{ref}-{sec}.par'),
        algorithm, rlks, azlks, iflg
    )
    # - Initial SLC image offset estimation from orbit state-vectors
    # - and image parameters
    pg.init_offset_orbit(
        os.path.join(data_dir, f'{ref}.par'),
        os.path.join(data_dir, f'{sec}.par'),
        os.path.join(data_dir, f'{ref}-{sec}.par')
    )


def main() -> None:
    """
    Interferogram generation pipeline
    """
    parser = argparse.ArgumentParser(
        description="""RCM Interferogram generation pipeline."""
    )
    parser.add_argument('param_yaml', type=str,
                        help='YAML file with processing parameters')
    args = parser.parse_args()

    # - Read Processing Parameters
    processing_parameters_yml = args.param_yaml
    with open(processing_parameters_yml, encoding='utf8') as file:
        proc_param = yaml.load(file, Loader=yaml.FullLoader)

    # - Print Processing Parameters
    # - Input/Output Directories
    data_dir = Path(proc_param['global_parameters']['data_directory'])
    out_dir = Path(proc_param['global_parameters']['output_directory'])

    # - Processing Parameters
    ref = proc_param['global_parameters']['refrence_slc']   # - Reference SLC
    sec = proc_param['global_parameters']['seconday_slc']   # - Secondary SLC

    # - Track Output directory
    if not (out_dir/f'Track{ref}-{sec}').is_dir():
        make_dir(out_dir, f'Track{ref}-{sec}')
    out_dir = out_dir/f'Track{ref}-{sec}'

    # - Create symbolic links for each of the .slc and .par files
    if out_dir != data_dir:
        for slc in [ref, sec]:
            # - SLC
            slc_ln = out_dir/f'{slc}.slc'
            if slc_ln.is_file():
                os.remove(str(slc_ln))
            slc_ln.symlink_to((data_dir/f'{slc}.slc').resolve())
            # - PAR
            par_ln = out_dir/f'{slc}.par'
            if par_ln.is_file():
                os.remove(str(par_ln))
            par_ln.symlink_to((data_dir/f'{slc}.par').resolve())
        # - Update data_dir value
        data_dir = out_dir

    # - Convert Pathlib objects to string type
    data_dir = str(data_dir)
    out_dir = str(out_dir)

    # - Create New ISP Parameter file
    create_isp_par(data_dir, ref, sec)

    # - Registration offset fields Preliminary Offset
    p_doff = proc_param['preliminary_offsets_parameters']['p_doff']
    s_window = proc_param['preliminary_offsets_parameters']['search_window']
    skip = proc_param['preliminary_offsets_parameters']['skip']
    nr = proc_param['preliminary_offsets_parameters']['nr']
    naz = proc_param['preliminary_offsets_parameters']['naz']

    if p_doff:
        pg.offset_pwr_tracking(
            os.path.join(data_dir, f'{ref}.slc'),
            os.path.join(data_dir, f'{sec}.slc'),
            os.path.join(data_dir, f'{ref}.par'),
            os.path.join(data_dir, f'{sec}.par'),
            os.path.join(data_dir, f'{ref}-{sec}.par'),
            os.path.join(out_dir, 'sparse_offsets'),
            os.path.join(out_dir, 'sparse_offsets.ccp'),
            s_window, s_window,
            os.path.join(out_dir, 'sparse_offsets.txt'),
            '-', '-', skip, skip, '-', '-', '-', '-', '-', '-',
        )
    else:
        pg.offset_pwr(os.path.join(data_dir, f'{ref}.slc'),
                      os.path.join(data_dir, f'{sec}.slc'),
                      os.path.join(data_dir, f'{ref}.par'),
                      os.path.join(data_dir, f'{sec}.par'),
                      os.path.join(data_dir, f'{ref}-{sec}.par'),
                      os.path.join(data_dir, 'sparse_offsets'),
                      os.path.join(data_dir, 'sparse_offsets.ccp'),
                      s_window, s_window,
                      os.path.join(data_dir, 'sparse_offsets.txt'),
                      '-', nr, naz
                      )


# - run main program
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print(f"# - Computation Time: {end_time - start_time}")
