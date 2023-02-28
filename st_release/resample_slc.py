#!/usr/bin/env python
u"""
Enrico Ciraci' - 12/2022

Resample SLC -> Change Azimuth Resolution and PRF
"""
# - Python module imports
import os
import numpy as np
# - GAMMA's Python integration with the py_gamma module
import py_gamma as pg


def resample_slc_azimuth(data_dir: str, ref: str, sec: str,
                         multi_look: bool = False) -> None:
    """
    Change Azimuth Resolution of the reference and secondary SLCs
    :param data_dir: data directory - absolute path
    :param ref: reference SLC - relative path from data_dir
    :param sec: secondary SLC - relative path from data_dir
    :param multi_look: generate multi-looked secondary SLCs
    :return: None
    """
    # - Read Reference SLC par file
    ref_param = pg.ParFile(os.path.join(data_dir, f'{ref}.par'))
    sec_param = pg.ParFile(os.path.join(data_dir, f'{sec}.par'))
    off_param = pg.ParFile(os.path.join(data_dir, f'{ref}-{sec}.par'))

    # - Get Azimuth Spacing values
    ref_az_spacing = float(ref_param.get_value('azimuth_pixel_spacing')[0])
    sec_az_spacing = float(sec_param.get_value('azimuth_pixel_spacing')[0])
    off_az_poly = off_param.get_value('azimuth_offset_polynomial')
    # - Get Azimuth Offset Polynomial
    print(f'# - REF Azimuth Spacing: {ref_az_spacing}')
    print(f'# - SEC Azimuth Spacing: {sec_az_spacing}')
    off_az_poly[2] = str(1 - float((sec_az_spacing / ref_az_spacing)))
    print(f'# - Interpolation Offsets Polynomial: {off_az_poly}')
    off_param.set_value('azimuth_offset_polynomial', off_az_poly)
    off_param.write_par(os.path.join(data_dir, f'{ref}-{sec}.par'))

    # - Interpolate Secondary SLC
    pg.SLC_interp(
        os.path.join(data_dir, f'{sec}.slc'),
        os.path.join(data_dir, f'{ref}.par'),
        os.path.join(data_dir, f'{sec}.par'),
        os.path.join(data_dir, f'{ref}-{sec}.par'),
        os.path.join(data_dir, f'{sec}_r.slc'),
        os.path.join(data_dir, f'{sec}_r.par')
    )
    sec_r_param = pg.ParFile(os.path.join(data_dir, f'{sec}_r.par'))
    sec_r_az_spacing = float(sec_r_param.get_value('azimuth_pixel_spacing')[0])
    print(f'# - REF Azimuth Spacing: {ref_az_spacing}')
    print(f'# - SEC Azimuth Spacing: {sec_az_spacing}')
    print(f'# - SEC Resampled Azimuth Spacing: {sec_r_az_spacing}')

    # - Generate new offset file
    algorithm = 1  # - offset estimation algorithm
    rlks = 1  # - number of interferogram range looks (enter -  for default: 1)
    azlks = 1  # - number of interferogram azimuth looks (enter-for default: 1)
    iflg = 0  # -  interactive mode flag (enter -  for default)
    # - Create Offset Parameter File
    pg.create_offset(
        os.path.join(data_dir, f'{ref}.par'),
        os.path.join(data_dir, f'{sec}_r.par'),
        os.path.join(data_dir, f'{ref}-{sec}_r.par'),
        algorithm, rlks, azlks, iflg
    )

    # - Initial SLC image offset estimation from orbit state-vectors
    # - and image parameters
    pg.init_offset_orbit(
        os.path.join(data_dir, f'{ref}.par'),
        os.path.join(data_dir, f'{sec}_r.par'),
        os.path.join(data_dir, f'{ref}-{sec}_r.par')
    )

    if multi_look:
        # - generate multi-looked secondary SLCs
        mli_slc_name = os.path.join(data_dir, f'{sec}_r_mli.slc')
        mli_par_name = os.path.join(data_dir, f'{sec}_r_mli.par')
        pg.multi_look(os.path.join(data_dir, f'{sec}_r.slc'),
                      os.path.join(data_dir, f'{sec}_r.par'),
                      mli_slc_name, mli_par_name,
                      10, 5
                      )
        # - Read Multi-Looked SLCs par file
        par_dict = pg.ParFile(mli_par_name).par_dict
        n_rsmpl = int(par_dict['range_samples'][0])

        # - Calculate a raster image from data with power-law scaling
        pg.raspwr(os.path.join(data_dir, f'{sec}_r_mli.slc'), n_rsmpl)


def resample_slc_prf(data_dir: str, ref: str, sec: str,
                     multi_look: bool = False) -> None:
    """
    Change Pulse Repetition Frequency of the reference and secondary SLCs
    :param data_dir: data directory - absolute path
    :param ref: reference SLC - relative path from data_dir
    :param sec: secondary SLC - relative path from data_dir
    :param multi_look: generate multi-looked secondary SLCs
    :return: None
    """
    # - Read Reference SLC par file
    ref_param = pg.ParFile(os.path.join(data_dir, f'{ref}.par'))
    sec_param = pg.ParFile(os.path.join(data_dir, f'{sec}.par'))

    # - Get Reference PRF value
    ref_az_prf = float(ref_param.get_value('prf')[0])
    # - Get Secondary PRF value
    sec_az_prf = float(sec_param.get_value('prf')[0])
    # - other parameters
    start_time = float(sec_param.get_value('start_time')[0])
    end_time = float(sec_param.get_value('end_time')[0])
    azimuth_pixel_spacing \
        = float(sec_param.get_value('azimuth_pixel_spacing')[0])

    # - Generate resampled SLC parameters file
    # - Azimuth line time
    # - time between SLC image lines in the resampled SLC [s]
    azimuth_line_time = 1/ref_az_prf
    # - Azimuth pixel spacing [m]
    azimuth_pixel_spacing *= (sec_az_prf/ref_az_prf)

    # - Time duration of the original SLC derived from the start_time
    # - and end_time and divide by the new azimuth line time [-]
    azimuth_lines = np.ceil(((end_time - start_time) / azimuth_line_time)) + 1
    # - Update end_time and center_time
    center_time = start_time + (azimuth_line_time * (azimuth_lines / 2))
    end_time = start_time + (azimuth_line_time * azimuth_lines)

    # - Generate Resampled Secondary SLC values
    sec_param.set_value('prf', ref_az_prf)
    sec_param.set_value('azimuth_pixel_spacing', azimuth_pixel_spacing)
    sec_param.set_value('azimuth_line_time', azimuth_line_time)
    sec_param.set_value('azimuth_lines', int(azimuth_lines))
    sec_param.set_value('center_time', center_time)
    sec_param.set_value('end_time', end_time)

    # - save new parameter file
    sec_param.write_par(os.path.join(data_dir, f'{sec}_r.par'))
    # - add stater vectors to the new parameter file
    with open(os.path.join(data_dir, f'{sec}.par'), 'r') as s_fid:
        st_vl = s_fid.readlines()
    with open(os.path.join(data_dir, f'{sec}_r.par'), 'a') as r_fid:
        for line in st_vl:
            if line.startswith('state_vector_position_') or \
                    line.startswith('state_vector_velocity_'):
                r_fid.write(line)

    # - resample secondary SLC
    pg.resamp_image_par(
        os.path.join(data_dir, f'{sec}.slc'),
        os.path.join(data_dir, f'{sec}.par'),
        os.path.join(data_dir, f'{sec}_r.par'),
        os.path.join(data_dir, f'{sec}_r.slc'),
        6,      # - use default interpolation method Bspline [4] / Lanczos [6]
        1,      # - input/output data type - FCOMPLEX
        7,      # - interpolation function order
    )

    # - Generate new offset file
    algorithm = 1  # - offset estimation algorithm
    rlks = 1  # - number of interferogram range looks (enter -  for default: 1)
    azlks = 1  # - number of interferogram azimuth looks (enter-for default: 1)
    iflg = 0  # -  interactive mode flag (enter -  for default)
    # - Create Offset Parameter File
    pg.create_offset(
        os.path.join(data_dir, f'{ref}.par'),
        os.path.join(data_dir, f'{sec}_r.par'),
        os.path.join(data_dir, f'{ref}-{sec}_r.par'),
        algorithm, rlks, azlks, iflg
    )

    # - Initial SLC image offset estimation from orbit state-vectors
    # - and image parameters
    pg.init_offset_orbit(
        os.path.join(data_dir, f'{ref}.par'),
        os.path.join(data_dir, f'{sec}_r.par'),
        os.path.join(data_dir, f'{ref}-{sec}_r.par')
    )

    if multi_look:
        # - generate multi-looked secondary SLCs
        mli_par_name = os.path.join(data_dir, f'{sec}_r_mli.par')
        mli_slc_name = os.path.join(data_dir, f'{sec}_r_mli.slc')
        pg.multi_look(os.path.join(data_dir, f'{sec}_r.slc'),
                      os.path.join(data_dir, f'{sec}_r.par'),
                      mli_slc_name, mli_par_name, 10, 5
                      )
        # - Read Multi-Looked SLCs par file
        par_dict = pg.ParFile(mli_par_name).par_dict
        n_rsmpl = int(par_dict['range_samples'][0])

        # - Calculate a raster image from data with power-law scaling
        pg.raspwr(os.path.join(data_dir, f'{sec}_r_mli.slc'), n_rsmpl)


