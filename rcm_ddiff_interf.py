#!/usr/bin/env python
u"""
Enrico Ciraci' - 03/2023

Compute Double-Difference Interferogram between the selected pair of
Geocoded Interferograms - GAMMA Pipeline.

usage: rcm_ddiff_inerf.py [-h] param_yaml

Compute RCM Double-Difference Interferograms.

positional arguments:
  param_yaml  YAML file with processing parameters

options:
  -h, --help  show this help message and exit

# - Python Dependencies
argparse: Parser for command-line options, arguments and subcommands
          https://docs.python.org/3/library/argparse.html
datetime: Basic date and time types
          https://docs.python.org/3/library/datetime.html
shutil: High-level file operations
          https://docs.python.org/3/library/shutil.html
yaml: YAML parser and emitter for Python
          https://pyyaml.org/wiki/PyYAMLDocumentation
pathlib: Object-oriented filesystem paths
          https://docs.python.org/3/library/pathlib.html

# - GAMMA's Python integration with the py_gamma module
py_gamma: Python interface to GAMMA SAR processing software

"""
# - Python Dependencies
from __future__ import print_function
import os
import sys
import argparse
import datetime
import yaml
from pathlib import Path
import shutil
# - GAMMA's Python integration with the py_gamma module
import py_gamma as pg
import py_gamma2019 as pg9
from utils.make_dir import make_dir


def main() -> None:
    """
    Double-difference Interferogram Calculation - GAMMA Pipeline.
    """
    parser = argparse.ArgumentParser(
        description="""Compute RCM Double-Difference Interferograms."""
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
    data_dir = Path(proc_param['global_parameters']['data_directory']).resolve()
    out_dir = Path(proc_param['global_parameters']['output_directory']).resolve()
    current_dir = str(Path.cwd().resolve())

    if not data_dir.is_dir():
        raise NotADirectoryError(f'{data_dir} - Not Found.')
    if not out_dir.is_dir():
        raise NotADirectoryError(f'{out_dir} - Not Found.')

    # - Processing Parameters
    # - Reference SLC
    igram_ref = proc_param['global_parameters']['reference_intf']
    # - Secondary SLC
    igram_sec = proc_param['global_parameters']['secondary_intf']

    data_dir_ref = os.path.join(str(data_dir), f'Track{igram_ref}')
    data_dir_sec = os.path.join(str(data_dir), f'Track{igram_sec}')

    # - Verify that the selected interferograms exist
    if not os.path.isdir(data_dir_ref):
        print(f'# - {data_dir_ref} - Not Found.')
        sys.exit()
    if not os.path.isdir(data_dir_sec):
        print(f'# - {data_dir_sec} - Not Found.')
        sys.exit()

    # - Path to Geocoded Interferogram Parameter Files
    ref_par = os.path.join(data_dir_ref, 'DEM_gc_par')
    sec_par = os.path.join(data_dir_sec, 'DEM_gc_par')

    # - Read Reference Geocode Interferogram Parameter File
    dem_param_dict = pg.ParFile(ref_par).par_dict
    dem_width = int(dem_param_dict['width'][0])

    # - Geocoded Interferograms
    ref_interf \
        = os.path.join(data_dir_ref, 'coco' + igram_ref + '.flat.topo_off.geo')
    sec_interf \
        = os.path.join(data_dir_sec, 'coco' + igram_sec + '.flat.topo_off.geo')
    # - Geocoded Reference SLC Power Intensity
    ref_pwr \
        = os.path.join(data_dir_ref, igram_ref.split('-')[0] + '.pwr1.geo')

    # - Path to Interferograms Baseline Files
    ref_base \
        = os.path.join(data_dir_ref, 'base' + igram_ref + '.dat')
    sec_base \
        = os.path.join(data_dir_sec, 'base' + igram_sec + '.dat')

    # - Create Output Directory
    out_dir = make_dir(out_dir, 'DOUBLE_DIFFERENCES')
    out_dir = make_dir(out_dir, f'Track{igram_ref}-Track{igram_sec}')

    # Change the current working directory
    os.chdir(out_dir)

    # - Resampling Secondary Interferogram on the Reference grid.
    pg.map_trans(sec_par, sec_interf, ref_par,
                 os.path.join('.', 'coco' + igram_sec + '.flat.topo_off.res'),
                 '-', '-', '-', 1)

    # - Path to co-registered complex interferogram
    reg_intf = os.path.join('.', 'coco' + igram_sec + '.flat.topo_off.res')

    # - Combine Complex Interferograms
    pg.comb_interfs(ref_interf, reg_intf, ref_base, sec_base, 1, -1,
                    dem_width,
                    'coco' + igram_ref + '-' + igram_sec + '.flat.topo_off.geo',
                    'base' + igram_ref + '-' + igram_sec + '.flat.topo_off.geo',
                    )

    # - Show Double Difference on Top of the Reference SLC power image
    pg9.rasmph_pwr('coco' + igram_ref + '-' + igram_sec + '.flat.topo_off.geo',
                   ref_pwr, dem_width)

    # - Smooth the obtained interferogram with pg.adf
    # - Adaptive interferogram filter using the power spectral density.
    pg.adf('coco' + igram_ref + '-' + igram_sec + '.flat.topo_off.geo',
           'coco' + igram_ref + '-' + igram_sec + '.flat.topo_off.geo.filt',
           'coco' + igram_ref + '-' + igram_sec + '.flat.topo_off.geo.filt.coh',
           dem_width)
    pg9.rasmph_pwr('coco' + igram_ref + '-' + igram_sec
                   + '.flat.topo_off.geo.filt', ref_pwr, dem_width)

    # - Calculate real part, imaginary part, intensity, magnitude,
    # - or phase of FCOMPLEX data
    # - Extract Interferogram Phase.
    pg9.cpx_to_real('coco' + igram_ref + '-' + igram_sec
                    + '.flat.topo_off.geo.filt',
                    'phs.geo', dem_width, 4)
    pg9.raspwr('phs.geo', dem_width)
    # - Save Geocoded Interferogram phase as a GeoTiff
    pg9.data2geotiff(ref_par, 'phs.geo', 2,
                     'coco' + igram_ref + '-' + igram_sec
                     + '.flat.topo_off.geo.filt.tiff', -9999)
    # - Save Coherence Interferogram Map as a GeoTiff
    pg9.data2geotiff(ref_par, 'coco' + igram_ref + '-' + igram_sec
                     + '.flat.topo_off.geo.filt.coh', 2,
                     'coco' + igram_ref + '-' + igram_sec
                     + '.flat.topo_off.geo.filt.coh.tiff', -9999)

    # - Change Permission Access to all the files contained inside the
    # - output directory.
    for out_file in os.listdir('.'):
        os.chmod(out_file, 0o0755)

    # - If the processing is successful, move parameter file to output directory
    shutil.move(os.path.join(current_dir, processing_parameters_yml), out_dir)


# - run main program
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print(f"# - Computation Time: {end_time - start_time}")
