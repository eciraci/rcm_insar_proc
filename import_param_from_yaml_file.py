# - Python Dependencies
import os
import datetime
import yaml


def main() -> None:
    """
    Interferometruic
    :return:
    """
    # - Read Processing Parameters
    processing_parameters_yml = os.path.join('.', 'processing_parameters.yml')
    with open(processing_parameters_yml, encoding='utf8') as file:
        proc_param = yaml.load(file, Loader=yaml.FullLoader)

    # - Print Processing Parameters
    # - Input/Output Directories
    data_dir = proc_param['global_parameters']['data_directory']
    out_dir = proc_param['global_parameters']['output_directory']

    # - Processing Parameters
    ref = proc_param['global_parameters']['refrence_slc']  # - Reference SLC
    sec = proc_param['global_parameters']['seconday_slc']   # - Secondary SLC





# - Add if main statement
if __name__ == "__main__":
    main()
