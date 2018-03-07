from config_reader import get_configuration
import sys,os,time
import numpy as np


def get_configuration_parameters(xml_filename):
    # get a dictionary of configuration parameters (pixels per
    # a-scan, a-scans per b-scan, b-scans per volume, etc.)
    config = get_configuration(xml_filename)
    
    n_vol = int(config['n_vol'])
    n_slow = int(config['n_slow'])
    n_fast = int(config['n_fast'])
    n_depth = int(config['n_depth'])

    return n_vol,n_slow,n_fast,n_depth

def get_frame(filename,volume_index,bscan_index,dtype=np.uint16):
    '''Get a raw frame from a UNP file. This function will
    try to read configuration details from a UNP file with
    the same name but .xml extension instead of .unp.
    Parameters:
        filename: the name of the .unp file
        volume_index: the index of the desired volume
        bscan_index: the index of the desired bscan
        dtype: the data type, assumed to be numpy.uint16
    Returns:
        a 2D numpy array of size n_depth x n_fast
    '''

    # append extension if it's not there
    if not filename[-4:].lower()=='.unp':
        filename = filename + '.unp'

    n_vol,n_slow,n_fast,n_depth = get_configuration_parameters(filename.replace('.unp','.xml'))

    # we also need to know how many bytes there are per pixel; rely
    # on dtype to figure this out
    n_bytes = dtype(1).itemsize

    # open the file and read in the b-scan
    with open(filename,'rb') as fid:
        position = volume_index * n_depth * n_fast * n_slow * n_bytes + bscan_index * n_depth * n_fast * n_bytes
        fid.seek(position,0)
        bscan = np.fromfile(fid,dtype=dtype,count=n_depth*n_fast)
        bscan = bscan.reshape(n_fast,n_depth).T

    return bscan


