import numpy as np

def logscale(frame,lower_limit=None,upper_limit=None,bit_depth=16):
    '''A convenience function for scaling OCT b-scans
    expects a complex, linear-scale matrix. Returns a
    rounded value scaled to the desired bit depth.'''

    frame = np.log(np.abs(frame))
    if lower_limit is None:
        lower_limit = np.median(frame)+0.5*np.std(frame)
    if upper_limit is None:
        upper_limit = np.median(frame)+3.5*np.std(frame)
    
    return np.round(((frame - lower_limit)/(upper_limit-lower_limit)*2**bit_depth)).clip(0,2**bit_depth)

def linearscale(frame,lower_limit=None,upper_limit=None,bit_depth=16):
    '''A convenience function for scaling OCT b-scans
    expects a complex, linear-scale matrix. Returns a
    rounded value scaled to the desired bit depth.'''
    frame = np.abs(frame)
    if lower_limit is None:
        lower_limit = np.median(frame)+0.5*np.std(frame)
    if upper_limit is None:
        upper_limit = np.median(frame)+3.5*np.std(frame)
    
    return np.round(((frame - lower_limit)/(upper_limit-lower_limit)*2**bit_depth)).clip(0,2**bit_depth)
