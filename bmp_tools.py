import numpy as np

def logscale(frame,lstd=0.75,ustd=3.5,bit_depth=16):
    '''A convenience function for scaling OCT b-scans
    expects a complex, linear-scale matrix. Returns a
    rounded value scaled to the desired bit depth.'''
    frame = np.log(np.abs(frame))
    llim,ulim = (np.median(frame)+np.std(frame)*lstd,np.median(frame)+np.std(frame)*ustd)
    return np.round(((frame - llim)/(ulim-llim)*2**bit_depth)).clip(0,2**bit_depth)
