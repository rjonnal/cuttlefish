from growing_array import GrowingArray
import utils
import numpy as np
from matplotlib import pyplot as plt
import os,sys

class BStack:

    def __init__(self,reference,oversampling_factor=1,n_strips=4):
        """BStack is a registered stack of B-scan strips.

        Args:
            reference (2D numpy.ndarray): the reference B-scan, to which other B-scans will
                aligned
            oversampling_factor (integer): the factor by which to oversample the reference
                scan and inserted scans before registering and aligning
            n_sections (integer): the number of strips into which the inserted B-scans should
                be cut before registering
        """

        rsy,rsx = reference.shape
        orsx=rsx*oversampling_factor
        if not orsx % n_strips == 0:
            sys.exit('Oversampled width %d fails to evenly divide into %d strips. Please modify parameters or crop reference image accordingly.'%(orsx,n_strips))
        else:
            strip_width = orsx//n_strips
            self.strip_starts = range(0,orsx,strip_width)
            self.strip_ends = [s+strip_width for s in self.strip_starts]
            
        self.oversampling_factor = oversampling_factor
        self.n_strips = n_strips
        self.ref = self.oversample(reference)
        self.refs = self.make_strips(self.ref)
        # precompute the reference FFTs so we don't have to do it
        # for every new target
        self.frefs = [np.fft.fft2(r) for r in self.refs]
        
        # a flexible data structure for storing images
        self.ga = GrowingArray()
        self.ga.put(self.ref,(0,0,0))
        self.t = 1 # where to insert the next strip

    def oversample(self,arr):
        """FFT-based oversampling. Uses this object's oversampling_factor to zero-pad
        the IFFT.

        Args:
            arr (2D numpy.ndarray): the image to oversample

        Returns:
            the oversampled image
        """
        if self.oversampling_factor==1:
            return arr
        sy,sx = arr.shape
        osy,osx = sy*self.oversampling_factor,sx*self.oversampling_factor
        return np.abs(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(arr)),s=(osy,osx)))*self.oversampling_factor**2


    def make_strips(self,arr):
        """Divide an input image arr into a number of strips determined by this
        object's n_strips value. Return a list of these strips.
        """
        return [arr[:,x1:x2] for x1,x2 in zip(self.strip_starts,self.strip_ends)]
    
    def add(self,target,do_plot=False):
        """Add a target to this BStack. This method divides the target into
        this object's n_strips strips, and cross-correlates these strips with
        the reference image. It then puts the strips into this object's GrowableArray
        at the correct coordinates.
        """
        tar = self.oversample(target)
        tars = self.make_strips(tar)

        for tar,fref,offset in zip(tars,self.frefs,self.strip_starts):
            ftar = np.conj(np.fft.fft2(tar))
            fprod = ftar*fref
            xc = np.abs(np.fft.ifft2(fprod))
            peaky,peakx = np.unravel_index(np.argmax(xc),xc.shape)
            if peaky>xc.shape[0]//2:
                peaky=peaky-xc.shape[0]
            if peakx>xc.shape[1]//2:
                peakx=peakx-xc.shape[1]

            self.ga.put(tar,coords=(self.t,peaky,peakx+offset))
            
        self.t = self.t + 1

        if do_plot:
            plt.cla()
            im = np.nanmean(self.ga.data,axis=0)
            im = utils.nanreplace(im,'mean')
            im = np.log(im)
            clim = np.percentile(im,(30,99.9))
            plt.imshow(im,clim=clim,cmap='gray',aspect='auto')
            #plt.ylim((920,700))
            plt.title('t=%d'%self.t) 
            plt.pause(.1)


