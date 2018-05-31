from growing_array import GrowingArray,FlatGrowingArray
import utils
import numpy as np
from matplotlib import pyplot as plt
import os,sys

class RegisteredAverage:

    def __init__(self,reference,oversampling_factor=(1,1),n_strips=4,use_window_for_oversampling=False):
        """RegisteredAverage is a registered average of B-scan strips.

        Args:
            reference (2D numpy.ndarray): the reference B-scan, to which other B-scans will
                aligned
            oversampling_factor (integer): the factor by which to oversample the reference
                scan and inserted scans before registering and aligning
            n_sections (integer): the number of strips into which the inserted B-scans should
                be cut before registering
        """
        self.use_window_for_oversampling = use_window_for_oversampling
        rsy,rsx = reference.shape
        orsx=rsx*oversampling_factor[1]
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
        self.fga = FlatGrowingArray()
        self.fga.put(self.ref,(0,0))
        self.t = 1 # where to insert the next strip
        self.valid_strip_count = 0.0
        self.total_strip_count = 0.0
        
        
    def oversample(self,arr):
        """FFT-based oversampling. Uses this object's oversampling_factor to zero-pad
        the IFFT.

        Args:
            arr (2D numpy.ndarray): the image to oversample

        Returns:
            the oversampled image
        """
        if self.oversampling_factor==(1,1):
            return arr
        sy,sx = arr.shape
        osy,osx = sy*self.oversampling_factor[0],sx*self.oversampling_factor[1]

        f_arr = np.fft.fftshift(np.fft.fft2(arr))
        
        if self.use_window_for_oversampling:
            win = np.hamming(sy)
            f_arr = (f_arr.T*win).T

        oversampled = np.abs(np.fft.ifft2(f_arr,s=(osy,osx)))*np.prod(self.oversampling_factor)
        #oversampled = np.abs(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(arr)),s=(osy,osx)))*self.oversampling_factor**2

        return oversampled


    def make_strips(self,arr):
        """Divide an input image arr into a number of strips determined by this
        object's n_strips value. Return a list of these strips.
        """
        return [arr[:,x1:x2] for x1,x2 in zip(self.strip_starts,self.strip_ends)]
    
    def add(self,target,do_plot=False,correlation_threshold=-np.inf):
        """Add a target to this BStack. This method divides the target into
        this object's n_strips strips, and cross-correlates these strips with
        the reference image. It then puts the strips into this object's GrowableArray
        at the correct coordinates.
        """
        tar = self.oversample(target)
        tars = self.make_strips(tar)

        for tar,ref,fref,offset in zip(tars,self.refs,self.frefs,self.strip_starts):
            self.total_strip_count += 1.0
            ftar = np.conj(np.fft.fft2(tar))
            fprod = ftar*fref
            xc = np.abs(np.fft.ifft2(fprod))
            peaky,peakx = np.unravel_index(np.argmax(xc),xc.shape)
            if peaky>xc.shape[0]//2:
                peaky=peaky-xc.shape[0]
            if peakx>xc.shape[1]//2:
                peakx=peakx-xc.shape[1]

            # if correlation threshold is set (i.e. if it's greater than -np.inf,
            # then check aligned correlation; if fail, continue out of the loop
            # iteration; if success, do nothing--loop will proceed to the fga.put call
            if correlation_threshold>-np.inf:
                sy,sx = tar.shape
                # make length 2 arrays for x and y shifts, where the reference shifts
                # (both 0) are the first items, and the target shifts (peakx and peaky)
                # are the second, then subtract their minimum values to make all shifts
                # positive while preserving relative correctness of target shifts
                xshifts = np.array([0.0,peakx],dtype=np.integer)
                yshifts = np.array([0.0,peaky],dtype=np.integer)
                xshifts = xshifts-np.min(xshifts)
                yshifts = yshifts-np.min(yshifts)
                # compute dims of slightly expanded arrays and make arrays of
                # zeros to hold the shifted ref and tar
                esy = sy+int(round(yshifts.max()))
                esx = sx+int(round(xshifts.max()))
                eref = np.zeros((esy,esx))
                etar = np.zeros((esy,esx))
                # insert the shifted ref and tar at matching locations in
                # expanded arrays:
                eref[yshifts[0]:yshifts[0]+sy,xshifts[0]:xshifts[0]+sx] = ref
                etar[yshifts[1]:yshifts[1]+sy,xshifts[1]:xshifts[1]+sx] = tar
                # ravel and compute correlation between aligned arrays:
                corr = np.corrcoef(np.array([etar.ravel(),eref.ravel()]))[0,1]

                if corr<=correlation_threshold:
                    #print 'Skipping strip with correlation %0.3f.'%corr
                    continue
            self.valid_strip_count = self.valid_strip_count + 1.0
            self.fga.put(tar,coords=(peaky,peakx+offset))
            
        self.t = self.t + 1
        valid_fraction = self.valid_strip_count/self.total_strip_count
        #print 'Valid strip fraction: %0.2f'%(valid_fraction)

        if do_plot:
            plt.cla()
            im = self.fga.get_average()
            im = utils.nanreplace(im,'mean')
            im = np.log(im)
            clim = np.percentile(im,(40,99.9))
            plt.imshow(im,clim=clim,cmap='gray',aspect='auto')
            #plt.ylim((920,700))
            plt.title('t=%d,valid=%0.2f'%(self.t,valid_fraction)) 
            plt.pause(.5)


class BStackDepricated:

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
        self.ga = FlatGrowingArray()
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


