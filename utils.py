import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as spo
import scipy.signal as sps
import scipy.ndimage as spn
import sys,os

def nanreplace(mat,replacement_value):
    if replacement_value=='mean':
        replacement_value = np.nanmean(mat)
    mat[np.where(np.isnan(mat))] = replacement_value
    return mat

def nxcorr(vec1,vec2,do_plot=False):
    '''Given two vectors TARGET and REFERENCE, nxcorr(TARGET,REFERENCE)
    will return a pair (tuple) of values, (SHIFT, CORR). CORR is a quantity
    corresponding to the Pearson correlation of the two vectors, accounting
    for a time delay between the two. Put slightly differently, CORR is the
    Pearson correlation of the best alignment of the two vectors.
    SHIFT gives the number of pixels of delay between the two, such that
    shifting TARGET by SHIFT pixels (rightward for positive, leftward for
    negative) will produce the optimal alignment of the vectors.'''

    l1 = len(vec1)
    l2 = len(vec2)

    vec1 = (vec1 - np.mean(vec1))/np.std(vec1)
    vec2 = (vec2 - np.mean(vec2))/np.std(vec2)

    temp1 = np.zeros([l1+l2-1])
    temp2 = np.zeros([l1+l2-1])

    temp1[:l1] = vec1
    temp2[:l2] = vec2
        
    nxcval = np.real(np.fft.fftshift(np.fft.ifft(np.fft.fft(temp1)*np.conj(np.fft.fft(temp2)))))
    
    peakVal = np.max(nxcval)
    peakIdx = np.where(nxcval==peakVal)[0][0]


    if len(nxcval)%2:
        shift = (len(nxcval)-1)/2.0 - peakIdx
    else:
        shift = len(nxcval)/2.0 - peakIdx

    if do_plot:
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(vec1)
        plt.subplot(3,1,2)
        plt.plot(vec2)
        plt.subplot(3,1,3)
        plt.plot(nxcval)
        plt.show()
    return shift,peakVal/len(vec1)

def findShift(vec1,vec2):
    shift,peakVal = nxcorr(vec1,vec2)
    return shift

def shear(im,order,oversample=1.0,sameshape=False,y1=None,y2=None):
    sy,sx = im.shape
    sy0 = sy
    newsy = int(np.round(float(sy)*float(oversample)))

    if y1 is None:
        cropY1 = 0
    else:
        cropY1 = y1

    if y2 is None:
        cropY2 = sy
    else:
        cropY2 = y2

    imOriginal = np.zeros(im.shape)
    imOriginal[:] = im[:]

    im = im[cropY1:cropY2,:]

    fim = np.fft.fft(im,axis=0)
    fim = np.fft.fftshift(fim,axes=0)
    im = np.abs(np.fft.ifft(fim,axis=0,n=newsy))

    windowSize = 5

    refIdx = 0
    tarIdx = refIdx

    rx1 = refIdx
    rx2 = rx1 + windowSize

    tx1 = rx1
    tx2 = rx2

    xVector = []
    yVector = []

    ref = im[:,rx1:rx2]
    ref = np.mean(ref,axis=1)

    while tx2<sx:
        tar = im[:,tx1:tx2]
        tar = np.mean(tar,axis=1)
        shift = -findShift(ref,tar)
        xVector.append(tx1-rx1)
        yVector.append(shift)
        tx1 = tx1 + 1
        tx2 = tx2 + 1


    p = np.polyfit(xVector,yVector,order)
    newY = np.round(np.polyval(p,range(sx))).astype(np.int16)
    newY = newY - np.min(newY)

    
    newim = np.zeros([newsy+np.max(newY),sx])
    for ix in range(sx):
        newim[newY[ix]:newY[ix]+newsy,ix] = im[:,ix]

    newSum = np.sum(newim)

    osy,osx = newim.shape

    outsy = int(float(osy)/float(oversample))

    if oversample!=1.0:
        newim = imresize(newim,(outsy,sx),interp='bicubic')*oversample

    newim = newim/np.sum(newim)*newSum
    resampledSum = np.sum(newim)


    if sameshape:
        dy = newim.shape[0] - sy
        newim = newim[dy/2:dy/2+sy,:]

    return newim


def find_peaks(prof,intensity_threshold=-np.inf,gradient_threshold=-np.inf,permit_edge_peaks=False):

    if permit_edge_peaks:
        # add -inf to either end to permit peaks at edges
        prof = np.array([-np.inf]+list(prof)+[-np.inf])
    else:
        # add inf to either end to exclude peaks at edges
        prof = np.array([np.inf]+list(prof)+[np.inf])
        
    left = prof[:-2]
    center = prof[1:-1]
    right = prof[2:]
    peaks = np.where(np.logical_and(center>right,center>left))[0]
    peak_vals = prof[peaks]
    all_gradients = np.abs(np.diff(prof))
    l_gradients = all_gradients[:-1]
    r_gradients = all_gradients[1:]
    gradients = np.max([l_gradients,r_gradients],axis=0)
    peak_vals = np.array(peak_vals)
    gradient_vals = np.array(gradients[peaks-1])
    valid = np.where(np.logical_and(peak_vals>=intensity_threshold,gradient_vals>=gradient_threshold))[0]

    return peaks[valid]


def peak_edges(vec,start):
    """Return the indices of the start and end of the peak
    indexed by the variable start."""
    start = int(start)
    start = ascend(vec,start)
    troughs = np.array([0] + list(find_peaks(-vec)) + [len(vec)-1])
    try:
        left = np.max(troughs[np.where(troughs<start)])
    except Exception as e:
        left = 0
    try:
        right = np.min(troughs[np.where(troughs>start)])
    except Exception as e:
        right = 0
        
    return left,right
    
def ascend(vec,start,do_plot=False):
    start = int(start)
    floor = lambda x: int(np.floor(x))
    ceil = lambda x: int(np.ceil(x))
    if vec[floor(start)]>vec[ceil(start)]:
        start = floor(start)
    elif vec[floor(start)]<vec[ceil(start)]:
        start = ceil(start)

    out = start

    try:
        while vec[out+1]>vec[out]:
            out = out + 1
    except Exception as e:
        print e

    try:
        while vec[out-1]>vec[out]:
            out = out - 1
    except Exception as e:
        print e

    if do_plot:
        plt.plot(vec)
        plt.plot(start,vec[start],'go')
        plt.plot(out,vec[out],'ro')
        plt.show()
        
    return int(out)

def descend(vec,start):
    return ascend(-vec,start)


def bscan_coms(bscan):
    bscan = bscan.T
    
    idx = np.arange(bscan.shape[1])
    coms = np.sum(bscan*idx,axis=1)/np.sum(bscan,axis=1)
    return coms

def flatten_volume0(avol,order=2):
    sy,sz,sx = avol.shape
    fast_proj = avol.mean(axis=2).T
    fast_proj = fast_proj**2
    coms = bscan_coms(fast_proj)
    tempvol = np.zeros(avol.shape)
    x = np.arange(len(coms))
    p = np.polyfit(x,coms,order)
    fitcoms = np.round(np.polyval(p,x)).astype(np.int16)
    fitcoms = fitcoms-fitcoms.min()
    for y in range(sy):
        z1 = fitcoms[y]
        tempvol[y,:sz-z1,:] = avol[y,z1:,:]
    slow_proj = avol.mean(axis=0)
    slow_proj = slow_proj**2
    coms = bscan_coms(slow_proj)
    tempvol2 = np.zeros(avol.shape)
    x = np.arange(len(coms))
    p = np.polyfit(x,coms,order)
    fitcoms = np.round(np.polyval(p,x)).astype(np.int16)
    fitcoms = fitcoms-fitcoms.min()
    for x in range(sx):
        z1 = fitcoms[x]
        tempvol2[:,:sz-z1,x] = tempvol[:,z1:,x]

    return tempvol2


def get_flat_offsets(avol,smoothing_sigma=1.0,model_radius=7):
    sy,sz,sx = avol.shape
    # transpose dimensions to permit fft2 and broadcasting
    avol = np.transpose(avol,(1,0,2))

    model = avol[:,sy//2-model_radius:sy//2+model_radius,sx//2-model_radius:sx//2+model_radius].mean(2).mean(1)
    kernel = np.zeros((sy,sx))
    XX,YY = np.meshgrid(np.arange(sx),np.arange(sy))
    XX = XX - sx/2.0
    YY = YY - sy/2.0
    kernel = np.exp(-(XX**2+YY**2)/(2*smoothing_sigma**2))
    kernel = kernel/np.sum(kernel)
    savol = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(avol)*np.fft.fft2(kernel)),axes=(1,2)))

    fmodel = np.fft.fft(model)
    shifts = np.zeros((sy,sx),dtype=np.integer)
    for y in range(sy):
        for x in range(sx):
            p = savol[:,y,x]
            xc = np.abs(np.fft.ifft(np.conj(np.fft.fft(p))*fmodel))
            shift = np.argmax(xc)
            if shift>sz//2:
                shift = shift-sz
            shifts[y,x] = shift
    #shifts = sps.medfilt(shifts,11)
    shifts = shifts - np.min(shifts)
    return shifts

def flythrough(vol,start=0,end=None):
    plt.figure(figsize=(16,12))
    if end is None:
        end=vol.shape[1]
    for k in range(start,end):
        plt.cla()
        plt.imshow(vol[:,k,:],interpolation='none',cmap='gray')
        plt.title(k)
        plt.pause(.1)

def model_segment_volume(vol,model,label_dictionary,gaussian_sigma=0.0,subpixel=True):
    # an attempt at a shortcut to the very slow loop approach below
    # 1. nanreplace the volume with mean value to allow convolution without errors
    avol = np.abs(vol)

    # make a mask to keep nan values nan in the output
    mask = np.sum(avol,axis=1)
    mask = mask/mask
    

    nan_true = np.zeros(mask.shape)
    nan_true[np.where(np.isnan(mask))] = 1
    nan_false = 1-nan_true
    
    n_slow,n_depth,n_fast = avol.shape
    mean_val_vec = np.nanmean(np.nanmean(avol,2),0)
    
    for depth in range(n_depth):
        cut = avol[:,depth,:]
        cut[np.where(np.isnan(cut))] = mean_val_vec[depth]

    if gaussian_sigma>0:
        # 2. make a 3D gaussian kernel with nonzero sigma in fast and slow dimensions,
        #    but depth 1 make it's width and height 10*sigma to get very close to zero
        #    at edges
        diameter = int(np.ceil(gaussian_sigma*10.0))
        if not diameter%2:
            diameter = diameter + 1
        radius = float(diameter)/2.0
        XX,YY = np.meshgrid(np.arange(diameter)-radius,np.arange(diameter)-radius)
        gcut = np.exp(-(XX**2+YY**2)/(2*gaussian_sigma**2))
        kernel = np.zeros((diameter,n_depth,diameter))
        kernel[:,n_depth//2,:] = gcut

        # 3. use fftconvolve to convolve 
        savol = sps.fftconvolve(avol,kernel,mode='same')
    else:
        savol = avol
    #looks like this works?
    #flythrough(avol,170,200)
    #flythrough(savol,170,200)

    # 4. explicitly compute normxcorr to incorporate broadcasting
    #    first we have to move the depth dimension to the back
    savol = np.transpose(savol,(0,2,1))
    avol = np.transpose(avol,(0,2,1))
    
    if len(model)>savol.shape[2]:
        model = model[:savol.shape[2]]
    if len(model)<savol.shape[2]:
        savol = savol[:,:,:len(model)]
        avol = avol[:,:,:len(model)]
        
    nxlen = len(model)
    nxc = np.abs(np.fft.ifft(np.fft.fft(savol,axis=2)*np.conj(np.fft.fft(model))))

    peak = np.argmax(nxc,axis=2)
    peak[np.where(peak>nxlen//2)] = peak[np.where(peak>nxlen//2)] - nxlen

    out_dict = {}

    for key in label_dictionary.keys():
        surf = peak+label_dictionary[key]
        surf = surf*mask
        
        out_dict[key+'_surface'] = surf
        asurf = mask*nan_false.copy()

        for i_slow in range(n_slow):
            print '%d of %d'%(i_slow+1,n_slow)
            for i_fast in range(n_fast):
                # continue if this profile is all nans
                if nan_true[i_slow,i_fast]:
                    continue
                vec = savol[i_slow,i_fast,:]
                plt.plot(vec)
                plt.plot(model)
                print surf[i_slow,i_fast]
                plt.show()
                
                #print surf[i_slow,i_fast],'->',
                try:
                    asurf[i_slow,i_fast] = ascend(vec,surf[i_slow,i_fast],do_plot=False)
                except ValueError as ve:
                    print ve
                    asurf[i_slow,i_fast] = surf[i_slow,i_fast]
                #print surf[i_slow,i_fast]
        out_dict[key+'_surface_ascended'] = asurf

    return out_dict

        
    

def model_segment_volume_slow(vol,model,label_dictionary,gaussian_sigma=0.0):
    # this method must accept volumes with nans; use nanmean instead of mean
    # also, transpose slow/depth to facilitate broadcasting
    vol = np.transpose(vol,(1,0,2))
    n_depth,n_slow,n_fast = vol.shape
    XX,YY = np.meshgrid(np.arange(n_fast),np.arange(n_slow))

    out_dict = {}
    for key in label_dictionary.keys():
        out_dict[key] = np.ones((n_slow,n_fast))*np.nan
    
    for i_slow in range(n_slow):
        print '%d of %d'%(i_slow+1,n_slow)
        for i_fast in range(n_fast):
            print '\t%d of %d'%(i_fast+1,n_fast)
            # continue if this profile is all nans
            if all(np.isnan(vol[:,i_slow,i_fast])):
                continue
            if gaussian_sigma>0.0:
                xx = XX-i_fast
                yy = YY-i_slow
                g = np.exp(-(xx**2+yy**2)/gaussian_sigma**2)
                g = g/np.sum(g)
            
                temp = np.abs(vol)*g
                temp = np.nansum(np.nansum(temp,2),1)
            else:
                temp = np.abs(vol[:,i_slow,i_fast])

            shift,corr = nxcorr(model,temp,do_plot=False)

            for lab in label_dictionary.keys():
                pos = ascend(temp,int(label_dictionary[lab]+shift))
                out_dict[lab][i_slow,i_fast] = int(pos)
    return out_dict
    

def find_cones(data,neighborhood_size,nstd=0.0,do_plot=False):
    threshold = nstd*np.std(data)
    
    #neighborhood = morphology.generate_binary_structure(3,3)
    #rad_ceil = np.ceil(neighborhood_radius)
    
    #XX,YY = np.meshgrid(np.arange(-rad_ceil,rad_ceil+1),np.arange(-rad_ceil,rad_ceil+1))
    #d = np.sqrt(XX**2+YY**2)
    #neighborhood = np.zeros(d.shape)

    #neighborhood[np.where(d<=neighborhood_radius)] = 1.0

    ymax,xmax = data.shape
    ymax = ymax-1
    xmax = xmax-1
    
    data_max = spn.filters.maximum_filter(data, neighborhood_size)
    if do_plot:
        clim = np.percentile(data,(5,99))
        plt.figure()
        plt.imshow(data,cmap='gray',interpolation='none',clim=clim)
        plt.colorbar()
        plt.figure()
        plt.imshow(data_max,cmap='gray',interpolation='none',clim=clim)
        plt.colorbar()
        plt.figure()
        plt.imshow(data_max-data,cmap='gray',interpolation='none',clim=clim)
        plt.colorbar()
    
    maxima = (data == data_max)
    data_min = spn.filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    
    labeled, num_objects = spn.label(maxima)
    slices = spn.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center+1)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center+1)

    outcx,outcy = np.clip(np.array(x,dtype=np.float),0,xmax),np.clip(np.array(y,dtype=np.float),0,ymax)
    outcx = outcx - 1
    outcy = outcy - 1
    if do_plot:
        plt.figure()
        plt.imshow(data,cmap='gray',interpolation='none',clim=clim)
        plt.autoscale(False)
        plt.plot(outcx,outcy,'y+')
        plt.plot(outcx,outcy,'rs')

    return outcx.astype(np.integer),outcy.astype(np.integer)

def project_cones0(vol,peak_threshold=0.5,projection_depth=5,do_plot=False,tag=''):
    # flatten a complex valued volume and project
    # the goal of this is to make volumes just flat enough
    # for en face projection of cone mosaic
    avol = np.abs(vol)

    test_orders = range(5)
    grads = []
    for to in test_orders:
        flatvol = flatten_volume(avol,to)
        grads.append(np.max(np.abs(np.diff(np.mean(np.mean(flatvol,2),0)))))

    flatvol = flatten_volume(avol,test_orders[np.argmax(grads)])

    if False:
        plt.subplot(2,2,1)
        plt.imshow(np.mean(avol,2))
        plt.subplot(2,2,2)
        plt.imshow(np.mean(flatvol,2))
        plt.subplot(2,2,3)
        plt.semilogy(np.mean(np.mean(avol,2),0))
        plt.ylim((100,1100))
        plt.subplot(2,2,4)
        plt.semilogy(np.mean(np.mean(flatvol,2),0))
        plt.ylim((100,1100))
        plt.show()
        sys.exit()
    # NEXT: FIND PEAKS, SEGMENT LAYERS, AND PROJECT

    isos = np.zeros((flatvol.shape[0],flatvol.shape[2]))
    sisos = np.zeros((flatvol.shape[0],flatvol.shape[2]))
    cost = np.zeros((flatvol.shape[0],flatvol.shape[2]))
    srs = np.zeros((flatvol.shape[0],flatvol.shape[2]))
    rpe = np.zeros((flatvol.shape[0],flatvol.shape[2]))

    prof = flatvol[20:-20,:,10:-10].mean(2).mean(0)
    z = np.arange(len(prof))
    peaks = find_peaks(prof)
    peak_threshold = peak_threshold*np.max(prof)
    peaks = peaks[np.where(prof[peaks]>peak_threshold)]

    
    if len(peaks)==3:
        # expected case:
        isos_idx = peaks[0]
        cost_idx = peaks[1]
        rpe_idx = peaks[2]
    elif len(peaks)==2 and np.mean(peaks)>len(prof)//2:
        #the peaks are shifted outward, and IS/OS and COST might be
        #usable
        isos_idx = peaks[0]
        cost_idx = peaks[1]
        rpe_idx = None
        plt.figure()
        plt.plot(z,prof)
        plt.plot(peaks,prof[peaks],'r^')
        plt.title('expected 3 peaks over %0.1f; found these'%peak_threshold)
        try:
            os.mkdir('./project_cones_plots')
        except:
            pass
        plt.savefig('./project_cones_plots/project_cones_peaks_partial_%s.png'%tag)
    else:
        isos_idx = None
        cost_idx = None
        rpe_idx = None
        plt.figure()
        plt.plot(z,prof)
        plt.plot(peaks,prof[peaks],'r^')
        plt.title('expected 3 peaks over %0.1f; found these'%peak_threshold)
        try:
            os.mkdir('./project_cones_plots')
        except:
            pass
        plt.savefig('./project_cones_plots/project_cones_peaks_fail_%s.png'%tag)

    relative_projection_half = (projection_depth-1)//2
    
    try:
        isos = np.mean(flatvol[:,isos_idx-relative_projection_half:isos_idx+relative_projection_half+1,:],1)
    
        cost = np.mean(flatvol[:,cost_idx-relative_projection_half:cost_idx+relative_projection_half+1,:],1)
    except Exception as e:
        print e

    try:
        sisos_idx = (isos_idx+cost_idx)//2-2
        sisos = np.mean(flatvol[:,sisos_idx-relative_projection_half:sisos_idx+relative_projection_half+1,:],1)

        srs = np.mean(flatvol[:,rpe_idx-relative_projection_half:rpe_idx+relative_projection_half+1,:],1)
        rpe = np.mean(flatvol[:,rpe_idx:rpe_idx+2*relative_projection_half+1,:],1)
    except Exception as e:
        print e

    
    
    if do_plot:
        plt.subplot(2,3,1)
        plt.cla()
        plt.plot(prof)
        plt.axvspan(isos_idx-relative_projection_half,isos_idx+relative_projection_half,alpha=0.5,color='g')
        plt.axvspan(cost_idx-relative_projection_half,cost_idx+relative_projection_half,alpha=0.5,color='b')
        #plt.subplot(2,3,2)
        plt.axes([.33,.5,.67,.5])
        plt.cla()
        plt.imshow(np.mean(flatvol,axis=0),cmap='gray',aspect='auto')
        plt.axhspan(isos_idx-relative_projection_half,isos_idx+relative_projection_half,alpha=0.5,color='g')
        plt.axhspan(cost_idx-relative_projection_half,cost_idx+relative_projection_half,alpha=0.5,color='b')

        plt.subplot(2,3,4)
        plt.cla()
        plt.imshow(isos,cmap='gray')

        plt.subplot(2,3,5)
        plt.cla()
        plt.imshow(cost,cmap='gray')

        plt.subplot(2,3,6)
        plt.cla()
        plt.imshow(sisos,cmap='gray')
        plt.pause(.0001)
        
    return isos,sisos,cost,srs,rpe
    
def project_cones(vol,peak_threshold=0.5,projection_depth=5,do_plot=False,tag=''):
    # flatten a complex valued volume and project
    # the goal of this is to make volumes just flat enough
    # for en face projection of cone mosaic
    avol = np.abs(vol)

    flatvol = flatten_volume(avol)

    if False:
        plt.subplot(2,2,1)
        plt.imshow(np.mean(avol,2))
        plt.subplot(2,2,2)
        plt.imshow(np.mean(flatvol,2))
        plt.subplot(2,2,3)
        plt.semilogy(np.mean(np.mean(avol,2),0))
        plt.ylim((100,1100))
        plt.subplot(2,2,4)
        plt.semilogy(np.mean(np.mean(flatvol,2),0))
        plt.ylim((100,1100))
        plt.show()
        sys.exit()
    # NEXT: FIND PEAKS, SEGMENT LAYERS, AND PROJECT

    isos = np.zeros((flatvol.shape[0],flatvol.shape[2]))
    sisos = np.zeros((flatvol.shape[0],flatvol.shape[2]))
    cost = np.zeros((flatvol.shape[0],flatvol.shape[2]))
    srs = np.zeros((flatvol.shape[0],flatvol.shape[2]))
    rpe = np.zeros((flatvol.shape[0],flatvol.shape[2]))

    prof = flatvol[20:-20,:,10:-10].mean(2).mean(0)
    z = np.arange(len(prof))
    peaks = find_peaks(prof)
    peak_threshold = peak_threshold*np.max(prof)
    peaks = peaks[np.where(prof[peaks]>peak_threshold)]

    
    if len(peaks)==3:
        # expected case:
        isos_idx = peaks[0]
        cost_idx = peaks[1]
        rpe_idx = peaks[2]
    elif len(peaks)==2 and np.mean(peaks)>len(prof)//2:
        #the peaks are shifted outward, and IS/OS and COST might be
        #usable
        isos_idx = peaks[0]
        cost_idx = peaks[1]
        rpe_idx = None
        plt.figure()
        plt.plot(z,prof)
        plt.plot(peaks,prof[peaks],'r^')
        plt.title('expected 3 peaks over %0.1f; found these'%peak_threshold)
        try:
            os.mkdir('./project_cones_plots')
        except:
            pass
        plt.savefig('./project_cones_plots/project_cones_peaks_partial_%s.png'%tag)
    else:
        isos_idx = None
        cost_idx = None
        rpe_idx = None
        plt.figure()
        plt.plot(z,prof)
        plt.plot(peaks,prof[peaks],'r^')
        plt.title('expected 3 peaks over %0.1f; found these'%peak_threshold)
        try:
            os.mkdir('./project_cones_plots')
        except:
            pass
        plt.savefig('./project_cones_plots/project_cones_peaks_fail_%s.png'%tag)

    relative_projection_half = (projection_depth-1)//2
    
    try:
        isos = np.mean(flatvol[:,isos_idx-relative_projection_half:isos_idx+relative_projection_half+1,:],1)
    
        cost = np.mean(flatvol[:,cost_idx-relative_projection_half:cost_idx+relative_projection_half+1,:],1)
    except Exception as e:
        print e

    try:
        sisos_idx = (isos_idx+cost_idx)//2-2
        sisos = np.mean(flatvol[:,sisos_idx-relative_projection_half:sisos_idx+relative_projection_half+1,:],1)

        srs = np.mean(flatvol[:,rpe_idx-relative_projection_half:rpe_idx+relative_projection_half+1,:],1)
        rpe = np.mean(flatvol[:,rpe_idx:rpe_idx+2*relative_projection_half+1,:],1)
    except Exception as e:
        print e

    
    
    if do_plot:
        plt.subplot(2,3,1)
        plt.cla()
        plt.plot(prof)
        plt.axvspan(isos_idx-relative_projection_half,isos_idx+relative_projection_half,alpha=0.5,color='g')
        plt.axvspan(cost_idx-relative_projection_half,cost_idx+relative_projection_half,alpha=0.5,color='b')
        #plt.subplot(2,3,2)
        plt.axes([.33,.5,.67,.5])
        plt.cla()
        plt.imshow(np.mean(flatvol,axis=0),cmap='gray',aspect='auto')
        plt.axhspan(isos_idx-relative_projection_half,isos_idx+relative_projection_half,alpha=0.5,color='g')
        plt.axhspan(cost_idx-relative_projection_half,cost_idx+relative_projection_half,alpha=0.5,color='b')

        plt.subplot(2,3,4)
        plt.cla()
        plt.imshow(isos,cmap='gray')

        plt.subplot(2,3,5)
        plt.cla()
        plt.imshow(cost,cmap='gray')

        plt.subplot(2,3,6)
        plt.cla()
        plt.imshow(sisos,cmap='gray')
        plt.pause(.0001)
        
    return isos,sisos,cost,srs,rpe

def get(n):
    return sio.loadmat(flist[n])['Fvolume1']

def com(vec):
    vsum = np.sum(vec)
    idx = np.arange(len(vec))
    return int(round(np.sum(idx*vec)/vsum))

def peak_com(vec,start,rad=0):
    x = np.arange(len(vec))
    if rad==0:
        start = ascend(vec,start)
        left = descend(vec,start-1)+short
        right = descend(vec,start+1)-short
    else:
        left = start-rad
        right = start+rad
    x = x[left:right+1]
    subvec = vec.copy()[left:right+1]
    return np.sum(x*subvec)/np.sum(subvec)

def get_desine_samples(deg1,deg2,n_pixels):
    theta1 = deg1/180.0*np.pi
    theta2 = deg2/180.0*np.pi
    dtheta = (float(theta2)-float(theta1))/n_pixels
    def func(x):
        return np.round((np.arccos(1.0-2.0*x/n_pixels)-theta1)/dtheta).astype(np.int)

    x = np.arange(0,n_pixels)
    fx = func(x)

    return fx[np.where(np.logical_and(fx>=0,fx<n_pixels))]

def pearson(im1,im2):
    # normalize
    im1 = (im1-im1.mean())/im1.std()
    im2 = (im2-im2.mean())/im2.std()
    # I think this should be /2 instead of -1.0; not sure what's wrong
    corr = np.sqrt(np.sum(im1**2*im2**2))/np.sqrt(float(len(im1)))-1.0
    return corr

def four_gauss(x,dc,x0,s0,a0,x1,s1,a1,x2,s2,a2,x3,s3,a3):
    sig0 = a0*np.exp(-(x-x0)**2/(2*s0**2))
    sig1 = a1*np.exp(-(x-x1)**2/(2*s1**2))
    sig2 = a2*np.exp(-(x-x2)**2/(2*s2**2))
    sig3 = a3*np.exp(-(x-x3)**2/(2*s3**2))
    return dc+sig0+sig1+sig2+sig3

def gaussian(x,dc,xmean,s,a):
    sig = a*np.exp(-(x-xmean)**2/(2*s**2))+dc
    return sig

def gaussian_fit(x,y):
    a0 = np.max(y)
    dc0 = 0.0
    fwhm_guess = float(len(np.where(y>y.max()/2.0)[0]))
    # fwhm = 2.355 sigma
    s0 = fwhm_guess/2.355
    xmean0 = x[np.argmax(y)]
    #plt.plot(x,y)
    #plt.plot(x,gaussian(x,dc0,xmean0,s0,a0),'k--')
    #plt.show()
    guess = np.array([dc0,xmean0,s0,a0])
    popt,pvar = spo.curve_fit(gaussian,x,y,guess)
    return popt

def gaussian_convolve(im,sigma,mode='same',hscale=1.0,vscale=1.0):
    if not sigma:
        return im
    else:
        kernel_width = np.ceil(sigma*8) # 4 standard deviations gets pretty close to zero
        vec = np.arange(kernel_width)-kernel_width/2.0
        XX,YY = np.meshgrid(vec,vec)
        g = np.exp(-((XX/hscale)**2+(YY/vscale)**2)/2.0/sigma**2)
        return sps.fftconvolve(im,g,mode=mode)/np.sum(g)


def gaussian_mixture_fit(x,profile,elm,isos,cost,rpe):
    dcg = np.min(profile)
    x0g = elm
    s0g = 5.0
    a0g = profile[elm]-dcg
    x1g = isos
    s1g = 5.0
    a1g = profile[isos]-dcg
    x2g = cost
    s2g = 5.0
    a2g = profile[cost]-dcg
    x3g = rpe
    s3g = 5.0
    a3g = profile[rpe]-dcg
    guess = np.array([dcg,x0g,s0g,a0g,x1g,s1g,a1g,x2g,s2g,a2g,x3g,s3g,a3g])
    popt,pvar = spo.curve_fit(four_gauss,x,profile,guess)
    return popt

def gauss(x,dc,x0,s,a):
    return dc+a*np.exp(-(x-x0)**2/(2*s**2))

def gaussian_com(profile,center,rad,error_limit=5.0,do_plot=True):
    dcg = np.median(profile)
    x0g = 0.0
    ag = profile[center]-dcg
    sg = 5.0
    guess = np.array([dcg,x0g,sg,ag])
    p = profile[center-rad:center+rad+1]
    x = np.arange(2*rad+1)-rad
    try:
        popt,pvar = spo.curve_fit(gauss,x,p,guess)
        offset = popt[1]
    except Exception as e:
        offset = np.sum(p*x)/np.sum(p)

    if offset>error_limit:
        offset=0.0
    print offset
    #pfit = gauss(x,*popt)
    #plt.plot(x,p,'ks')
    #plt.plot(x,pfit,'r--')
    #plt.show()
    return center+offset



def strip_register(target,reference,oversample_factor,strip_width,do_plot=False,use_gaussian=True,background_diameter=0,refine=False):

    # this function returns the x and y shifts required to align lines in TARGET
    # to REFERENCE, such that xshift -1 and yshift +2 for a given line means that
    # it must be moved left one pixel and down two pixels to match REFERENCE
    
    if do_plot:
        plt.figure(figsize=(24,12))

    sy,sx = target.shape
    sy2,sx2 = reference.shape
    #ir_stack = np.zeros((sy,sy,sx))

    assert sy==sy2 and sx==sx2

    # wtf? Why did I have the abs in here originally?
    # ref = np.abs((reference - np.mean(reference)))/np.std(reference)
    # tar = np.abs((target - np.mean(target)))/np.std(target)
                 
    #ref = (reference - np.mean(reference))/np.std(reference)
    #tar = (target - np.mean(target))/np.std(target)

    ref = reference
    tar = target

    #ref = np.random.rand(ref.shape[0],ref.shape[1])
    #tar = ref.copy()
    
    def show(im):
        plt.figure()
        plt.imshow(im,interpolation='none',cmap='gray')
        plt.colorbar()

    x_peaks = []
    y_peaks = []
    goodnesses = []
    
    f1 = np.fft.fft2(ref)
    f1c = f1.conjugate()

    Ny = sy*oversample_factor
    Nx = sx*oversample_factor

    ###ref_autocorr_max = np.max(np.abs(np.fft.ifft2(f1*f1c,s=(Ny,Nx))))
    ref_autocorr_max = np.max(np.abs(np.fft.ifft2(f1*f1c)))
    XX,YY = np.meshgrid(np.arange(Nx),np.arange(Ny))
    
    for iy in range(sy):

        pct_done = round(float(iy)/float(sy)*100)
        if pct_done%10==0:
            print '%d percent done.'%pct_done
        
        # make a y coordinate vector centered about the
        # region of interest
        y = np.arange(sy)-float(iy)

        if use_gaussian:
            # use strip_width/2.0 for sigma (hence no 2x in denom of exponent)
            g = np.exp((-y**2)/(float(strip_width)**2))#/np.sqrt(2*strip_width**2*np.pi)
        else:
            g = np.zeros(y.shape)
            g[np.where(np.abs(y)<=strip_width/2.0)] = 1.0
            #g[iy:iy+strip_width] = 1.0

        temp_tar = (tar.T*g).T
        factor = float(sy)/np.sum(g)

        f0 = np.fft.fft2(temp_tar)
        num = f0*f1c

        # original:
        #num = np.abs(np.fft.ifft2(num,s=(Ny,Nx)))

        # fftshifted:
        num = np.abs(np.fft.ifft2(np.fft.fftshift(num),s=(Ny,Nx)))

        #num = (num-num.mean())/num.std()
        
        #tar_autocorr_max = np.max(np.abs(np.fft.ifft2(f0*f0.conjugate())))
        #denom = np.sqrt(ref_autocorr_max)*np.sqrt(tar_autocorr_max)

        xc = num#/denom
        centered_xc = np.fft.fftshift(xc)
        centered_xc = (centered_xc - centered_xc.mean())/centered_xc.std()
        centered_xc = (centered_xc.T - np.mean(centered_xc,axis=1)).T
        
        #centered_xc = centered_xc - centered_xc.min()

        if background_diameter:
            centered_xc = simple_background_subtract(centered_xc,background_diameter)
        
        xcmax = np.max(centered_xc)
        xcmin = np.min(centered_xc)
        xcstd = np.std(centered_xc)

        cpeaky,cpeakx = np.where(centered_xc==np.max(centered_xc))

        if not len(cpeaky):
            peaky = 0
            peakx = 0
            goodness = 0
        else:
            goodness = centered_xc.max()
            cpeaky = float(cpeaky[0])
            cpeakx = float(cpeakx[0])
            peakx = cpeakx
            peaky = cpeaky
            peakx = peakx - Nx // 2
            peaky = peaky - Ny // 2
            peaky = peaky/oversample_factor
            peakx = peakx/oversample_factor


        if refine:
            refine_rad = oversample_factor*int(strip_width//2)
            refined_tar = np.zeros(tar.shape)
            refined_tar[iy,:] = tar[iy,:]
            refined_f0 = np.fft.fft2(refined_tar)
            refined_num = refined_f0*f1c
            refined_xc = np.abs(np.fft.ifft2(np.fft.fftshift(refined_num),s=(Ny,Nx)))
            #refined_xc = (refined_num-refined_num.mean())/refined_num.std()
            refined_centered_xc = np.fft.fftshift(refined_xc)

            refined_centered_xc = (refined_centered_xc-refined_centered_xc.mean())/refined_centered_xc.std()
            refined_centered_xc = (refined_centered_xc.T - np.mean(refined_centered_xc,axis=1)).T
            #refined_centered_xc = refined_centered_xc - refined_centered_xc.min()
            mask = np.zeros(refined_centered_xc.shape)
            mask[int(cpeaky)-refine_rad:int(cpeaky)+refine_rad+1,int(cpeakx)-refine_rad:int(cpeakx)+refine_rad+1] = 1
            refined_centered_xc = refined_centered_xc * mask

            # choose the refined peak position if its value is almost as high (say
            # within 80%) of the coarse peak
            if refined_centered_xc.max()>.8*centered_xc.max() or True:
                rcpeaky,rcpeakx = np.where(refined_centered_xc==np.max(refined_centered_xc))

                rpeaky = float(rcpeaky[0])
                rpeakx = float(rcpeakx[0])
                rpeakx = rpeakx - Nx // 2
                rpeaky = rpeaky - Ny // 2
                peaky = float(rpeaky)/float(oversample_factor)
                peakx = float(rpeakx)/float(oversample_factor)
                goodness = refined_centered_xc.max()

                if False:#(rcpeaky!=cpeaky or rcpeakx!=cpeakx):
                    disp_factor = 1
                    def norm(im):
                        return (im-im.min())/(im.max()-im.min())
                    plt.figure(figsize=(12,4))
                    plt.subplot(1,3,1)
                    plt.imshow(centered_xc)
                    plt.colorbar()
                    plt.xlim((cpeakx-refine_rad*disp_factor,cpeakx+refine_rad*disp_factor))
                    plt.ylim((cpeaky-refine_rad*disp_factor,cpeaky+refine_rad*disp_factor))
                    plt.subplot(1,3,2)
                    plt.imshow(refined_centered_xc)
                    plt.colorbar()
                    plt.xlim((cpeakx-refine_rad*disp_factor,cpeakx+refine_rad*disp_factor))
                    plt.ylim((cpeaky-refine_rad*disp_factor,cpeaky+refine_rad*disp_factor))
                    plt.subplot(1,3,3)
                    plt.imshow(norm(mask*centered_xc) - norm(refined_centered_xc),cmap='gray')
                    plt.plot(cpeakx,cpeaky,'gs')
                    plt.plot(rcpeakx,rcpeaky,'rs')
                    plt.colorbar()
                    plt.xlim((cpeakx-refine_rad*disp_factor,cpeakx+refine_rad*disp_factor))
                    plt.ylim((cpeaky-refine_rad*disp_factor,cpeaky+refine_rad*disp_factor))
                    plt.title(iy)
                    plt.show()
            

            
            # xl = (cpeakx-20,cpeakx+20)
            # yl = (cpeaky-20,cpeaky+20)
            # plt.figure()
            # plt.imshow(refined_centered_xc,interpolation='none')
            # plt.xlim(xl)
            # plt.ylim(yl)
            # plt.autoscale(False)
            # plt.plot(rcpeakx,rcpeaky,'rs')
            # plt.colorbar()
            # plt.figure()
            # plt.imshow(centered_xc,interpolation='none')
            # plt.xlim(xl)
            # plt.ylim(yl)
            # plt.autoscale(False)
            # plt.plot(cpeakx,cpeaky,'rs')
            # plt.colorbar()
            # plt.show()
            # continue
        
        y_peaks.append(peaky)
        x_peaks.append(peakx)
        goodnesses.append(goodness)
        half_window = 20
        if do_plot:
            plt.clf()

            plt.subplot(2,4,1)
            plt.cla()
            plt.imshow(centered_xc,cmap='gray',interpolation='none')
            plt.colorbar()
            plt.subplot(2,4,2)
            plt.cla()
            plt.plot(x_peaks,label='x')
            plt.plot(y_peaks,label='y')
            plt.legend()
            plt.title('lags')
            
            plt.subplot(2,4,3)
            plt.cla()
            plt.plot(goodnesses)
            plt.title('goodness')
            plt.subplot(2,4,4)
            plt.cla()

            clim = (np.min(centered_xc),np.max(centered_xc))
            plt.imshow(centered_xc,cmap='gray',interpolation='none',aspect='auto',clim=clim)
            plt.xlim((cpeakx-half_window,cpeakx+half_window))
            plt.ylim((cpeaky-half_window,cpeaky+half_window))
            
            #plt.imshow(centered_xc[peaky+Ny//2-half_window:peaky+Ny//2+half_window,peakx+Nx//2-half_window:peakx+Nx//2+half_window],cmap='gray',interpolation='none',aspect='auto')
            #plt.colorbar()

            clim = (tar.min(),tar.max())
            plt.subplot(2,4,5)
            plt.cla()
            plt.imshow(tar,cmap='gray',interpolation='none',aspect='auto',clim=clim)

            plt.subplot(2,4,6)
            plt.cla()
            plt.imshow(temp_tar,cmap='gray',interpolation='none',aspect='auto',clim=clim)

            plt.subplot(2,4,7)
            plt.cla()
            plt.imshow(ref,cmap='gray',interpolation='none',aspect='auto',clim=clim)
            plt.pause(.0001)
        
    if do_plot:
        plt.close()
            
    return y_peaks,x_peaks,goodnesses




















def rb_strip_register(target,reference,oversample_factor,strip_width,do_plot=False,rb_xmax=5,rb_ymax=5):

    # this function returns the x and y shifts required to align lines in TARGET
    # to REFERENCE, such that xshift -1 and yshift +2 for a given line means that
    # it must be moved left one pixel and down two pixels to match REFERENCE
    
    if do_plot:
        plt.figure()
        
    sy,sx = target.shape
    sy2,sx2 = reference.shape
    #ir_stack = np.zeros((sy,sy,sx))

    assert sy==sy2 and sx==sx2

    ref = reference
    tar = target
    
    def show(im):
        plt.figure()
        plt.imshow(im,interpolation='none',cmap='gray')
        plt.colorbar()

    x_peaks = []
    y_peaks = []
    goodnesses = []

    f1 = np.fft.fft2(ref)
    f1c = f1.conjugate()


    f2 = np.fft.fft2(tar)
    rb_nxc = np.fft.fftshift(np.abs(np.fft.ifft2(f1c*f2)))
    plt.subplot(2,2,1)
    plt.imshow(ref,cmap='gray')
    plt.title('ref')
    plt.subplot(2,2,2)
    plt.imshow(tar,cmap='gray')
    plt.title('tar')
    plt.subplot(2,2,3)
    plt.imshow(rb_nxc,cmap='gray')
    plt.subplot(2,2,4)
    
    peaky,peakx = np.unravel_index(np.argmax(rb_nxc),rb_nxc.shape)
    rb_yshift = peaky-sy//2
    rb_xshift = peakx-sx//2

    if False:
        test_size = 50
        ref_test = ref[:test_size,:test_size]
        tar_test = tar[rb_yshift:rb_yshift+test_size,rb_xshift:rb_xshift+test_size]
        plt.figure()
        plt.imshow(ref_test)
        plt.figure()
        plt.imshow(tar_test)

        plt.figure()
        plt.imshow(ref_test-tar_test)

        plt.figure()
        plt.imshow(rb_nxc)
        plt.show()

        sys.exit()

    
    Ny = sy*oversample_factor
    Nx = sx*oversample_factor

    for iy in range(sy):

        pct_done = round(float(iy)/float(sy)*100)
        if pct_done%10==0:
            print '%d percent done.'%pct_done



        ref_y1 = iy-strip_width//2
        ref_y2 = iy+strip_width//2

        tar_y1 = ref_y1 + rb_yshift
        tar_y2 = ref_y2 + rb_yshift

        no_overlap = False
        while ref_y1<0 or tar_y1<0:
            ref_y1 += 1
            tar_y1 += 1
            if ref_y1>ref_y2:
                no_overlap = True
                break
            if tar_y1>tar_y2:
                no_overlap = True
                break

        while ref_y2>=sy or tar_y2>=sy:
            ref_y2 -= 1
            tar_y2 -= 1
            if ref_y2<ref_y1:
                no_overlap = True
                break
            if tar_y2<tar_y1:
                no_overlap = True
                break

        if no_overlap:
            x_peaks.append(np.nan)
            y_peaks.append(np.nan)
            goodnesses.append(np.nan)
            continue

        ref_strip = ref[ref_y1:ref_y2+1,:]
        tar_strip = tar[tar_y1:tar_y2+1,:]

        ref_strip = (ref_strip - ref_strip.mean())/ref_strip.std()
        tar_strip = (tar_strip - tar_strip.mean())/tar_strip.std()
        
        ssy,ssx = ref_strip.shape
        Ny,Nx = ssy*oversample_factor,ssx*oversample_factor


        
        rsf_conj = np.conj(np.fft.fft2(ref_strip))
        tsf = np.fft.fft2(tar_strip)

        nxc = np.fft.fftshift(np.abs(np.fft.ifft2(np.fft.fftshift(rsf_conj*tsf),s=(Ny,Nx))))

        if do_plot:
            plt.cla()
            plt.imshow(nxc,aspect='auto')
            plt.pause(.1)
        
        x1 = rb_xshift - rb_xmax + ssx//2
        x2 = rb_xshift + rb_xmax+1 + ssx//2
        x1 = x1*oversample_factor
        x2 = x2*oversample_factor
        nxc[:,:x1] = np.nan
        nxc[:,x2:] = np.nan


        if nxc.shape[0]>oversample_factor*(2*rb_ymax+1):
            y1 = strip_width//2-rb_ymax
            y2 = strip_width//2+rb_ymax+1
            y1 = y1*oversample_factor
            y2 = y2*oversample_factor
            nxc[:y1,:] = np.nan
            nxc[y2:,:] = np.nan
            


        if False:
            plt.subplot(3,1,1)
            plt.imshow(ref_strip,aspect='auto')
            plt.subplot(3,1,2)
            plt.imshow(tar_strip,aspect='auto')
            plt.subplot(3,1,3)
            plt.imshow(nxc,aspect='auto')
            plt.colorbar()
            plt.show()
            continue


        cpeaky,cpeakx = np.where(nxc==np.nanmax(nxc))
        if not len(cpeaky):
            peaky = np.nan
            peakx = np.nan
            goodness = np.nan
        else:
            goodness = np.nanmax(nxc)
            cpeaky = float(cpeaky[0])
            cpeakx = float(cpeakx[0])
            peakx = cpeakx
            peaky = cpeaky
            peakx = peakx - Nx // 2
            peaky = peaky - Ny // 2
            peakx = peakx/oversample_factor
            peaky = peaky/oversample_factor
            peaky = peaky+rb_yshift
            
        y_peaks.append(peaky)
        x_peaks.append(peakx)
        goodnesses.append(goodness)
            
    return y_peaks,x_peaks,goodnesses






















def graph_strip_register(target,reference,oversample_factor,strip_width,do_plot=False,use_gaussian=True,background_diameter=0,refine=False):

    # this function returns the x and y shifts required to align lines in TARGET
    # to REFERENCE, such that xshift -1 and yshift +2 for a given line means that
    # it must be moved left one pixel and down two pixels to match REFERENCE
    
    if do_plot:
        plt.figure(figsize=(24,12))

    sy,sx = target.shape
    sy2,sx2 = reference.shape
    #ir_stack = np.zeros((sy,sy,sx))

    
    assert sy==sy2 and sx==sx2

    # wtf? Why did I have the abs in here originally?
    # ref = np.abs((reference - np.mean(reference)))/np.std(reference)
    # tar = np.abs((target - np.mean(target)))/np.std(target)
                 
    #ref = (reference - np.mean(reference))/np.std(reference)
    #tar = (target - np.mean(target))/np.std(target)

    ref = reference
    tar = target

    #ref = np.random.rand(ref.shape[0],ref.shape[1])
    #tar = ref.copy()
    
    def show(im):
        plt.figure()
        plt.imshow(im,interpolation='none',cmap='gray')
        plt.colorbar()

    x_peaks = []
    y_peaks = []
    goodnesses = []
    
    f1 = np.fft.fft2(ref)
    f1c = f1.conjugate()

    Ny = sy*oversample_factor
    Nx = sx*oversample_factor

    ###ref_autocorr_max = np.max(np.abs(np.fft.ifft2(f1*f1c,s=(Ny,Nx))))
    ref_autocorr_max = np.max(np.abs(np.fft.ifft2(f1*f1c)))
    XX,YY = np.meshgrid(np.arange(Nx),np.arange(Ny))



    nxc_stack = np.zeros((sy,Ny,Nx))
    
    
    for iy in range(sy):

        pct_done = round(float(iy)/float(sy)*100)
        if pct_done%10==0:
            print '%d percent done.'%pct_done
        
        # make a y coordinate vector centered about the
        # region of interest
        y = np.arange(sy)-float(iy)

        if use_gaussian:
            # use strip_width/2.0 for sigma (hence no 2x in denom of exponent)
            g = np.exp((-y**2)/(float(strip_width)**2))#/np.sqrt(2*strip_width**2*np.pi)
        else:
            g = np.zeros(y.shape)
            g[np.where(np.abs(y)<=strip_width/2.0)] = 1.0
            #g[iy:iy+strip_width] = 1.0

        temp_tar = (tar.T*g).T
        factor = float(sy)/np.sum(g)

        f0 = np.fft.fft2(temp_tar)
        num = f0*f1c

        # original:
        #num = np.abs(np.fft.ifft2(num,s=(Ny,Nx)))

        # fftshifted:
        num = np.abs(np.fft.ifft2(np.fft.fftshift(num),s=(Ny,Nx)))

        #num = (num-num.mean())/num.std()
        
        #tar_autocorr_max = np.max(np.abs(np.fft.ifft2(f0*f0.conjugate())))
        #denom = np.sqrt(ref_autocorr_max)*np.sqrt(tar_autocorr_max)

        xc = num#/denom
        centered_xc = np.fft.fftshift(xc)
        centered_xc = (centered_xc - centered_xc.mean())/centered_xc.std()
        centered_xc = (centered_xc.T - np.mean(centered_xc,axis=1)).T


        nxc_stack[iy,:,:] = centered_xc
        
        continue
        xcmax = np.max(centered_xc)
        xcmin = np.min(centered_xc)
        xcstd = np.std(centered_xc)

        cpeaky,cpeakx = np.where(centered_xc==np.max(centered_xc))

        goodness = centered_xc.max()
        cpeaky = float(cpeaky[0])
        cpeakx = float(cpeakx[0])
        peakx = cpeakx
        peaky = cpeaky
        peakx = peakx - Nx // 2
        peaky = peaky - Ny // 2
        peaky = peaky/oversample_factor
        peakx = peakx/oversample_factor
        
        y_peaks.append(peaky)
        x_peaks.append(peakx)

    nxc_stack = nxc_stack**2



    # dead simple approach:
    # find the brightest pixel in nxc_stack and
    # cost function ourselves to the ends 0 and sy-1
    pz0,py0,px0 = np.unravel_index(np.argmax(nxc_stack),nxc_stack.shape)

    for z in range(sy):
        plt.clf()
        plt.imshow(nxc_stack[z,:,:],clim=(0,1))
        plt.colorbar()
        plt.pause(.1)
    sys.exit()

    
    nsz,nsy,nsx = nxc_stack.shape
    
    def make_mask(sy,sx,cy,cx,rad):
        XX,YY = np.meshgrid(np.arange(sx),np.arange(sy))
        XX = XX-cx
        YY = YY-cy
        d = np.sqrt(XX**2+YY**2)
        mask = np.zeros(d.shape)
        mask[np.where(d<=rad)] = 1
        return mask
    
    gcpeaky_vec = np.zeros((nsz))
    gcpeakx_vec = np.zeros((nsz))
    ggoodness_vec = np.zeros((nsz))

    ggoodness_vec[pz0] = nxc_stack[pz0,py0,px0]
    gcpeaky_vec[pz0] = py0
    gcpeakx_vec[pz0] = px0
    
    py = py0
    px = px0
    zrange = range(pz0-1,-1,-1)+range(pz0+1,sy)
    for z in zrange:
        mask = make_mask(nsy,nsx,py,px,oversample_factor*2)
        temp0 = nxc_stack[z,:,:]
        temp = temp0*mask
        py,px = np.unravel_index(np.argmax(temp),temp.shape)
        if False:
            plt.subplot(1,2,1)
            plt.cla()
            plt.imshow(mask)
            plt.subplot(1,2,2)
            plt.cla()
            plt.imshow(temp0,cmap='gray',clim=(0.0,0.6))
            plt.autoscale(False)
            plt.plot(px,py,'rs',alpha=0.15,markersize=16,markerfacecolor=None)
            plt.title(z)
            plt.pause(.1)
        ggoodness_vec[z] = temp[py,px]
        gcpeaky_vec[z] = py
        gcpeakx_vec[z] = px

    corr_peaks = np.max(np.max(nxc_stack,axis=2),axis=1)
    cpeaky_vec = []
    cpeakx_vec = []
    for idx,cp in enumerate(corr_peaks):
        y,x = np.where(nxc_stack[idx,:,:]==cp)
        try:
            cpeaky_vec.append(y[0])
            cpeakx_vec.append(x[0])
        except Exception as e:
            print e
            cpeaky_vec.append(np.nan)
            cpeakx_vec.append(np.nan)

    # find flat regions of cpeaky_vec and cpeakx_vec,
    # i.e. regions in which point to point differences
    # are smaller than a given threshold
    cpeakx_vec = np.array(cpeakx_vec)
    cpeaky_vec = np.array(cpeaky_vec)

    plt.subplot(2,1,1)
    plt.plot(cpeakx_vec)
    plt.plot(gcpeakx_vec)
    #plt.plot(flat_indices,cpeakx_vec[flat_indices],'ro')
    plt.plot(cpeaky_vec)
    plt.plot(gcpeaky_vec)

    #plt.plot(flat_indices,cpeaky_vec[flat_indices],'ro')
    plt.subplot(2,1,2)
    plt.plot(np.arange(sy),corr_peaks)
    plt.plot(np.arange(sy),ggoodness_vec)

    plt.show()
        
    return y_peaks,x_peaks,goodnesses
