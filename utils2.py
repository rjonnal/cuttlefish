import numpy as np
from matplotlib import pyplot as plt

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


    if False:
        if l1%2!=l2%2:
            shift = np.fix(peakIdx-len(nxcval)/2.0)
        else:
            shift = np.fix(peakIdx-len(nxcval)/2.0) + 1

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
        sys.exit()

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


def find_peaks(prof,intensity_threshold=-np.inf,gradient_threshold=-np.inf):
    left = prof[:-2]
    center = prof[1:-1]
    right = prof[2:]
    peaks = np.where(np.logical_and(center>right,center>left))[0]+1
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

def flatten_volume(avol,order=2):
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

def project_cones(vol,peak_threshold=500.0,projection_depth=5,do_plot=False):
    # flatten a complex valued volume and project
    # the goal of this is to make volumes just flat enough
    # for en face projection of cone mosaic
    avol = np.abs(vol)
        
    flatvol = flatten_volume(flatten_volume(avol),5)

    # NEXT: FIND PEAKS, SEGMENT LAYERS, AND PROJECT
    prof = flatvol.mean(2).mean(0)
    z = np.arange(len(prof))
    peaks = find_peaks(prof)
    peaks = peaks[np.where(prof[peaks]>peak_threshold)]

    
    # expected case:
    if len(peaks)==3:
        isos_idx = peaks[0]
        cost_idx = peaks[1]
    else:
        plt.plot(z,prof)
        plt.plot(peaks,prof[peaks],'r^')
        plt.title('expected 3 peaks over %0.1f; found these'%peak_threshold)
        plt.show()
        sys.exit()

        
    relative_projection_half = (projection_depth-1)//2
    isos = np.mean(flatvol[:,isos_idx-relative_projection_half:isos_idx+relative_projection_half+1,:],1)
    
    cost = np.mean(flatvol[:,cost_idx-relative_projection_half:cost_idx+relative_projection_half+1,:],1)

    sisos_idx = (isos_idx+cost_idx)//2-2
    sisos = np.mean(flatvol[:,sisos_idx-relative_projection_half:sisos_idx+relative_projection_half+1,:],1)
    
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
        
    return isos,sisos,cost
    
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
    popt,pvar = soo.curve_fit(four_gauss,x,profile,guess)
    return popt

def gauss(x,dc,x0,s,a):
    return dc+a*np.exp(-(x-x0)**2/(2*s**2))

def gaussian_com(profile,center,rad,error_limit=5.0):
    dcg = np.median(profile)
    x0g = 0.0
    ag = profile[center]-dcg
    sg = 5.0
    guess = np.array([dcg,x0g,sg,ag])
    p = profile[center-rad:center+rad+1]
    x = np.arange(2*rad+1)-rad
    try:
        popt,pvar = soo.curve_fit(gauss,x,p,guess)
        offset = popt[1]
    except Exception as e:
        offset = np.sum(p*x)/np.sum(p)

    if offset>error_limit:
        offset=0.0
    #pfit = gauss(x,*popt)
    #plt.plot(x,p,'ks')
    #plt.plot(x,pfit,'r--')
    #plt.show()
    return center+offset

