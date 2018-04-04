import numpy as np
from matplotlib import pyplot as plt
import sys,os
from hive import Hive
import utils
import glob
from scipy.ndimage import zoom
from scipy.interpolate import griddata
from scipy.signal import fftconvolve,medfilt
from scipy.optimize import curve_fit
from scipy.io import savemat
import matplotlib.colors as colors
import matplotlib.cm as cmx
from fig2gif import GIF


class Cone:

    def __init__(self,vol,isos_idx,cost_idx,x,y,origin='',x0=None,y0=None,properties={}):
        """This class stores information about a single cone. Required
        parameters are a 3D array representing the cone's reflectivity;
        indices of the ISOS and COST reflections; the x and y
        coordinates of the cone in the reference coordinate
        space. Optional params are a string describing the cone's
        original data set (e.g. the dataset tag and volume index); x0
        and y0, the coordinates of the cone in its original volume; and
        a dictionary of other properties, e.g. whether the cone had been
        stimulated or not, its spectral class, etc."""
        
        self.iidx = isos_idx
        self.cidx = cost_idx
        self.vol = vol
        self.x = x
        self.y = y
        self.origin = origin
        self.x0 = x0
        self.y0 = y0
        self.properties = properties
        self.prof = np.mean(np.mean(np.abs(vol),1),0)


    def gfunc(self,x,x0,s,a):
        XX = np.arange(len(x)) - x0
        g = np.exp(-(XX)**2/(2*s**2))*a
        return g

    def gaussian_fit(self,xvec,yvec):
        x = np.argmax(yvec)
        a = np.max(yvec)
        s = 1.0
        p = [x,s,a]
        try:
            params,errs = curve_fit(self.gfunc,xvec,yvec,p0=p,ftol=.01)
        except Exception as e:
            params = p
        return params
        
    def isos_subpixel(self,bias=None):
        if bias is None:
            bias = np.min(self.prof)
        #bias = 0.0
        prof = self.prof - bias
        ileft,iright = utils.peak_edges(prof,self.iidx)
        prof[:ileft] = 0.0
        prof[iright:] = 0.0
        x,s,a = self.gaussian_fit(np.arange(len(prof)),prof)
        a = a+bias
        #plt.plot(self.prof)
        #plt.plot(self.gfunc(np.arange(len(prof)),x,s,a))
        #plt.show()
        return x,s,a
        
    def cost_subpixel(self,bias=None):
        if bias is None:
            bias = np.min(self.prof)
        #bias = 0.0
        prof = self.prof - bias
        ileft,iright = utils.peak_edges(prof,self.cidx)
        prof[:ileft] = 0.0
        prof[iright:] = 0.0
        x,s,a = self.gaussian_fit(np.arange(len(prof)),prof)
        a = a+bias
        #plt.plot(self.prof)
        #plt.plot(self.gfunc(np.arange(len(prof)),x,s,a))
        #plt.show()
        return x,s,a
        
    def gaussian_mixture_model(self,do_plot=False):

        if do_plot:
            plt.figure()


        if len(self.prof)>9:
            testprof = self.prof
        else:
            testprof = np.random.randn(10)
            testprof[:len(self.prof)] = self.prof
            
        def gmm(x,x00,x0mid,x01,s0,smid,s1,a0,amid,a1):
            fit = np.zeros(len(x))
            
            x0vec = [x00,x0mid,x01]
            svec = [s0,smid,s1]
            avec = [a0,amid,a1]
            
            for x0,s,a in zip(x0vec,svec,avec):
                XX0 = np.arange(len(x)) - x0
                fit = fit+np.exp(-(XX0**2)/(2*s**2))*a
            if do_plot:
                plt.cla()
                plt.plot(x,fit,lw=2)
                plt.plot(x,testprof,lw=2)
                plt.pause(.00000001)
            return fit

        # initial guess:
        p_a0 = testprof[self.iidx]
        p_a1 = testprof[self.cidx]
        p_amid = 0.0
        
        p_s0 = 1.0
        p_s1 = 1.0
        p_smid = 1.0
        
        p_x0 = self.iidx
        p_x1 = self.cidx
        p_xmid = (p_x0+p_x1)/2.0
        
        
        p = [p_x0,p_xmid,p_x1,p_s0,p_smid,p_s1,p_a0,p_amid,p_a1]
        #lower = [p_x0-1.0,p_x1-1.0,0.1,0.1,p_a0*.75,p_a1*.75]
        #upper = [p_x0+1.0,p_x1+1.0,3.0,3.0,p_a0*1.25,p_a1*1.25]
        #bounds = (lower,upper)
        
        aa,bb = curve_fit(gmm,np.arange(len(testprof)),testprof,p0=p,ftol=.0001)
        
        if do_plot:
            plt.close()

        print 'bye'
        return aa,bb
        


class Database:

    def __init__(self,filename,sep='\t'):
        self.filename = filename
        self.sep = sep
        self.dictionary = {}
        try:
            fid = open(self.filename,'r')
            for line in fid:
                key,val = self.parse(line)
                self.dictionary[key] = val
        except IOError:
            pass

    def keys(self):
        return self.dictionary.keys()
    
    def parse(self,file_line):
        contents = file_line.split(self.sep)
        return contents[0],[contents[1],int(contents[2])]
        
    def put(self,key,target_filename,volume_index):
        fid = open(self.filename,'a')
        outstring = '%s\t%s\t%d\n'%(key,target_filename,volume_index)
        fid.write(outstring)
        fid.close()
        self.dictionary[key] = [target_filename,volume_index]
        
    def get(self,key):
        return self.dictionary[key]
        
class Series:

    def __init__(self,series_directory,reference_frame=None):
        self.hive = Hive(series_directory)
        self.n_frames = 0
        self.series_directory = series_directory
        self.db = Database(os.path.join(self.series_directory,'target_db.txt'))
            
        if not reference_frame is None:
            self.reference = reference_frame
            self.hive.put('/reference_frame',reference_frame)
        else:
            try:
                self.reference = self.hive['/reference_frame'][:,:]
            except Exception as e:
                print 'Warning: empty Series started w/o reference frame.'

    def set_reference_frame(self,reference_frame):
        self.reference = reference_frame
        self.hive.put('/reference_frame',reference_frame)

    def crop_borders(self,im):
        # remove edges of im that contain no information
        vprof = np.std(im,axis=1)
        valid_region_y = np.where(vprof)[0]
        hprof = np.std(im,axis=0)
        valid_region_x = np.where(hprof)[0]
        y1 = valid_region_y[0]
        y2 = valid_region_y[-1]
        x1 = valid_region_x[0]
        x2 = valid_region_x[-1]
        return im[y1:y2,x1:x2]

    def make_cone_catalog(self,points,minimum_goodness=10.0,output_radius=2,match_radius=2.0,do_plot=False):
        """Take a set of points in this Series' reference image,
        corresponding to the x and y coordinate of cone centers,
        and identify and crop the corresponding cone out of all
        this Series' volumes."""


        vdict = self.get_volume_dictionary()

        nvols = len(vdict.keys())
        ncones = len(points)

        os_length_sheet = np.zeros((ncones,nvols))
        isos_intensity_sheet = np.zeros((ncones,nvols))
        cost_intensity_sheet = np.zeros((ncones,nvols))
        
        fkeys = self.hive['/frames'].keys()
        for fkidx,fk in enumerate(fkeys):
            dataset_fn = os.path.join(self.working_directory,fk)
            dataset_h5 = H5(dataset_fn)
            ikeys = self.hive['/frames'][fk].keys()
            for ikidx,ik in enumerate(ikeys):
                vol = dataset_h5['/flattened_data'][int(ik),:,:,:]
                cost_depths = dataset_h5['model/volume_labels/COST'][int(ik),:,:]
                isos_depths = dataset_h5['model/volume_labels/ISOS'][int(ik),:,:]

                x = self.hive['/frames'][fk][ik]['x_shifts'][:]
                y = self.hive['/frames'][fk][ik]['y_shifts'][:]
                c = self.hive['/frames'][fk][ik]['correlations'][:]
                g = self.hive['/frames'][fk][ik]['goodnesses'][:]
                
                model_profile = dataset_h5['model/profile'][:]
                model_isos = dataset_h5['model/labels/ISOS'].value
                model_cost = dataset_h5['model/labels/COST'].value
                yramp = np.arange(len(y)) - y
                # yramp[n] is now the location of the nth target row in the reference space
                # so we need to find n such that yramp[n]-pty is minimized
                
                cmed = int(np.median(cost_depths))
                imed = int(np.median(isos_depths))
                volume_enface_projection = np.abs(vol[:,imed-2:cmed+2,:]).mean(axis=1)

                border = 3
                
                for idx,(ptx,pty) in enumerate(points):
                    cone_idx = idx
                    print 'frame %d/%d; vol %d/%d; cone %d/%d at %d,%d'%(fkidx+1,len(fkeys),ikidx+1,len(ikeys),idx+1,len(points),ptx,pty)
                    yerr = np.abs(pty-yramp)
                    match_index = np.argmin(yerr)
                    if yerr[match_index]<=match_radius and g[match_index]>minimum_goodness:
                        #print 'match exists'
                        # get the target coordinates, and then ascend the target en face projection
                        # to the peak (center) of the cone:
                        yout = int(match_index)
                        xout = int(ptx + x[match_index])
                        xout,yout = utils.ascend2d(volume_enface_projection,xout,yout,do_plot=False)
                        
                        # 3D-crop the cone out of the volume, and make an axial profile from it
                        try:
                            cone_volume = self.get_subvol(vol,yout,0,xout,output_radius,np.inf,output_radius)
                            cone_profile = np.mean(np.mean(np.abs(cone_volume),axis=2),axis=0)
                            shift,corr = utils.nxcorr(model_profile,cone_profile)
                        except Exception as e:
                            print e
                            continue

                        # get some overall stats of the volume
                        # we'll write these to the hdf5 file later
                        avol = np.abs(cone_volume)
                        aprof = np.mean(np.mean(avol,axis=2),axis=0)
                        noise_floor = aprof[np.argsort(aprof)[:10]]
                        noise_mean = noise_floor.mean()
                        noise_std = noise_floor.std()
                        full_volume_mean = avol.mean()
                        full_volume_std = avol.std()
                        full_profile_mean = aprof.mean()
                        full_profile_std = aprof.std()
                        
                        
                        # let's try to label this cone's peaks
                        #print 'mp',model_profile
                        #print 'cp',cone_profile
                        isos_guess = int(model_isos + shift)
                        cost_guess = int(model_cost + shift)

                        # now use the height of the cone_profile at
                        # these guesses, combined with locations and
                        # distances of other peaks to refine the guesses
                        max_displacement = 4
                        peaks = utils.find_peaks(cone_profile)
                        heights = cone_profile[peaks]/cone_profile.std()
                        isos_dpeaks = np.abs(peaks-isos_guess)
                        isos_dpeaks[np.where(isos_dpeaks>=max_displacement)] = 2**16
                        cost_dpeaks = np.abs(peaks-cost_guess)
                        cost_dpeaks[np.where(cost_dpeaks>=max_displacement)] = 2**16
                        isos_scores = heights - isos_dpeaks
                        cost_scores = heights - cost_dpeaks
                        isos_guess = peaks[np.argmax(isos_scores)]
                        cost_guess = peaks[np.argmax(cost_scores)]

                        # now cross correlate the cuts through the cone with
                        # cone_profile to see if axial eye movement between
                        # b-scans has resulted in a peak shift
                        sy,sz,sx = cone_volume.shape
                        colors = 'rgbkcym'
                        sheet = []
                        for idx,coney in enumerate(range(sy)):
                            color = colors[idx%len(colors)]
                            cut_profile = np.abs(cone_volume[coney,:,:]).mean(axis=1)
                            shift,corr = utils.nxcorr(cut_profile,cone_profile)
                            z1 = int(isos_guess-border-shift)
                            z2 = int(cost_guess+border-shift)
                            cut = cone_volume[coney,z1:z2,:]
                            sheet.append(cut)

                        sheet = np.array(sheet)
                        sheet = np.transpose(sheet,(0,2,1))
                        point_string = '%d_%d'%(ptx,pty)
                        os_length = cost_guess-isos_guess
                        key_root = '/cone_catalog/%s/%s/%s'%(point_string,fk,ik)

                        c = Cone(sheet,border,border+os_length,ptx,pty)

                        isos_z,isos_s,isos_a = c.isos_subpixel(bias=noise_mean)
                        cost_z,cost_s,cost_a = c.cost_subpixel(bias=noise_mean)

                        self.hive.put('%s/x'%key_root,xout)
                        self.hive.put('%s/y'%key_root,yout)
                        self.hive.put('%s/isos_z'%key_root,border)
                        self.hive.put('%s/cost_z'%key_root,border+os_length)
                        self.hive.put('%s/cone_volume'%key_root,sheet)
                        self.hive.put('%s/noise_mean'%key_root,noise_mean)
                        self.hive.put('%s/noise_std'%key_root,noise_std)
                        self.hive.put('%s/full_volume_mean'%key_root,full_volume_mean)
                        self.hive.put('%s/full_volume_std'%key_root,full_volume_std)
                        self.hive.put('%s/full_profile_mean'%key_root,full_profile_mean)
                        self.hive.put('%s/full_profile_std'%key_root,full_profile_std)
                        
                        self.hive.put('%s/subpixel/isos_z'%key_root,isos_z)
                        self.hive.put('%s/subpixel/cost_z'%key_root,cost_z)
                        self.hive.put('%s/subpixel/isos_sigma'%key_root,isos_s)
                        self.hive.put('%s/subpixel/cost_sigma'%key_root,cost_s)
                        self.hive.put('%s/subpixel/isos_amplitude'%key_root,isos_a)
                        self.hive.put('%s/subpixel/cost_amplitude'%key_root,cost_a)

                        vidx = vdict[(fk,ik)]
                        os_length = cost_z - isos_z
                        os_length_sheet[cone_idx,vidx] = os_length
                        isos_intensity_sheet[cone_idx,vidx] = isos_a/full_profile_mean
                        cost_intensity_sheet[cone_idx,vidx] = cost_a/full_profile_mean


            plt.subplot(1,3,1)
            plt.imshow(os_length_sheet,aspect='normal')
            plt.colorbar()
            plt.subplot(1,3,2)
            plt.imshow(isos_intensity_sheet,aspect='normal')
            plt.colorbar()
            plt.subplot(1,3,3)
            plt.imshow(cost_intensity_sheet,aspect='normal')
            plt.colorbar()
            plt.show()


            
            
        np.save('os_length.npy',os_length_sheet)
        np.save('isos_intensity.npy',isos_intensity_sheet)
        np.save('cost_intensity.npy',cost_intensity_sheet)
        
                        
    def get_n_volumes(self):
        count = 0
        fkeys = self.hive['/frames'].keys()
        for fk in fkeys:
            ikeys = self.hive['/frames'][fk].keys()
            count = count + len(ikeys)
        return count


    def get_volume_dictionary(self,order=None):
        d = {}
        counter = 0
        frame_keys = self.hive['/frames'].keys()
        if order is not None:
            frame_keys = frame_keys[order]
        for fk in frame_keys:
            index_keys = self.hive['/frames/%s'%fk].keys()
            for ik in index_keys:
                d[(fk,ik)] = counter
                counter = counter + 1
        return d
                
    
    def get_n_cones(self):
        try:
            n_cones = self.hive['cone_catalog/globals/n_cones']
        except Exception as e:
            cone_catalog = self.hive['cone_catalog']
            cone_keys = cone_catalog.keys()
            n_cones = len(cone_keys)
            self.hive.put('/cone_catalog/globals/n_cones',n_cones)
        return n_cones
    
    def get_cone_volume_size(self):
        fast_maxes = []
        try:
            slow_max = self.hive['cone_catalog/globals/slow_max']
            fast_max = self.hive['cone_catalog/globals/fast_max']
            depth_max = self.hive['cone_catalog/globals/depth_max']
        except Exception as e:
            cone_catalog = self.hive['cone_catalog']
            cone_keys = cone_catalog.keys()
            slow_max,fast_max,depth_max = 0,0,0
            for ck in cone_keys:
                if ck in ['globals']:
                    continue
                frame_keys = cone_catalog['%s'%ck].keys()
                for fk in frame_keys:
                    index_keys = cone_catalog['%s/%s'%(ck,fk)].keys()
                    for ik in index_keys:
                        dims = cone_catalog['%s/%s/%s/cone_volume'%(ck,fk,ik)].shape
                        print ck,fk,dims
                        if dims[1]>5:
                            continue
                        if dims[0]>slow_max:
                            slow_max = dims[0]
                        if dims[1]>fast_max:
                            fast_max = dims[1]
                        if dims[2]>depth_max:
                            depth_max = dims[2]
                        fast_maxes.append(dims[1])
                        
            self.hive.put('/cone_catalog/globals/slow_max',slow_max)
            self.hive.put('/cone_catalog/globals/fast_max',fast_max)
            self.hive.put('/cone_catalog/globals/depth_max',depth_max)
        return slow_max,fast_max,depth_max
                    

    def analyze_cone_phase(self,do_plot=True):
        av = self.crop_average_to_reference()

        clim = np.percentile(av,(1,99.5))
        
        if do_plot:
            jet = cm = plt.get_cmap('jet') 
            cNorm  = colors.Normalize(vmin=-1.0, vmax=1.0)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
            sy,sx = av.shape
            ar = float(sx)/float(sy)
            dpi = 100.0
            outx = 800.0/dpi
            outy = outx/ar
            plt.figure(2)
            plt.figure(1,figsize=(outx,outy))
            plt.axes([0,0,1,1])
            plt.imshow(av,clim=clim,cmap='gray',interpolation='none')
            plt.autoscale(False)
            outfn = '%s_unmarked_average.png'%self.tag
            plt.savefig(outfn)
            
        volume_dictionary = self.get_volume_dictionary()
        
        n_volumes = self.get_n_volumes()
        n_cones = self.get_n_cones()

        cone_catalog = self.hive['cone_catalog']
        cone_keys = cone_catalog.keys()

        
        for cone_index,ck in enumerate(cone_keys):
            stim_d_phase = []
            no_stim_d_phase = []
            if ck in ['slow_max','fast_max','depth_max','n_cones']:
                continue
            frame_keys = cone_catalog['%s'%ck].keys()

            coords = ck.split('_')
            x0 = int(coords[0])
            y0 = int(coords[1])
            
            for fk in frame_keys:
                if fk.find('no_stimulus')>-1:
                    stimulus = 0
                elif fk.find('stimulus')>-1:
                    stimulus = 1
                else:
                    stimulus = 0
                    
                index_keys = cone_catalog['%s/%s'%(ck,fk)].keys()
                for ik in index_keys:
                    vol = cone_catalog['%s/%s/%s/cone_volume'%(ck,fk,ik)][:,:,:]
                    
                    isos_depth = cone_catalog['%s/%s/%s/isos_z'%(ck,fk,ik)]
                    cost_depth = cone_catalog['%s/%s/%s/cost_z'%(ck,fk,ik)]

                    x = cone_catalog['%s/%s/%s/x'%(ck,fk,ik)]
                    y = cone_catalog['%s/%s/%s/y'%(ck,fk,ik)]
                    cone_origin = '%s_%s'%(fk,ik)
                    xy0 = ck.split('_')
                    x0 = int(xy0[0])
                    y0 = int(xy0[1])

                    c = Cone(vol,isos_depth,cost_depth,x,y,cone_origin,x0,y0)
                    
                    isos = vol[:,:,isos_depth].ravel()
                    cost = vol[:,:,cost_depth].ravel()
                    
                    
                    isos_phase = np.angle(isos)
                    cost_phase = np.angle(cost)
                    isos_amp = np.abs(isos)
                    cost_amp = np.abs(cost)
                    isos_real = np.real(isos)
                    cost_real = np.real(cost)
                    isos_imag = np.imag(isos)
                    cost_imag = np.imag(cost)

                    
                    amp = (isos_amp+cost_amp)/2.0

                    valid = np.where(amp>amp.mean()-0*amp.std())[0]
                    d_phase = (cost_phase-isos_phase)
                    d_phase = utils.unwrap(d_phase)
                    d_phase_median = np.median(d_phase[valid])

                    
                    if stimulus:
                        stim_d_phase.append(d_phase_median)
                    else:
                        no_stim_d_phase.append(d_phase_median)

                    
                    #print x0,y0,x,y,fk,ik,stimulus,d_phase_median
            pv = np.nanvar(stim_d_phase)/np.nanvar(no_stim_d_phase)
            print x0,y0,np.var(no_stim_d_phase),np.nanvar(stim_d_phase),pv
            if do_plot and pv>0.01:
                plt.figure(1)
                lu = min(1.0,-np.log(np.var(stim_d_phase)))
                colorVal = scalarMap.to_rgba(lu)
                plt.plot(x0*3,y0*3,marker='o',color=colorVal,alpha=0.5,markersize=10)
                
        if do_plot:
            outfn = '%s_marked_average.png'%self.tag
            plt.figure(1)
            plt.savefig(outfn)
            plt.show()
            
    def fit_cones(self):
        
        cone_catalog = self.hive['cone_catalog']
        cone_keys = cone_catalog.keys()
        
        for cone_index,ck in enumerate(cone_keys):
            if ck in ['slow_max','fast_max','depth_max','n_cones']:
                continue
            frame_keys = cone_catalog['%s'%ck].keys()

            coords = ck.split('_')
            x = int(coords[0])
            y = int(coords[1])
            
            for fk in frame_keys:
                    
                index_keys = cone_catalog['%s/%s'%(ck,fk)].keys()
                for ik in index_keys:
                    vol = cone_catalog['%s/%s/%s/cone_volume'%(ck,fk,ik)][:,:,:]
                    
                    isos_depth = cone_catalog['%s/%s/%s/isos_z'%(ck,fk,ik)]
                    cost_depth = cone_catalog['%s/%s/%s/cost_z'%(ck,fk,ik)]

                    x0 = cone_catalog['%s/%s/%s/x'%(ck,fk,ik)]
                    y0 = cone_catalog['%s/%s/%s/y'%(ck,fk,ik)]
                    cone_origin = '%s_%s'%(fk,ik)

                    c = Cone(vol,isos_depth,cost_depth,x,y,cone_origin,x0,y0)

                    
                    isos = vol[:,:,isos_depth].ravel()
                    cost = vol[:,:,cost_depth].ravel()
                    
                    
                    isos_phase = np.angle(isos)
                    cost_phase = np.angle(cost)
                    isos_amp = np.abs(isos)
                    cost_amp = np.abs(cost)
                    isos_real = np.real(isos)
                    cost_real = np.real(cost)
                    isos_imag = np.imag(isos)
                    cost_imag = np.imag(cost)

                    
                    amp = (isos_amp+cost_amp)/2.0

                    valid = np.where(amp>amp.mean()-0*amp.std())[0]
                    d_phase = (cost_phase-isos_phase)
                    d_phase = utils.unwrap(d_phase)
                    d_phase_median = np.median(d_phase[valid])

                    
                    if stimulus:
                        stim_d_phase.append(d_phase_median)
                    else:
                        no_stim_d_phase.append(d_phase_median)

                    
                    #print x0,y0,x,y,fk,ik,stimulus,d_phase_median
            pv = np.nanvar(stim_d_phase)/np.nanvar(no_stim_d_phase)
            print x0,y0,np.var(no_stim_d_phase),np.nanvar(stim_d_phase),pv
            if do_plot and pv>0.01:
                plt.figure(1)
                lu = min(1.0,-np.log(np.var(stim_d_phase)))
                colorVal = scalarMap.to_rgba(lu)
                plt.plot(x0*3,y0*3,marker='o',color=colorVal,alpha=0.5,markersize=10)
                
        if do_plot:
            outfn = '%s_marked_average.png'%self.tag
            plt.figure(1)
            plt.savefig(outfn)
            plt.show()
            
    def crop_average_to_reference(self):
        av = self.hive['/average_image/ISOS_COST'][:,:]
        ref = self.hive['/reference_frame'][:,:]
        x1 = int(self.hive['/reference_coordinates/x1']*3)
        x2 = int(self.hive['/reference_coordinates/x2']*3)
        y1 = int(self.hive['/reference_coordinates/y1']*3)
        y2 = int(self.hive['/reference_coordinates/y2']*3)
        av = av[2:,x1-90:x1+510]
        return av


    def make_key(self,filename,volume_idx,layer_names):
        return '%s_%06d_%s'%(filename.replace(os.sep,'_'),volume_idx,'_'.join(layer_names))
        
    
    def map_into_reference_space(self,filename,vidx,layer_names,goodness_threshold=-np.inf,oversampling_factor=1.0,oversampling_method='nearest'):
        k = self.make_key(filename,vidx,layer_names)

        sign = -1

        target_hive = Hive(filename)
        target_data = target_hive['processed_data'][vidx,:,:,:]
        target_data_time = target_hive['data_time'][vidx,:,:]
        n_slow,n_depth,n_fast = target_data.shape

        goodnesses = self.hive['/frames/%s/goodnesses'%k][:]
        xshifts = oversampling_factor*sign*self.hive['/frames/%s/x_shifts'%k][:]
        yshifts = oversampling_factor*sign*self.hive['/frames/%s/y_shifts'%k][:]+np.arange(n_slow)

        ref_space_x = self.hive['reference_position/x'][:]
        ref_space_y = self.hive['reference_position/y'][:]
        
        out_vol = np.nan*np.ones((len(ref_space_y)+oversampling_factor,n_depth,len(ref_space_x)+oversampling_factor),dtype=np.complex64)
        max_goodness = np.ones((len(ref_space_y)+oversampling_factor,len(ref_space_x)+oversampling_factor))*goodness_threshold
        out_time = np.ones((len(ref_space_y)+oversampling_factor,len(ref_space_x)+oversampling_factor))*np.nan
        
        valid_idx = np.where(goodnesses>=goodness_threshold)[0]
        for idx in valid_idx:
            xs = xshifts[idx]
            ys = yshifts[idx]
            g = goodnesses[idx]
            xerr = np.abs(xs-ref_space_x)
            yerr = np.abs(ys-ref_space_y)
            put_x1 = np.argmin(xerr)
            put_y1 = np.argmin(yerr)
            
            
            in_data0 = target_data[idx,:,:]
            in_time0 = target_data_time[idx,:]

            
            for xo in range(oversampling_factor):
                for yo in range(oversampling_factor):
                    in_data = in_data0.copy()
                    in_time = in_time0.copy()
                    put_x = (np.arange(n_fast,dtype=np.integer)*oversampling_factor)+xo+put_x1
                    put_y = idx+yo+put_y1

                    while put_x.max()>=out_vol.shape[2]:
                        put_x=put_x[:-1]
                        in_data=in_data[:,:-1]
                        in_time=in_time[:-1]
                    while put_x.min()<0:
                        put_x=put_x[1:]
                        in_data=in_data[:,1:]
                        in_time=in_time[1:]
                    #print put_y,put_x,max_goodness.shape
                    if max_goodness[put_y,put_x].mean()<=g:
                        max_goodness[put_y,put_x]=g
                        out_vol[put_y,:,put_x] = in_data.T
                        out_time[put_y,put_x] = in_time

        return out_vol,out_time

    def add(self,filename,vidx,key=None,layer_names=None,overwrite=True,oversample_factor=3,strip_width=3.0,do_plot=False,use_gaussian=False,background_diameter=0):
        
        print 'Adding %s, volume %d.'%(filename,vidx)
        
        if key is None:
            self.key = self.make_key(filename,vidx,layer_names)
        else:
            self.key = key
        
        if self.hive.has('/frames/%s'%self.key) and not overwrite:
            print 'Series already has entry for %s.'%self.key
            return

        
        target,label = self.get_image(filename,vidx,layer_names)
        self.db.put(self.key,filename,vidx)

        reference = self.reference
        
        y,x,g = utils.strip_register(target,reference,oversample_factor,strip_width,do_plot=do_plot,use_gaussian=use_gaussian,background_diameter=background_diameter)

        self.hive.put('/frames/%s/x_shifts'%self.key,x)
        self.hive.put('/frames/%s/y_shifts'%self.key,y)
        self.hive.put('/frames/%s/goodnesses'%self.key,g)
        self.hive.put('/frames/%s/reference'%self.key,[0])
        self.hive.put('/frames/%s/oversample_factor'%self.key,oversample_factor)

    def get_image(self,filename,vidx,layer_names):
        
        target_hive = Hive(filename)
        if layer_names is None:
            # if the layer_names list is missing, use the first layer as a default
            # this seems like okay behavior since most times there's only one projection
            # anyway
            layer_names = [target_h5['projections'].keys()[0]]

        label = '_'.join(layer_names)

        if len(layer_names)>1:
            print target_hive['projections'][layer_names[0]][vidx,:,:]
            
            test = target_hive['projections'][layer_names[0]][vidx,:,:]
            n_slow,n_fast = test.shape
            stack = np.zeros((len(layer_names),n_slow,n_fast))
            for idx,layer_name in enumerate(layer_names):
                stack[idx,:,:] = target_hive['projections'][layer_name][vidx,:,:]
            out = np.mean(stack,axis=0)
            del stack
        else:
            out = target_hive['projections'][layer_names[0]][vidx,:,:]    
        return out,label

    def get_image0(self,filename_stub,vidx,layer_names):
        filename = os.path.join(self.working_directory,filename_stub)
        target_h5 = H5(filename)

        if layer_names is None:
            # if the layer_names list is missing, use the first layer as a default
            # this seems like okay behavior since most times there's only one projection
            # anyway
            layer_names = [target_h5['projections'].keys()[0]]

        label = '_'.join(layer_names)

        if len(layer_names)>1:
            test = target_h5['projections'][layer_names[0]][vidx,:,:]
            n_slow,n_fast = test.shape
            stack = np.zeros((len(layer_names),n_slow,n_fast))
            for idx,layer_name in enumerate(layer_names):
                stack[idx,:,:] = target_h5['projections'][layer_name][vidx,:,:]
            out = np.mean(stack,axis=0)
            del stack
        else:
            out = target_h5['projections'][layer_names[0]][vidx,:,:]    
        return out,label
    
    def is_registered(self):
        counter = 0
        try:
            my_frames = self.hive['/frames'].keys()
            return len(my_frames)
            for mf in my_frames:
                for fileindex in self.hive['/frames'][mf].keys():
                    counter = counter + 1
        except Exception as e:
            print 'foo',e
            sys.exit()
        return counter
                
        
    def get_volume(self,filename_stub,vidx,data_block):
        filename = os.path.join(self.working_directory,filename_stub)
        target_h5 = H5(filename)
        return target_h5[data_block][vidx,:,:,:]

    def get_z_offsets(self,filename_stub,vidx,layer_name=None):
        filename = os.path.join(self.working_directory,filename_stub)
        target_h5 = H5(filename)
        if layer_name is None:
            out = target_h5['/model/z_offsets'][vidx,:,:]
        else:
            out = target_h5['/model/volume_labels/%s'%layer_name][vidx,:,:]
        return out
    
    def get_n_frames(self):
        count = 0
        try:
            filenames = self.hive['/frames'].keys()
            for filename in filenames:
                count = count + len(self.hive['/frames/%s'%filename].keys())
        except:
            pass
        return count

    def is_stacked(self):
        try:
            junk = self.hive['stack_counter']
            return True
        except:
            return False

    def is_rendered(self):
        return len(self.hive['/sum_image'].keys())
            
    def is_volume_rendered(self):
        out = True
        try:
            test = self.hive['/sum_volume']
        except:
            out = False
        return out

            
    def is_corrected_a(self):
        out = True
        try:
            test = self.hive['/corrected_a']
        except:
            out = False

        return out

    
    def is_corrected_b(self):
        out = True
        try:
            test = self.hive['/corrected_b']
        except:
            out = False

        return out

    def goodness_histogram(self):
        all_goodnesses = []
        keys = self.db.dictionary.keys()
        for k in keys:
            gfn = os.path.join(os.path.join(os.path.join(self.series_directory,'frames'),k),'goodnesses.npy')
            goodnesses = list(np.load(gfn))
            all_goodnesses = all_goodnesses + goodnesses

        plt.hist(all_goodnesses,500)
        plt.show()
    
    def render(self,layer_names=None,goodness_threshold=0.0,correlation_threshold=-1.0,overwrite=False,oversample_factor=3,do_plot=False,left_crop=0,right_crop=0):

        keys = self.hive['frames'].keys()
        keys.sort()
        
        sign = -1
        # remember the convention here: x and y shifts are the
        # amount of shift required to align the line in question
        # with the reference image
        # first, find the minimum and maximum x and y shifts,
        # in order to know how big the canvas must be for fitting
        # all of the lines
        xmin = np.inf
        xmax = -np.inf
        ymin = np.inf
        ymax = -np.inf

        reg_dict = {}
        
        for k_idx,k in enumerate(keys):
            filename,vidx = self.db.get(k)
            test,label = self.get_image(filename,0,layer_names)

            n_slow,n_fast = test.shape
            goodnesses = self.hive['/frames/%s/goodnesses'%k][:]
            xshifts = sign*self.hive['/frames/%s/x_shifts'%k][:]
            yshifts = sign*self.hive['/frames/%s/y_shifts'%k][:]

            xshifts = np.squeeze(xshifts)
            yshifts = np.squeeze(yshifts)

            xshifts,yshifts,goodnesses,valid = self.filter_registration(xshifts,yshifts,goodnesses)

            use_for_limits = np.where(goodnesses>=goodness_threshold)[0]
            yshifts = yshifts + valid

            #try:
            #    print k,np.min(xshifts[use_for_limits]),np.max(xshifts[use_for_limits])
            #except Exception as e:
            #    print k,e
                
            try:
                newxmin = np.min(xshifts[use_for_limits])
                newxmax = np.max(xshifts[use_for_limits])
                newymin = np.min(yshifts[use_for_limits])
                newymax = np.max(yshifts[use_for_limits])

                xmin = min(xmin,newxmin)
                xmax = max(xmax,newxmax)
                ymin = min(ymin,newymin)
                ymax = max(ymax,newymax)
                if False:
                    print xmin,xmax,ymin,ymax
                    plt.plot(k_idx,xmin,'rs')
                    plt.plot(k_idx,xmax,'ro')
                    plt.plot(k_idx,ymin,'gs')
                    plt.plot(k_idx,ymax,'go')
                    plt.pause(.0001)
            except Exception as e:
                print e

            reg_dict[k] = (xshifts,yshifts,goodnesses,valid)

        canvas_width = xmax-xmin+n_fast
        canvas_height = ymax-ymin+10

        ref_x1 = 0 - xmin
        ref_y1 = 0 - ymin
        ref_x2 = ref_x1 + n_fast
        ref_y2 = ref_y1 + n_slow

        self.hive.put('/reference_coordinates/x1',ref_x1)
        self.hive.put('/reference_coordinates/x2',ref_x2)
        self.hive.put('/reference_coordinates/y1',ref_y1)
        self.hive.put('/reference_coordinates/y2',ref_y2)

        for key in reg_dict.keys():
            xs,ys,g,v = reg_dict[key]
            xs = xs - xmin
            ys = ys - ymin
            reg_dict[key] = (xs,ys,g,v)
                

        canvas_width = int((canvas_width+1)*oversample_factor)
        canvas_height = int((canvas_height+1)*oversample_factor)

        rmean = np.mean(self.reference)

        embedded_reference = np.ones((canvas_height,canvas_width))*rmean
        ref_x1 = int(ref_x1*oversample_factor)
        ref_x2 = int(ref_x2*oversample_factor)
        ref_y1 = int(ref_y1*oversample_factor)
        ref_y2 = int(ref_y2*oversample_factor)

        reference = self.reference
        ref_oversampled = zoom(reference,oversample_factor)
        
        embedded_reference[ref_y1:ref_y2,ref_x1:ref_x2] = ref_oversampled

        self.hive.put('/reference_position/y',np.arange(canvas_height)-ref_y1)
        self.hive.put('/reference_position/x',np.arange(canvas_width)-ref_x1)
        

        sum_image = np.zeros((canvas_height,canvas_width))
        counter_image = np.zeros((canvas_height,canvas_width))
        correlation_image = np.zeros((canvas_height,canvas_width))


        
        if do_plot:
            mov = GIF('temp.gif',fps=5)
            dpi = 100.0
            fig = plt.figure(figsize=(3*canvas_width/dpi,canvas_height/dpi))
            ax1 = fig.add_axes([0,0,.3333,1])
            ax2 = fig.add_axes([.3333,0,.3333,1])
            ax3 = fig.add_axes([.6667,0,.3333,1])
        
        for k in sorted(reg_dict.keys()):
            #if not k=='_home_rjonnal_data_Dropbox_Share_fdml_faooct_01_DataSet1_0100_000000_CONES':
            #    continue
            xshifts,yshifts,goodnesses,indices = reg_dict[k]
            temp = self.db.get(k)
            filename = temp[0]
            frame_index = int(temp[1])
            im,label = self.get_image(filename,frame_index,layer_names)
            correlation_vector = np.zeros((n_slow))
            
            this_image = np.zeros((canvas_height,canvas_width))
            
            for idx,xs,ys,g in zip(indices,xshifts,yshifts,goodnesses):
                if g<goodness_threshold:
                    continue
                line = im[idx,:]
                line = np.expand_dims(line,0)
                block = zoom(line,oversample_factor)
                bsy,bsx = block.shape
                x1 = int(np.round(xs*oversample_factor))
                x2 = x1 + bsx
                y1 = int(np.round(ys*oversample_factor))
                y2 = y1 + bsy

                #ref_section = embedded_reference[y1:y2,x1:x2]

                #while block.shape[1]>ref_section.shape[1]:
                #    block = block[:,:-1]

                #ref_section = ref_section.ravel()
                
                #corr = np.corrcoef(ref_section,block.ravel())[1,0]
                #correlation_image[y1:y2,x1:x2] = correlation_image[y1:y2,x1:x2] + corr
                #correlation_vector[idx] = corr
                if True:#corr>correlation_threshold:
                    #print 'sum_image shape',sum_image.shape
                    #print 'x1,x2',x1,x2
                    #print 'block shape',block.shape

                    this_image[y1:y2,x1:x2] = this_image[y1:y2,x1:x2] + block
                    sum_image[y1:y2,x1:x2] = sum_image[y1:y2,x1:x2] + block
                    counter_image[y1:y2,x1:x2] = counter_image[y1:y2,x1:x2] + 1.0
                    
            #self.hive.put('/frames/%s/correlations'%k,correlation_vector)
            if do_plot:

                try:
                    clim = np.percentile(this_image.ravel()[np.where(this_image.ravel())],(1,99))
                except Exception as e:
                    clim = None

                #fig.clear()

                ax1.clear()
                ax1.imshow(this_image,cmap='gray',interpolation='none',clim=clim)
                
                temp = counter_image.copy()
                temp[np.where(temp==0)] = 1.0
                av = sum_image/temp

                try:
                    clim = np.percentile(av.ravel()[np.where(av.ravel())],(1,99))
                except Exception as e:
                    clim = None
                ax2.clear()
                ax2.imshow(av,cmap='gray',interpolation='none',clim=clim)

                ax3.clear()
                ax3.imshow(counter_image)
                #plt.colorbar()
                mov.add(fig)
                plt.pause(.001)

        mov.make()
        temp = counter_image.copy()
        temp[np.where(temp==0)] = 1.0
        av = sum_image/temp

        #cropper = self.get_cropper(counter_image)
        cropper = lambda x: x
        self.hive.put('/correlation_image/%s'%label,cropper(correlation_image))
        self.hive.put('/counter_image/%s'%label,cropper(counter_image))
        self.hive.put('/average_image/%s'%label,cropper(av))
        self.hive.put('/sum_image/%s'%label,cropper(sum_image))
        
        if do_plot:
            plt.close()

            plt.subplot(1,2,1)
            plt.imshow(av,cmap='gray',interpolation='none')

            plt.subplot(1,2,2)
            plt.imshow(counter_image)
            plt.colorbar()

            plt.show()

    def get_cropper(self,im):
        # return a cropper function that
        # will crop images according to im's
        # empty (uninformative) borders
        vprof = np.std(im,axis=1)
        valid_region_y = np.where(vprof)[0]
        hprof = np.std(im,axis=0)
        valid_region_x = np.where(hprof)[0]
        y1 = valid_region_y[0]
        y2 = valid_region_y[-1]
        x1 = valid_region_x[0]
        x2 = valid_region_x[-1]
        return lambda x: x[y1:y2,x1:x2]
    
    def render_stack(self,layer_names=None,goodness_threshold=0.0,correlation_threshold=-1.0,overwrite=False,oversample_factor=3,do_plot=False,left_crop=0,right_crop=0):

        files = self.hive['frames'].keys()

        sign = -1
        # remember the convention here: x and y shifts are the
        # amount of shift required to align the line in question
        # with the reference image
        # first, find the minimum and maximum x and y shifts,
        # in order to know how big the canvas must be for fitting
        # all of the lines
        xmin = np.inf
        xmax = -np.inf
        ymin = np.inf
        ymax = -np.inf

        reg_dict = {}

        n_frames = 0
        for filename in files:
            keys = self.hive['/frames/%s'%filename].keys()
            keys.sort()
            for k in keys:
                print k
                test,label = self.get_image(filename,0,layer_names)
                n_slow,n_fast = test.shape

                goodnesses = self.hive['/frames/%s/%s/goodnesses'%k][:]
                xshifts = sign*self.hive['/frames/%s/%s/x_shifts'%k][:]
                yshifts = sign*self.hive['/frames/%s/%s/y_shifts'%k][:]

                xshifts = np.squeeze(xshifts)
                yshifts = np.squeeze(yshifts)
                
                xshifts,yshifts,goodnesses,valid = self.filter_registration(xshifts,yshifts,goodnesses)

                use_for_limits = np.where(goodnesses>=goodness_threshold)

                try:
                    newxmin = np.min(xshifts[use_for_limits])
                    newxmax = np.max(xshifts[use_for_limits])
                    newymin = np.min(yshifts[use_for_limits])
                    newymax = np.max(yshifts[use_for_limits])
                    
                    xmin = min(xmin,newxmin)
                    xmax = max(xmax,newxmax)
                    ymin = min(ymin,newymin)
                    ymax = max(ymax,newymax)
                except Exception as e:
                    print e

                yshifts = yshifts + valid

                reg_dict[k] = (xshifts,yshifts,goodnesses,valid)
                n_frames = n_frames + 1

        canvas_width = xmax-xmin+n_fast
        canvas_height = ymax-ymin+n_slow
        canvas_depth = n_frames

        ref_x1 = 0 - xmin
        ref_y1 = 0 - ymin
        ref_x2 = ref_x1 + n_fast
        ref_y2 = ref_y1 + n_slow
        
        self.hive.put('/reference_coordinates/x1',ref_x1)
        self.hive.put('/reference_coordinates/x2',ref_x2)
        self.hive.put('/reference_coordinates/y1',ref_y1)
        self.hive.put('/reference_coordinates/y2',ref_y2)
        
        for key in reg_dict.keys():
            xs,ys,g,v = reg_dict[key]
            xs = xs - xmin
            ys = ys - ymin
            reg_dict[key] = (xs,ys,g,v)
                

        canvas_width = int(canvas_width*oversample_factor)
        canvas_height = int((canvas_height+1)*oversample_factor)

        rmean = np.mean(self.reference)

        embedded_reference = np.ones((canvas_height,canvas_width))*rmean
        ref_x1 = int(ref_x1*oversample_factor)
        ref_x2 = int(ref_x2*oversample_factor)
        ref_y1 = int(ref_y1*oversample_factor)
        ref_y2 = int(ref_y2*oversample_factor)
        ref_oversampled = zoom(self.reference,oversample_factor)
        embedded_reference[ref_y1:ref_y2,ref_x1:ref_x2] = ref_oversampled

        rkeys = reg_dict.keys()
        rkeys.sort()
        for depth_idx in range(canvas_depth):
            
            frame = np.zeros((canvas_height,canvas_width))
            counter = np.zeros((canvas_height,canvas_width))

            k = rkeys[depth_idx]
            xshifts,yshifts,goodnesses,indices = reg_dict[k]
            filename = k[0]
            frame_index = int(k[1])
            im,label = self.get_image(filename,frame_index,layer_names)
            correlation_vector = np.zeros((n_slow))
            
            for idx,xs,ys,g in zip(indices,xshifts,yshifts,goodnesses):
                if g<goodness_threshold:
                    continue
                xs = xs + left_crop
                line = im[idx,left_crop:-right_crop]
                line = np.expand_dims(line,0)
                block = zoom(line,oversample_factor)
                bsy,bsx = block.shape
                x1 = int(np.round(xs*oversample_factor))
                x2 = x1 + bsx
                y1 = int(np.round(ys*oversample_factor))
                y2 = y1 + bsy

                ref_section = embedded_reference[y1:y2,x1:x2]

                while block.shape[1]>ref_section.shape[1]:
                    block = block[:,:-1]

                ref_section = ref_section.ravel()
                
                corr = np.corrcoef(ref_section,block.ravel())[1,0]
                correlation_vector[idx] = corr
                if corr>correlation_threshold:
                    frame[y1:y2,x1:x2] = frame[y1:y2,x1:x2] + block
                    counter[y1:y2,x1:x2] = counter[y1:y2,x1:x2] + 1.0

            counter[np.where(counter==0)] = 1.0
            frame = frame/counter
            if do_plot:
                plt.cla()
                plt.imshow(frame,cmap='gray')
                plt.pause(.0001)
            self.hive.put('/frames/%s/%s/correlations'%(filename,k[1]),correlation_vector)
            self.hive.put('/stack/%s/%06d'%(label,depth_idx),frame)


    def crop_stacks(self,label,dry_run=True,do_plot=False):
        """Using the stack labeled LABEL, crop all substacks
        equivalently."""

        try:
            counter = self.hive.get('/stack_counter')
            cy,cx = counter.shape
        except Exception as e:
            counter_exists = False
            keys = self.hive['/stack/%s'%label].keys()
            for idx,k in enumerate(keys):
                print '%d of %d'%(idx+1,len(keys))
                temp = self.hive.get('/stack/%s/%s'%(label,k))
                if not counter_exists:
                    counter = np.zeros(temp.shape)
                    counter_exists = True
                counter[np.where(temp)]+=1.0
            self.hive.put('/stack_counter',counter)

        xc,yc,junk = collector([counter],titles=['Please click upper-left and lower-right corners of ROI.'])

        # if the collector comes back with not enough points,
        # default to the full image dimensions
        if len(xc<2):
            xc = [0,counter.shape[1]]
            yc = [0,counter.shape[0]]
            
        x1 = int(np.min(xc))
        x2 = int(np.max(xc))
        y1 = int(np.min(yc))
        y2 = int(np.max(yc))
        cropped_sy = y2-y1
        cropped_sx = x2-x1
        counter = counter[y1:y2,x1:x2]

        if not dry_run:
            plt.imshow(counter,cmap='gray')
            plt.colorbar()
            plt.show()
            print 'Warning: cropping cannot be undone. Be sure this is right before proceeding.'
            ans = raw_input('Continue? [y/N] ')
            if not len(ans):
                return
            if not ans.lower()=='y':
                return

        if not dry_run:
            self.hive.put('/stack_counter',counter)
            
        substack_keys = self.hive['/stack/'].keys()
        for k in substack_keys:
            file_keys = self.hive['/stack/%s'%k].keys()
            for idx,fk in enumerate(file_keys):
                loc = '/stack/%s/%s'%(k,fk)
                uncropped = self.hive.get(loc)
                uy,ux = uncropped.shape
                if y2>uy or x2>ux:
                    continue
                cropped = uncropped[y1:y2,x1:x2]
                
                if do_plot or dry_run:
                    plt.subplot(1,2,1)
                    plt.cla()
                    plt.imshow(uncropped,cmap='gray')
                    plt.subplot(1,2,2)
                    plt.cla()
                    plt.imshow(cropped,cmap='gray')
                    if dry_run:
                        plt.title('Preview of crop (dry run).')
                        plt.pause(.00000001)
                        continue
                    else:
                        plt.title('Cropping.')
                        plt.pause(.00000001)
                self.hive.put(loc,cropped)
                print '%s: %d of %d'%(k,idx,len(file_keys))
                
    def average_stack(self,label,redo=False,do_plot=False):
        # open one image from the stack for testing and setup:
        # use it to see if the images retrieved from the hive
        # are right (i.e. that they're not leftover from an
        # old registration), or to make blank arrays for
        # recalculation.
        frame_keys = self.hive['/stack/%s'%label].keys()
        test = self.hive.get('/stack/%s/%s'%(label,frame_keys[0]))
        test_sy,test_sx = test.shape

        if not redo:
            try:
                counter_image=self.hive.get('/counter_image/%s'%label)
                av = self.hive.get('/average_image/%s'%label)
                sum_image = self.hive.get('/sum_image/%s'%label)
                stored_sy,stored_sx = counter_image.shape
                assert (test_sy==stored_sy and test_sx==stored_sx)
                return av
            except:
                pass
        
        counter_image = np.zeros((test_sy,test_sx))
        sum_image = np.zeros((test_sy,test_sx))
        for fk in frame_keys:
            print fk
            im = self.hive.get('/stack/%s/%s'%(label,fk))
            counter_image[np.where(im)] = counter_image[np.where(im)] + 1.0
            sum_image = sum_image + im
            if do_plot:
                fixed_counter = counter_image.copy()
                fixed_counter[np.where(counter_image==0)]=1.0
                av = sum_image/fixed_counter
                avmean = av[np.where(av)].mean()
                av[np.where(av==0)] = avmean
                if do_plot:
                    plt.subplot(1,2,1)
                    plt.cla()
                    plt.imshow(av,cmap='gray')
                    plt.subplot(1,2,2)
                    plt.cla()
                    plt.imshow(counter_image,cmap='gray')
                    plt.pause(.00001)

        fixed_counter = counter_image.copy()
        fixed_counter[np.where(counter_image==0)]=1.0
        av = sum_image/fixed_counter
            
        self.hive.put('/counter_image/%s'%label,counter_image)
        self.hive.put('/average_image/%s'%label,av)
        self.hive.put('/sum_image/%s'%label,sum_image)
        
            
    def render_volume(self,layer_names=None,goodness_threshold=0.0,correlation_threshold=-1.0,overwrite=False,oversample_factor=3,do_plot=False,left_crop=0,data_block='/flattened_data',offset_medfilt_kernel=9,layer_name='ISOS',align_bscan=False):

        files = self.hive['frames'].keys()
        print files
        
        sign = -1
        # remember the convention here: x and y shifts are the
        # amount of shift required to align the line in question
        # with the reference image
        # first, find the minimum and maximum x and y shifts,
        # in order to know how big the canvas must be for fitting
        # all of the lines
        xmin = np.inf
        xmax = -np.inf
        ymin = np.inf
        ymax = -np.inf
        zmin = np.inf
        zmax = -np.inf
        max_depth = -np.inf

        reg_dict = {}

        def fix_corners(im):
            fill_value = np.median(im)
            to_check = np.zeros(im.shape)
            rad = (offset_medfilt_kernel-1)//2
            to_check[:rad,:rad] = 1
            to_check[-rad:,:rad] = 1
            to_check[:rad,-rad:] = 1
            to_check[-rad:,-rad:] = 1
            im[np.where(np.logical_and(to_check,1-im))] = fill_value
            return im
        
        for filename in files:
            keys = self.hive['/frames/%s'%filename].keys()
            for k in keys:
                test = self.get_volume(filename,int(k),data_block)
                test = np.abs(test[:,:,left_crop:])
                orig_vol_shape = test.shape
                zshifts = self.get_z_offsets(filename,int(k),layer_name)[:,left_crop:]
                zshifts = medfilt(zshifts,offset_medfilt_kernel)
                zshifts = fix_corners(zshifts)
                
                n_slow,n_depth,n_fast = test.shape
                if n_depth>max_depth:
                    max_depth = n_depth
                if False:
                    for k in range(n_depth):
                        im = test[:,k,:]
                        plt.cla()
                        plt.imshow(im,cmap='gray',interpolation='none',clim=np.percentile(test,(5,99.95)))
                        plt.pause(.1)
                    sys.exit()
                
                goodnesses = self.hive['/frames/%s/%s/goodnesses'%(filename,k)][:]
                xshifts = sign*self.hive['/frames/%s/%s/x_shifts'%(filename,k)][:]
                yshifts = sign*self.hive['/frames/%s/%s/y_shifts'%(filename,k)][:]

                xshifts = np.squeeze(xshifts)
                yshifts = np.squeeze(yshifts)

                xshifts,yshifts,goodnesses,valid = self.filter_registration(xshifts,yshifts,goodnesses)

                zshifts = zshifts[valid,:]
                
                newxmin = np.min(xshifts)
                newxmax = np.max(xshifts)
                newymin = np.min(yshifts)
                newymax = np.max(yshifts)
                newzmin = np.min(zshifts)
                newzmax = np.max(zshifts)

                
                xmin = min(xmin,newxmin)
                xmax = max(xmax,newxmax)
                ymin = min(ymin,newymin)
                ymax = max(ymax,newymax)
                zmin = min(zmin,newzmin)
                zmax = max(zmax,newzmax)
                yshifts = yshifts + valid

                reg_dict[(filename,k)] = (xshifts,yshifts,zshifts,goodnesses,valid)

        
        canvas_width = int(xmax-xmin+n_fast)
        canvas_height = int(ymax-ymin+n_slow)
        canvas_depth = int(zmax-zmin+max_depth+1)
        print 'canvas_depth start',canvas_depth
        
        ref_x1 = 0 - xmin
        ref_y1 = 0 - ymin
        ref_x2 = ref_x1 + n_fast
        ref_y2 = ref_y1 + n_slow
        
        self.hive.put('/reference_coordinates/x1',ref_x1)
        self.hive.put('/reference_coordinates/x2',ref_x2)
        self.hive.put('/reference_coordinates/y1',ref_y1)
        self.hive.put('/reference_coordinates/y2',ref_y2)
        
        for key in reg_dict.keys():
            xs,ys,zs,g,v = reg_dict[key]
            xs = xs - xmin
            ys = ys - ymin
            zs = zs - zmin
            reg_dict[key] = (xs,ys,zs,g,v)

        if False:
            for key in reg_dict.keys():
                plt.clf()
                plt.imshow(reg_dict[key][2])
                plt.colorbar()
                plt.pause(1)
            sys.exit()
            
        canvas_width = canvas_width*oversample_factor
        canvas_height = (canvas_height+1)*oversample_factor
        canvas_depth0 = canvas_depth
        canvas_depth = canvas_depth*oversample_factor
        print 'canvas_depth oversampled',canvas_depth
        sum_image = np.zeros((canvas_height,canvas_depth,canvas_width))
        counter_image = np.ones((canvas_height,canvas_depth,canvas_width))*1e-10

        errorcount = 0
        for volume_count,k in enumerate(reg_dict.keys()):
            xshifts,yshifts,zshifts,goodnesses,indices = reg_dict[k]
            filename = k[0]
            frame_index = int(k[1])
            vol = np.abs(self.get_volume(filename,frame_index,data_block))[:,:,left_crop:]
            
            for bscan_count,(idx,xs,ys,zs,g) in enumerate(zip(indices,xshifts,yshifts,zshifts,goodnesses)):
                bscan = vol[idx,:,:]
                n_depth,n_fast = bscan.shape
                shifted_bscan = np.zeros((canvas_depth0,n_fast))
                print 'bscan shape',bscan.shape
                print 'shifted_bscan shape',shifted_bscan.shape
                print 'zs lims',zs.min(),zs.max()
                print 'zmax',zmax
                if align_bscan:
                    for i_fast in range(n_fast):
                        z1 = zmax-zs[i_fast]
                        z2 = z1 + n_depth
                        shifted_bscan[z1:z2,i_fast] = bscan[:,i_fast]
                else:
                    z1 = zmax - int(round(np.mean(zs)))
                    z2 = z1 + n_depth
                    cut_count = 0
                    while z2>shifted_bscan.shape[0]:
                        bscan = bscan[:-1,:]
                        z2 = z2 - 1
                        cut_count =+ 1
                    if cut_count:
                        print 'cut %d lines'%cut_count    
                    shifted_bscan[int(z1):int(z2),:] = bscan
                print

                if False:
                    plt.subplot(3,1,1)
                    plt.imshow(bscan,interpolation='none',cmap='gray',aspect='normal')
                    plt.subplot(3,1,2)
                    plt.imshow(shifted_bscan,interpolation='none',cmap='gray',aspect='normal')
                    plt.subplot(3,1,3)
                    plt.plot(zs)
                    plt.figure()
                    plt.plot(np.mean(bscan,axis=1))
                    plt.plot(np.mean(shifted_bscan,axis=1))
                    plt.show()
                    continue
                
                block = np.zeros((oversample_factor,canvas_depth,n_fast*oversample_factor))
                shifted_bscan = zoom(shifted_bscan,oversample_factor)
                
                for ofk in range(oversample_factor):
                    block[ofk,:,:] = shifted_bscan
                bsy,bsz,bsx = block.shape
                
                x1 = int(np.round(xs*oversample_factor))
                x2 = x1 + bsx
                y1 = int(np.round(ys*oversample_factor))
                y2 = y1 + bsy

                try:
                    sum_image[y1:y2,:,x1:x2] = sum_image[y1:y2,:,x1:x2] + block
                    counter_image[y1:y2,:,x1:x2] = counter_image[y1:y2,:,x1:x2] + 1.0
                except Exception as e:
                    errorcount = errorcount + 1
                    print e
                print volume_count,bscan_count
            if do_plot:
                vav = sum_image[:,500:600,200]/counter_image[:,500:600,200]
                hav = sum_image[200,500:600,:]/counter_image[200,500:600,:]
                plt.clf()
                plt.subplot(1,2,1)
                plt.cla()
                plt.imshow(vav.T,cmap='gray',interpolation='none',aspect='normal')
                plt.colorbar()
                plt.title(volume_count)
                plt.subplot(1,2,2)
                plt.cla()
                plt.imshow(hav,cmap='gray',interpolation='none',aspect='normal')
                plt.colorbar()
                plt.pause(.0000000001)

            
        av = sum_image/counter_image
        label = layer_name
        self.hive.put('/counter_volume/%s'%label,counter_image)
        self.hive.put('/average_volume/%s'%label,av)
        self.hive.put('/sum_volume/%s'%label,sum_image)
        print 'error count %d'%errorcount
        if do_plot:
            plt.close()
            plt.subplot(1,2,1)
            plt.cla()
            plt.imshow(vav.T,cmap='gray',interpolation='none',aspect='normal')
            plt.colorbar()
            plt.title(volume_count)
            plt.subplot(1,2,2)
            plt.cla()
            plt.imshow(hav,cmap='gray',interpolation='none',aspect='normal')
            plt.colorbar()
            plt.show()
        
            
    def filter_registration(self,xshifts,yshifts,goodnesses,xmax=25,ymax=25,medfilt_region=49,do_plot=False):
        xmed = medfilt(xshifts,medfilt_region)
        ymed = medfilt(yshifts,medfilt_region)
        xerr = np.abs(xshifts-xmed)
        yerr = np.abs(yshifts-ymed)
        xvalid = xerr<=xmax
        yvalid = yerr<=ymax

        valid = np.where(np.logical_and(xvalid,yvalid))[0]
        #print '%d points: '%len(valid),valid
        if do_plot:
            plt.figure()
            plt.subplot(1,2,1)
            plt.plot(xshifts,'k-')
            #plt.plot(xmed,'b-')
            plt.plot(yshifts,'k--')
            #plt.plot(ymed,'b--')

            plt.subplot(1,2,2)
            plt.plot(valid,xshifts[valid],'ks')
            plt.plot(valid,yshifts[valid],'ko')
            
            plt.show()

        return xshifts[valid],yshifts[valid],goodnesses[valid],valid
        
        


    def show_images(self):

        keys = self.hive.keys()
        for k in keys:
            plt.cla()
            im,label = self.get_image(full_filename,vidx)
            self.imshow(im)
            plt.title(k)
            plt.pause(.1)
        plt.close()
        

if __name__=='__main__':

    frames_fn = '/home/rjonnal/data/Dropbox/Share/2g_aooct_data/Data/2016.11.21_cones/14_13_13-1T_500.hdf5' # volume 7
    h5 = H5(frames_fn)
    h5.catalog()
    
    ref = (h5['projections/ISOS'][0,:,:]+h5['projections/ISOS'][0,:,:])/2.0

    s = Series(ref,frames_fn.replace('.hdf5','')+'_registered.hdf5')

    for k in range(12):
        s.add(frames_fn,k,['ISOS','COST'],do_plot=False,overwrite=False)

    s.render(['ISOS','COST'],do_plot=True)

    