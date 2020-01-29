import sys,os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QSpinBox, QDoubleSpinBox,QLabel,QFileDialog,QCheckBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSignal,QObject,pyqtSlot,Qt
# Make sure that we are using QT5
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import random
from PIL import Image
import oct_ui_config as ocfg
import scipy.interpolate as spi
import scipy.io as sio
import scipy.ndimage as spn
import glob
import tifffile

class ProcessingInfo:
    pass

def process(vol_in,L1,L2,c3,c2,dc_subtract=False):
    is_bscan = len(vol_in.shape)==2
    if is_bscan:
        vol = np.zeros((vol_in.shape[0],vol_in.shape[1],1))
        vol[:,:,0] = vol_in
    else:
        vol = vol_in
    
    # generate an array of k-values corresponding to the
    # linearly spaced lambda values of the acquired stack;
    # these k values are not linearly spaced
    k_in = 2*np.pi/np.linspace(L1,L2,vol.shape[0])
    # generate linearly spaced k values between the first and
    # last k values of k_in
    k_out = np.linspace(k_in[0],k_in[-1],len(k_in))

    if dc_subtract:
        dc = vol.mean(2).mean(1)
        vol = np.transpose(np.transpose(vol,[1,2,0])-dc,[2,0,1])
    else:
        dc = -1

    sk,sy,sx = vol.shape
    k_interpolator = spi.interp1d(k_in,vol,axis=0,copy=False)
    pvol = k_interpolator(k_out)

    dispersion_axis = k_out - np.mean(k_out)
    dispersion_coefficients = [c3,c2,0.0,0.0]
    oversample_factor = 1
    phase = np.exp(1j*np.polyval(dispersion_coefficients,dispersion_axis))
    pvol = (pvol.transpose([1,2,0])*phase).transpose([2,0,1])
    pvol = np.fft.fftshift(np.fft.fft(pvol,n=pvol.shape[0]*oversample_factor,axis=0),axes=0)
    if is_bscan:
        pvol = pvol[:,:,0]

    info = ProcessingInfo
    info.k_in = k_in
    info.k_out = k_out
    info.phase = phase
    info.dc = dc
    return pvol,info


def filter_volume(vol):
    x0 = ocfg.offaxis_x0
    x1 = ocfg.offaxis_x1
    y0 = ocfg.offaxis_y0
    y1 = ocfg.offaxis_y1
    sigma = ocfg.offaxis_sigma

    # 2D FFT each en face slice in the volume, multiply by a
    # gaussian centered about one of the off-axis shifted images,
    # and 2D FFT back

    sk,sy0,sx0 = vol.shape

    vol_square = np.zeros((sk,max(sx0,sy0),max(sx0,sy0)))
    vol_square[:,:sy0,:sx0] = vol

    sk,sy,sx = vol_square.shape

    filt = np.zeros(vol_square.shape)
    x_vec = np.linspace(x0,x1,sk)
    y_vec = np.linspace(y0,y1,sk)
    XX,YY = np.meshgrid(np.arange(sx),np.arange(sy))
    for k in range(sk):
        xx = XX-x_vec[k]
        yy = YY-y_vec[k]
        filt[k,:,:] = np.exp(-(xx**2+yy**2)/(2*sigma**2))
        
    fvol = np.fft.fft2(vol_square,axes=(1,2))
    fvol = fvol*filt
    vol = np.abs(np.fft.ifft2(fvol))
    vol = vol[:,:sy0,:sx0]
    return vol


class DataSet:
    def __init__(self,filename):

        if filename.lower()[-4:]=='mraw':
            self.mode = 'mraw'
        elif filename.lower()[-3:]=='tif':
            self.mode = 'tif'
        else:
            sys.exit('No action defined for files of type %s.'%(os.path.splitext(filename)[1]))
        
        if self.mode=='mraw':
            self.sx = ocfg.n_width
            self.sy = ocfg.n_height
            self.dtype = ocfg.mraw_dtype
            self.bytes_per_pixel = np.array([0],dtype=self.dtype).itemsize
            n_bytes = os.stat(filename).st_size
            n_frames = float(n_bytes)/(self.sx*self.sy*self.bytes_per_pixel)

            try:
                assert n_frames%1==0
            except AssertionError as ae:
                print ae
                print 'n_frames calculated to be non-integer value %f; please check that the n_width and n_height parameters are correct in oct_ui_config.py'%n_frames
                sys.exit()
            self.n_frames = int(round(n_frames))
            self.source = filename.replace('.mraw','')
            
        elif self.mode=='tif':
            temp = self.load_tif(filename)
            self.bytes_per_pixel = 2
            self.sy,self.sx = temp.shape
            self.dtype = np.uint16
            self.source = os.path.split(filename)[0]
            self.file_filter = os.path.join(self.source,'*.tif')
            self.file_list = glob.glob(self.file_filter)
            self.file_list.sort()
            self.n_frames = len(self.file_list)
            
        self.output_directory = self.source+'_processed'
        self.tiff_directory = self.source+'_projections'

        self.sz = ocfg.default_images_per_volume
        if self.n_frames<self.sz:
            sys.exit('Not enough frames in %s.'%self.source)
            
        self.blank_threshold = ocfg.blank_threshold
        self.filename = filename
        self.bytes_per_frame = self.sx*self.sy*self.bytes_per_pixel
        self.pixels_per_frame = self.sx*self.sy
        
        try:
            os.mkdir(self.output_directory)
        except Exception as e:
            print e
        try:
            os.mkdir(self.tiff_directory)
        except Exception as e:
            print e

        # read in all the frames and make a profile
        profile = np.zeros(self.n_frames)
        for k in range(self.n_frames):
            if k%1000==0:
                print 'loading frame %d of %d'%(k+1,self.n_frames)
            temp = self.load_frame(k)
            profile[k] = self.load_frame(k).mean()

        profile = np.array(profile)
        profile[np.where(profile<=self.blank_threshold)] = 0
        self.labels,n_features = spn.label(profile)

        self.start_indices = []
        self.end_indices = []
        for k in range(1,n_features+1):
            
            start = np.where(self.labels==k)[0][0]
            end = start+self.sz
            if end>self.n_frames:
                break
            else:
                self.start_indices.append(start)
                self.end_indices.append(end)
            
        self.n_volumes = len(self.start_indices)
        self.kstack = np.zeros((self.sz,self.sy,self.sx))
        print '(start,end) indices for %d volumes:'%self.n_volumes
        print zip(self.start_indices,self.end_indices)
        self.profile = profile
        self.load_kstack(0)
        self.current_kstack = 0

    def load_frame(self,idx):
        #print 'loading frame %d'%idx
        if self.mode=='tif':
            return self.load_tif(self.file_list[idx])
        elif self.mode=='mraw':
            offset = idx*self.bytes_per_frame
            count = self.pixels_per_frame
            with open(self.filename,'rb') as fid:
                fid.seek(offset)
                frame = np.fromfile(fid,self.dtype,count=count)
            frame = np.reshape(frame,(self.sy,self.sx))
            return frame
            
    def load_tif(self,fn):
        return np.array(Image.open(fn),dtype=np.float)
    
    def load_kstack(self,idx):
        start = self.start_indices[idx]
        end = self.end_indices[idx]
        for k in range(start,end):
            self.kstack[k-start,:,:] = self.load_frame(k)
        self.current_kstack = idx

    def save(self,d):
        out_fn = os.path.join(self.output_directory,'volume_%05d.mat'%self.current_kstack)
        proj_fn = os.path.join(self.tiff_directory,'volume_%05d_projection.tif'%self.current_kstack)
        d['starting_frame'] = self.start_indices[self.current_kstack]
        d['ending_frame'] = self.end_indices[self.current_kstack]
        
        print 'saving to %s'%out_fn
        sio.savemat(out_fn,d)
        bitmap = np.round(d['projection']*ocfg.tiff_multiplier).astype(np.uint16)
        tifffile.imwrite(proj_fn,bitmap)
            

class Action(QPushButton):
    def __init__(self,label,func):
        super(Action,self).__init__()
        self.func = func
        self.setText(label)
        self.clicked.connect(self.func)
        
class Number(QWidget):

    def __init__(self,label,default_value,slot_func,power=0.0,max_val=1e3,min_val=-1e3):
        super(Number,self).__init__()
        self.layout = QHBoxLayout()
        self.label = QLabel(label)
        self.layout.addWidget(self.label)
        if type(default_value)==int:
            self.box = QSpinBox()
        elif type(default_value)==float:
            self.box = QDoubleSpinBox()
            self.box.setDecimals(ocfg.coef_significant_digits)
            self.box.setSingleStep(ocfg.coef_single_step)
        self.power = power
        self.value = default_value
        self.box.setMaximum(max_val)
        self.box.setMinimum(min_val)
        self.box.setValue(self.value/(10**self.power))
        self.box.valueChanged.connect(self.set_value)
        self.func = slot_func
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.box)
        if self.power:
            mult_str = 'x 10^%0.1f'%self.power
            self.layout.addWidget(QLabel(mult_str))
        self.setLayout(self.layout)
        
    def set_value(self,val):
        self.value = val*(10**self.power)
        self.func(self.value)
        print 'Setting %s to %e.'%(self.label.text(),self.value)
        
    def set_maximum(self,val):
        self.box.setMaximum(val)
        
class OCTEngine(QObject):

    processed = pyqtSignal()
    projected = pyqtSignal()
    start_waiting = pyqtSignal()
    stop_waiting = pyqtSignal()
    
    def __init__(self):
        super(OCTEngine,self).__init__()
        self.c3_max = ocfg.c3_max
        self.c3_min = ocfg.c3_min
        self.c2_max = ocfg.c2_max
        self.c2_min = ocfg.c2_min
        self.c3 = ocfg.c3_default
        self.c2 = ocfg.c2_default
        self.L1 = ocfg.L1_default
        self.L2 = ocfg.L2_default
        self.pz1 = ocfg.projection_z1_default+ocfg.projection_z
        self.pz2 = ocfg.projection_z2_default+ocfg.projection_z
        
        self.cube = None
        self.fcube = None
        self.kscan = None
        self.bscan = None
        self.dc_subtract = ocfg.dc_subtract_default
        self.vsi = 0
        self.hsi = 0
        self.has_data = False
        self.use_filtered = ocfg.filter_volume_default
        
        
    def set_n_images(self,val):
        self.n_images = int(val)
        print self.n_images

    def load_files(self,file_list):
        if len(file_list)==0:
            return
        self.dataset = DataSet(file_list[0])
        self.change_kstack()
        
            
    def change_kstack(self):
        self.cube = self.dataset.kstack
        self.has_data = True
        self.sz = self.dataset.sz
        self.sy = self.dataset.sy
        self.sx = self.dataset.sx
        if self.use_filtered:
            self.start_waiting.emit()
            self.fcube = filter_volume(self.cube)
            self.stop_waiting.emit()
        self.process()
        
    def set_use_filtered(self,val):
        self.use_filtered = val
        if val and self.fcube is None and self.cube is not None:
            self.start_waiting.emit()
            self.fcube = filter_volume(self.cube)
            self.stop_waiting.emit()
        self.process()
    
    def set_dc_sub(self,val):
        self.dc_subtract = val
    
    def set_c3(self,val):
        self.c3 = val
        print 'engine c3 is %e'%self.c3
        self.process()

    def set_c2(self,val):
        self.c2 = val
        print 'engine c2 is %e'%self.c2
        self.process()

    def set_vertical_slice_index(self,val):
        self.vsi = val
        self.process()

    def set_horizontal_slice_index(self,val):
        self.hsi = val
        self.process()

    def set_L1(self,val):
        self.L1 = val
        self.process()

    def set_L2(self,val):
        self.L2 = val
        self.process()

    def set_pz1(self,val):
        self.pz1 = val
        self.processed.emit()
        
    def set_pz2(self,val):
        self.pz2 = val
        self.processed.emit()
        

    def process(self):
        if self.has_data:
            # choose a volume to use based on the filter checkbox
            if self.use_filtered:
                vol = self.fcube
            else:
                vol = self.cube
                
            self.kscan = vol[:,self.vsi,:]
            processed_frame,info = process(self.kscan,self.L1,self.L2,self.c3,self.c2,self.dc_subtract)
            n_depth = processed_frame.shape[0]
            
            # crop the complex conjugate:
            processed_frame = processed_frame[:n_depth/2,:]

            # crop out some DC as well
            self.bscan = np.abs(processed_frame)[:-ocfg.dc_crop_pixels,:]


            p = self.bscan.mean(1)
            rows_to_use = np.argsort(p)[:len(p)//3]
            noise_region = self.bscan[rows_to_use,:]
            
            temp = self.bscan.ravel()
            temp = list(temp)
            temp.sort()
            
            self.snr1 = 20*np.log10(np.max(temp)/np.std(temp[:len(temp)//4]))
            self.snr2 = 20*np.log10(np.max(temp)/noise_region.std())
            # ignore
            self.processed.emit()

    def process_volume(self):
        if self.has_data:
            
            if self.use_filtered:
                vol = self.fcube
            else:
                vol = self.cube
                
            self.start_waiting.emit()

            pvol,info = process(vol,self.L1,self.L2,self.c3,self.c2,dc_subtract=self.dc_subtract)
            self.pvol = pvol[:pvol.shape[0]//2,:,:]
            self.projection = np.abs(self.pvol[self.pz1:self.pz2,:,:]).mean(0)
            self.bscan = np.abs(self.pvol[:-ocfg.dc_crop_pixels,self.sy//2-5:self.sy//2+5,:]).mean(1)
            
            out_dict = {}
            out_dict['c3'] = self.c3
            out_dict['c2'] = self.c2
            out_dict['k_in'] = info.k_in
            out_dict['k_out'] = info.k_out
            out_dict['dispersion_phasor'] = info.phase
            out_dict['dc'] = info.dc
            out_dict['offaxis_filtered'] = self.use_filtered
            out_dict['volume'] = self.pvol
            out_dict['projection_z1'] = self.pz1
            out_dict['projection_z2'] = self.pz2
            out_dict['projection'] = self.projection
            self.dataset.save(out_dict)
            
            self.stop_waiting.emit()
            self.processed.emit()
            self.projected.emit()

    def process_all_volumes(self):
        for k in range(self.dataset.n_volumes):
            self.dataset.load_kstack(k)
            self.change_kstack()
            self.process_volume()
                
        
class App(QWidget):

    def __init__(self,flist=[]):
        super(App,self).__init__()

        try:
            os.mkdir('./.settings')
        except Exception as e:
            print e
        
        self.oct_engine = OCTEngine()
        self.oct_engine.load_files(flist)
            
        self.oct_engine.processed.connect(self.update_views)
        self.oct_engine.projected.connect(self.update_projection)
        self.oct_engine.start_waiting.connect(self.start_waiting)
        self.oct_engine.stop_waiting.connect(self.stop_waiting)
        
        self.left = 100
        self.top = 100
        self.title = 'OCT Parameter Explorer'
        self.width = 1200
        self.height = 800
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.main_layout = QHBoxLayout()
        self.control_layout = QVBoxLayout()

        self.bscan_view = ImageCanvas(width=4,height=4)
        self.ascan_view = PlotCanvas(width=4,height=4)
        self.projection_view = ImageCanvas(width=4,height=4)
        
        self.main_layout.addWidget(self.bscan_view)
        self.main_layout.addWidget(self.ascan_view)
        self.main_layout.addWidget(self.projection_view)

        # controls
        self.act_quit = Action('&Quit',sys.exit)
        self.act_load = Action('&Load',self.load_files)

        self.num_volume_index = Number('Volume index',0,self.set_volume_index,max_val=0,min_val=0)
        if self.oct_engine.has_data:
            self.num_volume_index.set_maximum(self.oct_engine.dataset.n_volumes-1)

        self.act_project = Action('Project volume',self.oct_engine.process_volume)
        self.act_process_all = Action('Process all volumes',self.oct_engine.process_all_volumes)

        
        self.control_layout.addWidget(self.act_quit)
        self.control_layout.addWidget(self.act_load)
        self.control_layout.addWidget(self.num_volume_index)
        self.control_layout.addWidget(self.act_project)
        self.control_layout.addWidget(self.act_process_all)

        self.num_c3 = Number('c3',self.oct_engine.c3,self.oct_engine.set_c3,ocfg.c3_power)
        self.num_c2 = Number('c2',self.oct_engine.c2,self.oct_engine.set_c2,ocfg.c2_power)
        self.control_layout.addWidget(self.num_c3)
        self.control_layout.addWidget(self.num_c2)

        self.num_L1 = Number('lambda 1',self.oct_engine.L1,self.oct_engine.set_L1,-9)
        self.num_L2 = Number('lambda 2',self.oct_engine.L2,self.oct_engine.set_L2,-9)
        self.control_layout.addWidget(self.num_L1)
        self.control_layout.addWidget(self.num_L2)

        self.num_pz1 = Number('projection z1',self.oct_engine.pz1,self.set_pz1,0)
        self.num_pz2 = Number('projection z2',self.oct_engine.pz2,self.set_pz2,0)
        self.control_layout.addWidget(self.num_pz1)
        self.control_layout.addWidget(self.num_pz2)
        
        self.cb_filter = QCheckBox('Filter volume')
        self.cb_filter.setChecked(self.oct_engine.use_filtered)
        self.cb_filter.stateChanged.connect(self.oct_engine.set_use_filtered)
        self.control_layout.addWidget(self.cb_filter)
        
        self.cb_dcsub = QCheckBox('DC subtract')
        self.cb_dcsub.setChecked(self.oct_engine.dc_subtract)
        self.cb_dcsub.stateChanged.connect(self.oct_engine.set_dc_sub)
        self.control_layout.addWidget(self.cb_dcsub)
        
        self.cb_log = QCheckBox('Log &scale')
        self.cb_log.setChecked(False)
        self.cb_log.stateChanged.connect(self.update_views)
        self.control_layout.addWidget(self.cb_log)
        
        self.ind_snr1 = Indicator('SNR1',fmt='%s = %0.2f dB')
        self.ind_snr2 = Indicator('SNR2',fmt='%s = %0.2f dB')
        self.control_layout.addWidget(self.ind_snr1)
        self.control_layout.addWidget(self.ind_snr2)

        self.main_layout.addLayout(self.control_layout)
        self.setLayout(self.main_layout)
        self.show()
        if self.oct_engine.has_data:
            self.update_views()

    def set_volume_index(self,val):
        self.oct_engine.dataset.load_kstack(int(val))
        self.oct_engine.change_kstack()
        
    def set_pz1(self,val):
        self.oct_engine.set_pz1(val)
        self.default_shear_offset = None

    def set_pz2(self,val):
        self.oct_engine.set_pz2(val)
        self.default_shear_offset = None

    @pyqtSlot()
    def start_waiting(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        
    @pyqtSlot()
    def stop_waiting(self):
        QApplication.restoreOverrideCursor()
        QApplication.processEvents()
        
    def load_files(self):
        dlg = QFileDialog()#self,'Choose TIF files','./2019.11.27')
        filter = '*%s'%ocfg.image_extension
        dlg.setFileMode(QFileDialog.ExistingFiles)
        try:
            with open('./.settings/last_directory.txt','rb') as fid:
                last_directory = fid.read().strip()
        except Exception as e:
            last_directory = ocfg.default_data_directory
            
        files = dlg.getOpenFileNames(self,'Choose %s files'%ocfg.image_extension,last_directory,filter)[0]
        if len(files):
            d = os.path.split(files[0])[0]
            with open('./.settings/last_directory.txt','wb') as fid:
                fid.write(d)
            
        self.oct_engine.load_files(files)
        self.num_volume_index.set_maximum(self.oct_engine.dataset.n_volumes-1)
        self.update_views()
        
    def update_views(self):
        bscan = self.oct_engine.bscan
        plim = (self.oct_engine.pz1,self.oct_engine.pz2)
        ascan = bscan.mean(1)
        self.ascan_view.plot(ascan,self.cb_log.isChecked(),plim)
        self.bscan_view.imshow(bscan,self.cb_log.isChecked(),plim)
        self.ind_snr1.set(self.oct_engine.snr1)
        self.ind_snr2.set(self.oct_engine.snr2)
        
    def update_projection(self):
        self.projection_view.imshow(self.oct_engine.projection,self.cb_log.isChecked(),aspect=None)


class Indicator(QLabel):

    def __init__(self,name,default_value=0.0,fmt='%s = %0.2f'):
        self.name = name
        self.fmt = fmt
        super(Indicator,self).__init__(self.fmt%(self.name,default_value))

    def set(self,val):
        self.setText(self.fmt%(self.name,val))
        
        
class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
    def plot(self,data,log=False,plims=(0,0)):
        self.axes.clear()
        if log:
            self.axes.semilogy(data, 'r-')
        else:
            self.axes.plot(data, 'r-')
        self.axes.axvline(plims[0])
        self.axes.axvline(plims[1])
        self.axes.grid(True)
        self.draw()


class ImageCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def imshow(self,data,log=False,plims=(0,0),aspect='auto',border=5):
        self.axes.clear()
        if log:
            im = np.log(data)
            clim = np.percentile(im[border:-border,border:-border],ocfg.bscan_log_contrast_percentile)
            self.axes.imshow(im,cmap='gray',aspect=aspect,clim=clim)
        else:
            clim = np.percentile(data,ocfg.bscan_linear_contrast_percentile)
            self.axes.imshow(data[border:-border,border:-border],cmap='gray',aspect=aspect,clim=clim)
        self.axes.axhline(plims[0])
        self.axes.axhline(plims[1])
        self.draw()

        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    if len(sys.argv)>1:
        flist = glob.glob(sys.argv[1])
        flist.sort()
        ex = App(flist)
    else:
        ex = App()
    sys.exit(app.exec_())

    
