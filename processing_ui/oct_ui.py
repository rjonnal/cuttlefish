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
import glob

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
        self.pz1 = ocfg.projection_z1_default
        self.pz2 = ocfg.projection_z2_default
        
        self.cube = None
        self.fcube = None
        self.hframe = None
        self.vframe = None
        self.dc_subtract = ocfg.dc_subtract_default
        self.vsi = 0
        self.hsi = 0
        self.has_data = False
        self.use_filtered = ocfg.filter_volume_default
        
        
    def set_n_images(self,val):
        self.n_images = int(val)
        print self.n_images

    def load_files(self,file_list):
        if not len(file_list):
            return
        if len(file_list)==1:
            temp = file_list[0]
            nfiles = 1
            while nfiles<ocfg.default_images_per_volume:
                temp = temp[:-1]
                flist = glob.glob(temp+'*%s'%ocfg.image_extension)
                nfiles = len(flist)
            flist.sort()
            file_list = flist[:ocfg.default_images_per_volume]

        
        self.file_list = file_list
        self.file_list.sort()
        self.sz = len(self.file_list)

        # make a volume filename
        self.volume_filename = '%s_volume_%03d.mat'%(self.file_list[0].replace(ocfg.image_extension,''),self.sz)
        
        self.sy,self.sx = self.load_tif(self.file_list[0]).shape
        self.cube = np.zeros((self.sz,self.sy,self.sx))
        for idx,f in enumerate(self.file_list):
            print 'loading %s'%f
            self.cube[idx,:,:] = self.load_tif(f)
        if self.use_filtered:
            self.start_waiting.emit()
            self.fcube = self.filter_volume(self.cube)
            self.stop_waiting.emit()
        else:
            self.fcube = None
        self.has_data = True
        self.process()

    def load_tif(self,fn):
        return np.array(Image.open(fn),dtype=np.float)

    def set_use_filtered(self,val):
        self.use_filtered = val
        if val and self.fcube is None and self.cube is not None:
            self.start_waiting.emit()
            self.fcube = self.filter_volume(self.cube)
            self.stop_waiting.emit()
    
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
        
    def filter_volume(self,vol):
        x0 = ocfg.offaxis_x0
        y0 = ocfg.offaxis_y0
        sigma = ocfg.offaxis_sigma
        
        # 2D FFT each en face slice in the volume, multiply by a
        # gaussian centered about one of the off-axis shifted images,
        # and 2D FFT back

        sk,sy0,sx0 = vol.shape

        vol_square = np.zeros((sk,max(sx0,sy0),max(sx0,sy0)))
        vol_square[:,:sy0,:sx0] = vol

        sk,sy,sx = vol_square.shape

        xx,yy = np.meshgrid(np.arange(sx),np.arange(sy))
        xx = xx-x0
        yy = yy-y0
        g = np.exp(-(xx**2+yy**2)/(2*sigma**2))
        fvol = np.fft.fft2(vol_square,axes=(1,2))

        fvol = fvol*g
        vol = np.abs(np.fft.ifft2(fvol))
        vol = vol[:,:sy0,:sx0]
        return vol

    def process(self):
        if self.has_data:
            # generate an array of k-values corresponding to the
            # linearly spaced lambda values of the acquired stack;
            # these k values are not linearly spaced
            k_in = 2*np.pi/np.linspace(self.L1,self.L2,self.sz)
            
            # generate linearly spaced k values between the first and
            # last k values of k_in
            k_out = np.linspace(k_in[0],k_in[-1],len(k_in))
            
            # choose a volume to use based on the filter checkbox
            if self.use_filtered:
                vol = self.fcube
            else:
                vol = self.cube
                
            self.hframe = vol[:,:,self.hsi]
            self.vframe = vol[:,self.vsi,:]
            
            for frame in [self.hframe,self.vframe]:
            
                
                if self.dc_subtract:
                    # estimate the DC by averaging
                    # all the k-scans in the frame together;
                    # we assume that decorrelation of intensity
                    # and phase among the k-scans will cause
                    # all of the fringes to be averaged out, such
                    # that we're left with incoherent DC
                    dc = np.mean(frame,axis=1)
                    frame = (frame.T-dc).T

                # interpolate image from nonlinear k (k_in) into linear k (k_out)
                k_interpolator = spi.interp1d(k_in,frame,axis=0,copy=False)
                processed_frame = k_interpolator(k_out)

                # generate a set of k values to define the phase polynomial
                dispersion_axis = k_out - np.mean(k_out)
                # for 3rd order polynomial, polyval requires four numbers; put
                # in zeros for linear and constant coefs:
                dispersion_coefficients = [self.c3,self.c2,0.0,0.0]

                # create phase polynomial to dechirp spectra:
                phase = np.exp(1j*np.polyval(dispersion_coefficients,dispersion_axis))
                
                # multiply phase polynomial by fringe image
                processed_frame = (processed_frame.T * phase).T
                oversample_factor = 1
                
                # fft with respect to k and fftshift to bring DC to the center of the b-scan
                processed_frame = np.fft.fftshift(np.fft.fft(processed_frame,n=processed_frame.shape[0]*oversample_factor,axis=0),axes=0)
                n_depth = processed_frame.shape[0]
                
                # crop the complex conjugate:
                processed_frame = processed_frame[:n_depth/2,:]

                # crop out some DC as well
                self.bscan = np.abs(processed_frame)[:-ocfg.dc_crop_pixels,:]
                
                # ignore
                self.processed.emit()

    def process_volume(self):
        if self.has_data:
            self.start_waiting.emit()
            k_in = 2*np.pi/np.linspace(self.L1,self.L2,self.sz)
            k_out = np.linspace(k_in[0],k_in[-1],len(k_in))
            if self.use_filtered:
                vol = self.fcube
            else:
                vol = self.cube
                
            if self.dc_subtract:
                dc = vol.mean(2).mean(1)
                vol = np.transpose(np.transpose(vol,[1,2,0])-dc,[2,0,1])
            else:
                dc = -1
            
            sk,sy,sx = vol.shape
            k_interpolator = spi.interp1d(k_in,vol,axis=0,copy=False)
            pvol = k_interpolator(k_out)

            dispersion_axis = k_out - np.mean(k_out)
            dispersion_coefficients = [self.c3,self.c2,0.0,0.0]
            oversample_factor = 1
            phase = np.exp(1j*np.polyval(dispersion_coefficients,dispersion_axis))
            pvol = (pvol.transpose([1,2,0])*phase).transpose([2,0,1])
            pvol = np.fft.fftshift(np.fft.fft(pvol,n=pvol.shape[0]*oversample_factor,axis=0),axes=0)
            self.pvol = pvol[:pvol.shape[0]//2,:,:]
            self.projection = np.abs(self.pvol[self.pz1:self.pz2,:,:]).mean(0)
            bscan = np.abs(self.pvol[:-ocfg.dc_crop_pixels,self.sy//2-5:self.sy//2+5,:]).mean(1)
            self.stop_waiting.emit()
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(bscan,cmap='gray',clim=np.percentile(bscan,(20,99.9)))
            plt.axhline(self.pz1)
            plt.axhline(self.pz2)
            plt.subplot(1,2,2)
            plt.imshow(self.projection,cmap='gray')
            plt.pause(.1)

            out_dict = {}
            out_dict['c3'] = self.c3
            out_dict['c2'] = self.c2
            out_dict['k_in'] = k_in
            out_dict['k_out'] = k_out
            out_dict['dispersion_phasor'] = phase
            out_dict['dc'] = dc
            out_dict['offaxis_filtered'] = self.use_filtered
            out_dict['volume'] = self.pvol
            out_dict['projection_z1'] = self.pz1
            out_dict['projection_z2'] = self.pz2
            out_dict['projection'] = self.projection
            
            sio.savemat(self.volume_filename,out_dict)
            
            self.projected.emit()
                
        
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
        self.act_project = Action('Project volume',self.oct_engine.process_volume)
        self.control_layout.addWidget(self.act_quit)
        self.control_layout.addWidget(self.act_load)
        self.control_layout.addWidget(self.act_project)

        self.num_c3 = Number('c3',self.oct_engine.c3,self.oct_engine.set_c3,ocfg.c3_power)
        self.num_c2 = Number('c2',self.oct_engine.c2,self.oct_engine.set_c2,ocfg.c2_power)
        self.control_layout.addWidget(self.num_c3)
        self.control_layout.addWidget(self.num_c2)

        self.num_L1 = Number('lambda 1',self.oct_engine.L1,self.oct_engine.set_L1,-9)
        self.num_L2 = Number('lambda 2',self.oct_engine.L2,self.oct_engine.set_L2,-9)
        self.control_layout.addWidget(self.num_L1)
        self.control_layout.addWidget(self.num_L2)

        self.num_pz1 = Number('projection z1',self.oct_engine.pz1,self.oct_engine.set_pz1,0)
        self.num_pz2 = Number('projection z2',self.oct_engine.pz2,self.oct_engine.set_pz2,0)
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
        
        self.main_layout.addLayout(self.control_layout)
        self.setLayout(self.main_layout)
        self.show()

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
        self.update_views()
        
    def update_views(self):
        bscan = self.oct_engine.bscan
        ascan = bscan.mean(1)
        plim = (self.oct_engine.pz1,self.oct_engine.pz2)
        self.ascan_view.plot(ascan,self.cb_log.isChecked(),plim)
        self.bscan_view.imshow(bscan,self.cb_log.isChecked(),plim)

    def update_projection(self):
        self.projection_view.imshow(self.oct_engine.projection,self.cb_log.isChecked(),aspect=None)
        
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

    def imshow(self,data,log=False,plims=(0,0),aspect='auto'):
        self.axes.clear()
        if log:
            im = np.log(data)
            clim = np.percentile(im,ocfg.bscan_log_contrast_percentile)
            self.axes.imshow(im,cmap='gray',aspect=aspect,clim=clim)
        else:
            clim = np.percentile(data,ocfg.bscan_linear_contrast_percentile)
            self.axes.imshow(data,cmap='gray',aspect=aspect,clim=clim)
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

    
