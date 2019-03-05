import os,sys,glob
import shutil
import numpy as np
import scipy.io as sio

class Hive():
    
    def __init__(self,root_location,mode='w',file_format='mat'):
        self.mode = mode
        self.root_location = root_location
        try:
            os.makedirs(self.root_location)
            print 'Made directory %s'%self.root_location
        except OSError as e:
            print 'Using preexisting directory %s'%self.root_location
        self.file_format = file_format

    def read(self,basename):
        print 'read basename:'+basename
        print self.file_format
        if self.file_format=='npy':
            return np.load(basename+'.npy')
        elif self.file_format=='mat':
            tag = os.path.split(basename)[1]
            fn = basename+'.mat'
            return sio.loadmat(fn)[tag]
    
    def write(self,basename,data):
        # print 'hive.write'
        # print basename
        # print type(data)
        #print data
        #print
        #if not basename.find('oversample')==-1:
        #    sys.exit()
        if self.file_format=='npy':
            outfn = basename+'.npy'
            np.save(outfn,data)
        elif self.file_format=='mat':
            outfn = basename+'.mat'
            tag = os.path.split(basename)[1]
            d = {}
            d[tag] = data
            sio.savemat(outfn,d)
    
    def has(self,key):
        return os.path.exists(os.path.join(self.root_location,key+'.%s'%self.file_format))

    def other(self):
        if self.file_format=='mat':
            return 'npy'
        elif self.file_format=='npy':
            return 'mat'
    
    def keys(self):
        temp = glob.glob(os.path.join(self.root_location,'*'))
        out = []
        for t in temp:
            t = t.replace(self.root_location,'')
            # ignore the wrong kind of files in case the directory
            # is messed up:
            if t[-4:].lower()=='.%s'%self.other():
                continue
            if t[-4:].lower()=='.%s'%self.file_format:
                t = t[:-4]
            while t[0]=='/':
                t = t[1:]
            while t[0]=='\\':
                t = t[1:]
            out.append(t)
        return out
        
    def put(self,location,data):
        self[location] = data

    def get(self,location):
        return self[location]

    def delete(self,location):
        path_to_delete = os.path.join(self.root_location,location)
        if os.path.exists(path_to_delete+'.%s'%self.file_format):
            os.remove(path_to_delete)
        elif os.path.exists(path_to_delete):
            shutil.rmtree(path_to_delete)
            
    def __getitem__(self,key):
        if key[0]=='/':
            key = key[1:]

        print self.root_location,key
        
        dat_basename = os.path.join(self.root_location,key)
        print dat_basename
        try:
            return self.read(dat_basename)
        except Exception as e:
            print 'getitem error: %s'%e
            return Hive(os.path.join(self.root_location,key))

    def __setitem__(self,location,data):
        if location[0]=='/':
            location = location[1:]

        if type(data)==list:
            data = np.array(data)
        elif not type(data)==np.ndarray and self.file_format=='npy':
            data = np.array([data])
            
        if self.mode.find('w')==-1:
            sys.exit('Mode is %s; cannot put. Exiting.'%self.mode)

        
        basename = os.path.join(self.root_location,location)

        path,shortfn = os.path.split(basename)
        if not os.path.exists(path):
            os.makedirs(path)

        self.write(basename,data)
