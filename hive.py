import os,sys,glob
import shutil
import numpy as np

class Hive():
    
    def __init__(self,root_location,mode='w'):
        self.mode = mode
        self.root_location = root_location
        try:
            os.makedirs(self.root_location)
            print 'Made directory %s'%self.root_location
        except OSError as e:
            print 'Using preexisting directory %s'%self.root_location
        
    def has(self,key):
        return os.path.exists(os.path.join(self.root_location,key+'.npy'))
    
    def keys(self):
        temp = glob.glob(os.path.join(self.root_location,'*'))
        out = []
        for t in temp:
            t = t.replace(self.root_location,'')
            if t[-4:].lower()=='.npy':
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
        if os.path.exists(path_to_delete+'.npy'):
            os.remove(path_to_delete)
        elif os.path.exists(path_to_delete):
            shutil.rmtree(path_to_delete)
            
    def __getitem__(self,key):
        if key[0]=='/':
            key = key[1:]

        
        npy_fn = os.path.join(self.root_location,key)+'.npy'
        if os.path.exists(npy_fn):
            return np.load(npy_fn)
        
        else:
            return Hive(os.path.join(self.root_location,key))

    def __setitem__(self,location,data):
        if location[0]=='/':
            location = location[1:]

        if type(data)==list:
            data = np.array(data)
        elif not type(data)==np.ndarray:
            data = np.array([data])
            
        if self.mode.find('w')==-1:
            sys.exit('Mode is %s; cannot put. Exiting.'%self.mode)

        fn = os.path.join(self.root_location,location)+'.npy'

        path,shortfn = os.path.split(fn)
        if not os.path.exists(path):
            os.makedirs(path)
        
        try:
            os.remove(fn)
        except Exception as e:
            pass
        print fn
        np.save(fn,data)
        
