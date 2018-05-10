#GrowingArray
#Written by: GhostOTM
#Last updated: 5.3.18

#=========================================================================================================
#CODE START
#=========================================================================================================

import numpy as np

class GrowingArray:
    """
    Class description:
        A modular array-handler which grows the array when values are placed in nonexistent
        indices, be those indices above or below the existing one.

    Public variables:
        data--
            The modular 3D numpy array data is being stored on. Indexing format is (t,y,x)
        zeropoint--
            The tuple containing the (y,x) coordinates of the reference (0,0) index after it
            has been moved around due to array resizing
        avgFunct--
            The function used to determine how to treat data placed into indices that already
            have something in them

    Keyword Arguments:
        startDims--
            Takes tuple with either two (y,x) or three (t,y,x) components which sets
            the initial dimensions of self.data. Initially set to None, which causes a (1,1,1)
            array to be created for self.data.
        avgFunct--
            Allows for the passage of a function as a means of handling overlaps. The __averagingFunction
            needs to take two array-form inputs and output a corresponding array of the same size. Intially
            set to None, which cases a private avgFunct to be called which simply takes two inputs (a and b)
            and returns (a+b)/2

    Public Methods:
        put--
            Takes an array input and adds it to self.data, resizing where necessary
    """
    data = np.empty((1,1,1))
    data[:] = np.nan
    zeropoint = (0,0)
    avgFunct=None

#=========================================================================================================
    #Constructor for the class. Parses through startDim and handles it differently depending
    #on whether it is a 2-int or 3-int tuple.
    def __init__(self,startDim=None,avgFunct=None):
        #handles the startDim and creates the appropriate starting self.data
        if isinstance(startDim, tuple):
            if len(startDim)==2:
                self.data = np.empty((1,startDim[0],startDim[1]))
            elif len(startDim)==3:
                self.data = np.empty(startDim)
            self.data[:] = np.nan
        #replaces the avgFunct built into this code with one the user can input
        if avgFunct is not None:
            self.avgFunct=avgFunct
        else:
            self.avgFunct=self.__averagingFunction

#=========================================================================================================
    #put function which allows the placing of "strips" into the GrowingArray at "coords"
    def put(self, strip=None, coords=None):
        """
            Function Description:
                Takes an inputed array and the location within the Growing Array you would
                like to place it.

            Function Dataflow:
                1) Checks to see whether you are inserting at an existing Z coordinate. If not,
                it creates those coordinates in the self.data array
                2) Checks whether the X and Y coordinates are within the range of the existing
                self.data array. If not, it calls self.__resizeXY to expand itself.
                3) Checks if the indices the new data is trying to be added in already has something
                in them. If so, it is handled with avgFunct. This is done throught calling
                self.__averageOverlap.
                4) The new data is added to self.data.

            Keyword Arguments:
                strip--
                    The data you are trying to pass into the GrowingArray. Must be a 2D numpy arrayself.
                    Initially set to None.
                cords--
                    Takes a tuple representing the coordinates you want to place the "strip" in. It must be
                    a 3-integer datapoint in format (t,y,x) and the coordinates are in referenace to the
                    zeropoint. Initially set to None.

        """
        #Checks if the desired Z coordinate exists and adds it if necessary
        z=coords[0]
        if z>=self.data.shape[0]:
            self.__resizeZ(z)
        #adjusts inputted coordinates for the zeropoint to make xlow, ylow, xhigh, and yhigh in terms of the
        #actual array's corods
        xlow=self.zeropoint[1]+coords[2]
        ylow=self.zeropoint[0]+coords[1]
        xhigh=xlow+strip.shape[1]
        yhigh=ylow+strip.shape[0]
        #checks if resizing is necessary
        if xlow<0 or ylow<0 or xhigh > self.data.shape[2]-self.zeropoint[1] or yhigh > self.data.shape[1]-self.zeropoint[0]:
            #resizes if necessary
            self.__resizeXY((ylow, xlow, yhigh, xhigh))
            #recalculates the adjusted coorinates in case the zeropoint changed
            xlow=self.zeropoint[1]+coords[2]
            ylow=self.zeropoint[0]+coords[1]
            xhigh=xlow+strip.shape[1]
            yhigh=ylow+strip.shape[0]
        #checks for and handles any place where you are trying to add data over exisiting data. Overwrites the inputted strips
        #with a version adjusted at the overlapping data positions in whatever way self.avgFunct dictates
        strip=self.__averageOverlap((z,ylow,xlow),strip)
        #adds the newly possibly-adjusted strip to the possibly-resized growing array
        self.data[z,ylow:yhigh,xlow:xhigh]=strip

#=========================================================================================================
    #Handles any sutations where data is trying to be overwritten
    def __averageOverlap(self,coords,strip):
        #identifies the place on self.data that the strip is going to be placed
        layspot=self.data[coords[0],coords[1]:(coords[1]+strip.shape[0]),coords[2]:(coords[2]+strip.shape[1])]
        #creats two masks of nan locations and uses them, as well as self.avgFunct to correct only the locations
        #where  data overlap is occuring
        maskStrip=np.isnan(strip)
        maskLayspot=np.isnan(layspot)
        mask_keep_Strip = ~maskStrip & maskLayspot
        mask_keep_Layspot = maskStrip & ~maskLayspot
        averageOut = self.avgFunct(strip,layspot)
        averageOut[mask_keep_Strip] = strip[mask_keep_Strip]
        averageOut[mask_keep_Layspot] = layspot[mask_keep_Layspot]
        #returns a version of the strip with the overlap spots adjusted so "overwriting" those spots will actually
        #represent some form of averaging of the old and new data
        return averageOut

#=========================================================================================================
    #Takes the offset the strip is trying to be entered at and determines if the array needs to be resized. If it does
    #it finds out which dimension(s) need to be grown and grows them
    def __resizeXY(self,offset):
        ydif=offset[0]
        xdif=offset[1]
        ymax=offset[2]-self.data.shape[1]
        xmax=offset[3]-self.data.shape[2]
        #The array needs to be extended in the -Y direction (up)
        if(ydif<0):
            temp=np.empty((self.data.shape[0],abs(ydif),self.data.shape[2]))
            temp[:]=np.nan
            self.data=np.append(temp,self.data,axis=1)
            self.zeropoint=(self.zeropoint[0]-ydif,self.zeropoint[1])
        #The array needs to be extended in the +Y direction (down)
        if(ymax>0):
            temp=np.empty((self.data.shape[0],ymax,self.data.shape[2]))
            temp[:]=np.nan
            self.data=np.append(self.data,temp,axis=1)
        #The array needs to be extended in the -X direction (left)
        if(xdif<0):
            temp=np.empty((self.data.shape[0],self.data.shape[1],abs(xdif)))
            temp[:]=np.nan
            self.data=np.append(temp,self.data, axis=2)
            self.zeropoint=(self.zeropoint[0],self.zeropoint[1]-xdif)
        #The array needs to be extended in the +X direction (rights)
        if (xmax>0):
            temp=np.empty((self.data.shape[0],self.data.shape[1],xmax))
            temp[:]=np.nan
            self.data=np.append(self.data,temp,axis=2)

#=========================================================================================================
    #This funciton is used to determine how two arrays are handled if they have overlapping
    #coordinates. By defult it takes and returns the average
    def __averagingFunction(self,array1,array2):
        return (array1+array2)/2

#=========================================================================================================
    #This function adds Z-layers to the existing self.data array. It is called if the requested
    #put location is not within the current z range.
    def __resizeZ(self,length):
        temp = np.empty((length+1,self.data.shape[1],self.data.shape[2]))
        temp[:self.data.shape[0]] = self.data
        temp[self.data.shape[0]:] = np.nan
        self.data=temp

#=========================================================================================================
#CODE END
#=========================================================================================================
