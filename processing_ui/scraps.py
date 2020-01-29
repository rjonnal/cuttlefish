    def shear(self,data,plims=(0,0),default_shear_offset=None):
        def helper(data,k):
            sy,sx = data.shape
            subdat = np.zeros(data.shape)
            for x in range(sx):
                offset = int(round((x-(sx//2))*k))
                get_z1 = offset
                get_z2 = offset+sy
                put_z1 = 0
                put_z2 = sy
                if get_z1<0:
                    upper_crop = -get_z1
                    get_z1 = get_z1 + upper_crop
                    put_z1 = put_z1 + upper_crop
                if get_z2>sy:
                    lower_crop = -(sy-get_z2)
                    get_z2 = get_z2 - lower_crop
                    put_z2 = put_z2 - lower_crop

                subdat[put_z1:put_z2,x] = data[get_z1:get_z2,x]
            return subdat

        if default_shear_offset is None:
            sy,sx = data.shape
            plims = list(plims)
            if plims[0]<0:
                plims[0] = 0
            if plims[1]<0:
                plims[1] = 1
            if plims[0]>=sy:
                plims[0]=sy-2
            if plims[1]>=sy:
                plims[1]=sy-1
            offset_vec = np.linspace(-0.05,0.05,100)
            profile_maxes = []
            for offset in offset_vec:
                profile_maxes.append(helper(data,offset)[plims[0]:plims[1],:].mean(1).max())
            winning_offset = offset_vec[np.argmax(profile_maxes)]
        else:
            winning_offset = default_shear_offset
        
        return helper(data,winning_offset),winning_offset

########################################################

