    return


    
    def find_flat_regions(vec,thresh=1.0):
        left = vec[:-2]
        center = vec[1:-1]
        right = vec[2:]

        end = [False]
        flat_to_left = np.abs(center-left)<=thresh
        flat_to_right = np.abs(center-right)<=thresh
        return np.array([False]+list(flat_to_left*flat_to_right)+[False])
        

    xflat = find_flat_regions(cpeakx_vec)
    yflat = find_flat_regions(cpeaky_vec)
    flat = np.logical_and(xflat,yflat)
    flat_indices = np.where(flat)[0]
    labeled, num_objects = spn.label(flat)    

    plt.subplot(2,1,1)
    plt.plot(cpeakx_vec)
    plt.plot(flat_indices,cpeakx_vec[flat_indices],'ro')
    plt.plot(cpeaky_vec)
    plt.plot(flat_indices,cpeaky_vec[flat_indices],'ro')
    plt.plot(labeled)
    plt.subplot(2,1,2)
    plt.plot(np.arange(sy),corr_peaks)

    find_flat_regions(cpeakx_vec)
    
    plt.show()

    
    # clim = np.percentile(nxc_stack,(80,100))
    # for iy in range(sy):
    #     plt.cla()
    #     plt.imshow(nxc_stack[iy,:,:],cmap='gray',aspect='auto',clim=clim)
    #     plt.pause(.1)
