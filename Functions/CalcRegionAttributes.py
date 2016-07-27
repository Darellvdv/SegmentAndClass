# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:54:05 2016

@author: darellvdv
"""

def calc_region_attributes(org_img, label_img, dsm = None, dtm = None):
    """
    CALCULATES OBJECT ATTRIBUTES FOR EACH REGION AND RETURNS A RAT
    
    org_img = orginal RGB image
    label_img = relabeld image (output from replace function)
    dsm = dsm of area (optional)
    dtm = dtm of area (optional)
    
    output = raster attribute table as numpy array
    
    """
    # Define index numbers
    index = np.unique(label_img)

    # Calculate GRVI and NDVI
    
    if args.att == 'rgb':
        print("Calculating Greennes..")
        grvi = 200*(org_img[:,:,1]*100.0) / (org_img[:,:,1]*100.0 + org_img[:,:,0]*100.0 + org_img[:,:,2]*100.0)
        mean_grvi = ndimage.mean(grvi, labels = label_img , index = index)
        
    if args.att == 'rgbnir':
        print("Calculating NDVI and Greennes..")
        grvi = 200*(org_img[:,:,1]*100.0) / (org_img[:,:,1]*100.0 + org_img[:,:,0]*100.0 + org_img[:,:,2]*100.0)
        mean_grvi = ndimage.mean(grvi, labels = label_img , index = index)
    
        ndvi = 100*(org_img[:,:,3]*100.0 - org_img[:,:,0]*100.0) / (org_img[:,:,3]*100.0 + org_img[:,:,0]*100.0) + 100
        mean_ndvi = ndimage.mean(ndvi, labels = label_img , index = index)

    # Calculate mean for all bands and heights and append to array
    if args.att == 'rgb':    
        w, h = 3, len(index)
    if args.att == 'rgbnir':
        w, h = 4, len(index)
    meanbands = np.zeros((h, w))
    varbands = np.zeros((h, w))
        
    if args.att == 'rgb':
        print("Averaging spectral RGB bands:")
        for i in tqdm.tqdm(xrange(0,3)):
            meanbands[:,i] = ndimage.mean(org_img[:,:,i].astype(float), labels = label_img, index = index)
            varbands[:,i] = ndimage.variance(org_img[:,:,i].astype(float), labels = label_img, index = index)
    else:
        print("Averaging spectral RGBNIR bands:")
        for i in tqdm.tqdm(xrange(0,4)):
            meanbands[:,i] = ndimage.mean(org_img[:,:,i].astype(float), labels = label_img, index = index)
            varbands[:,i] = ndimage.variance(org_img[:,:,i].astype(float), labels = label_img, index = index)
            
    print("Calculating ratios:")
    #r/g
    redgreen = meanbands[:,0] / meanbands[:,1]
    #g/r
    greenred = meanbands[:,1] / meanbands[:,0]
    #Blue/Nir
    bluenir = meanbands[:,2] / meanbands[:,3]
    #Green/Nir
    greennir = meanbands[:,1] / meanbands[:,3]

    ratios = np.vstack((redgreen, greenred, bluenir, greennir)).T
            
    print("Calculating roughness..")
    call('gdaldem roughness '+args.output+'/'+imagenamenoext[0]+'_noalpha.tif '+args.output+'/'+imagenamenoext[0]+'_texture.tif -of GTiff -b 1')
    texture = tiff.imread(args.output+'/'+imagenamenoext[0]+'_texture.tif')
    blurred = gaussian_filter(texture, sigma=7) # apply gaussian blur for smoothing
    meantexture = np.zeros((len(index), 1))
    meantexture[:,0] = ndimage.mean(blurred.astype(float), labels = label_img, index = index)
    
    print("Calculating vissible brightness..")
    vissiblebright = (org_img[:,:,0] + org_img[:,:,1] + org_img[:,:,2]) / 3
    meanbrightness = np.zeros((len(index), 1))
    meanbrightness[:,0] = ndimage.mean(vissiblebright.astype(float), labels = label_img, index = index)
    
    print("Calculating region props:")
    # Define the regions props and calulcate
    regionprops_per = []
    regionprops_ar = []
    regionprops_ex = []
    regionprops_ecc = []
    
    for region in tqdm.tqdm(regionprops(label_img)):
        regionprops_per.append(region.perimeter)
        regionprops_ar.append(region.area)
        regionprops_ex.append(region.extent)
        regionprops_ecc.append(region.eccentricity)

    # Convert regionprops results to numpy array
    regionprops_calc = np.vstack((regionprops_per, regionprops_ar, regionprops_ex, regionprops_ecc)).T

    # Calculate coordinate attributes of regions
    regionprops_cen = []
    regionprops_coor = []
    
    print("Calculating region coordinates:")
    for region in tqdm.tqdm(regionprops(label_img)):
        regionprops_cen.append(region.centroid)
        regionprops_coor.append(region.coords)
        
    # Calculate distance between points
    xcoords = []
    ycoords = []
    print("Calculating distance between points:")
    for i in tqdm.tqdm((xrange(len(regionprops_cen)))):
        xcoords.append(regionprops_cen[i][0])
        ycoords.append(regionprops_cen[i][1])
        
    xcoords = np.asarray(xcoords)
    ycoords = np.asarray(ycoords)

    # Initialize empty distance array for all points
    #distances = np.zeros(((len(regionprops_cen)), (len(regionprops_cen))))
    #distances_new = np.zeros(((len(regionprops_cen)), (len(regionprops_cen))))

    #mindist = []
    #for i in tqdm.tqdm((xrange(len(regionprops_cen)))):
    #    distances = []
    #    distances.append(math.hypot(xcoords[i] - xcoords[i], ycoords[i] - ycoords[i]))
        
        
    #    for j in (xrange(len(regionprops_cen))):
    #        distances = []
    #        distances.append(math.hypot(xcoords[i] - xcoords[j], ycoords[i] - ycoords[j]))
    
    # Calculate distances for all points
    #for i in (xrange(len(regionprops_cen))):
    #    for j in (xrange(len(regionprops_cen))):
    #        distances[i][j] = math.hypot(xcoords[i] - xcoords[j], ycoords[i] - ycoords[j])

    #distances_new = np.ma.masked_where(distances == 0.00, distances)

    # Determine distance to nearest point
    #mindist = []
    #for i in (xrange(len(distances_new))):
    #    mindist.append(np.min(distances_new[i]))        
    #mindist = np.asarray(mindist)

    # Collect all data in 1 np array
    if dsm != None:
        # Calculate crop surface model
        print("Calculating crop surface model..")
        csm = dsm.astype(float) - dtm.astype(float)     
        heights = np.dstack((dsm, dtm, csm))
    
        # Resample heights
        print("Resampling heights..")
        heights_res = ndimage.zoom(heights, (2, 2, 1))
        meanheight = np.zeros((h, w))
        meanheight[:,i] = ndimage.mean(heights_res[:,:,i].astype(float), labels = label_img, index = index)
        
        data = np.concatenate((meanbands, varbands, ratios, meanheight, regionprops_calc, meantexture, meanbrightness), axis=1)
        if np.shape(org_img)[2] == 3:
            data2 = np.column_stack(((index.astype(int), regionslist, mean_grvi, data)))
        else:
            data2 = np.column_stack(((index.astype(int), regionslist, mean_ndvi, mean_grvi, data)))
        print("Data collected, attributes calculated..all done!")

    else:
        data = np.concatenate((meanbands, varbands, ratios, regionprops_calc, meantexture, meanbrightness), axis=1)
        if args.att == 'rgb':
            print("grvi, data")
            data2 = np.column_stack(((index.astype(int), regionslist, mean_grvi, data)))
        else:
            print("ndvi, grvi, data")
            data2 = np.column_stack(((index.astype(int), regionslist, mean_ndvi, mean_grvi, data)))
        print("Data collected, attributes calculated..all done!")
        
    return(data2)