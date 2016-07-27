# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:54:05 2016

@author: darellvdv
"""

def calc_region_differences(label_img, rat):
    """
    DEFINES NEIGHBOR REGIONS AND CALCULATE ATTRIBUTES FOR NEIGHBORS
    
    label_img = relabeled image (from 1 to n) as numpy array
    rat = numpy array containing attributes as outputted from calc_regions_attributes
    
    output = updated rat with differences in percentage
    
    """
    resultarr = determine_neighbours(label_img)
        
    # Define wanted area and calculate mean value for al its neighbours
    result = []
    if args.att == 'rgb': # Only RGB no height
        print("Using Greeennes, texture and vissible brightness")
        meancalc = [2, 20, 21] # only use greennes

        print("Calculating avarage attributes for per neighbour:")
        for neighbours in tqdm.tqdm(resultarr):
            wantedneigh = rat[np.logical_or.reduce([rat[:,0] == x for x in neighbours])]
            for i in meancalc:
                result.append(np.mean(wantedneigh[0][i]))
                
        # Collect results in array and reshape to attribute table format
        meanvalues = np.asarray(result)
        meanvalresh = meanvalues.reshape(len(regionslist), 3)

        # Calculate differences in percentage
        greennesdiff = ((meanvalresh[:,0] - rat[:,2] ) / ((meanvalresh[:,0] + rat[:,2] )/2) ) * 100
        texturediff = ((meanvalresh[:,1] - rat[:,20] ) / ((meanvalresh[:,1] + rat[:,20] )/2) ) * 100
        brightnessdiff = ((meanvalresh[:,2] - rat[:,21] ) / ((meanvalresh[:,2] + rat[:,21] )/2) ) * 100

        diffattributes = np.vstack((greennesdiff, texturediff, brightnessdiff))
        allatributes = np.concatenate((rat, diffattributes), axis=1)
    
        print("Avarage attributes for neighbours calculated.. all done!")
        
    elif args.att == 'rgbheight': # RGB and height
        print("Using Greennes, height, texture and vissible brightness")
        meancalc = [2, 9, 20, 21] # Greennes and height

        print("Calculating avarage attributes for per neighbour:")
        for neighbours in tqdm.tqdm(resultarr):
            wantedneigh = rat[np.logical_or.reduce([rat[:,0] == x for x in neighbours])]
            for i in meancalc:
                result.append(np.mean(wantedneigh[0][i]))
                
        # Collect results in array and reshape to attribute table format
        meanvalues = np.asarray(result)
        meanvalresh = meanvalues.reshape(len(regionslist), 4)

        # Calculate differences in percentage
        greennesdiff = ((meanvalresh[:,0] - rat[:,2] ) / ((meanvalresh[:,0] + rat[:,2] )/2) ) * 100 
        heightdiff = ((meanvalresh[:,1] - rat[:,9] ) / ((meanvalresh[:,1] + rat[:,9] )/2) ) * 100
        texturediff = ((meanvalresh[:,2] - rat[:,20] ) / ((meanvalresh[:,2] + rat[:,20] )/2) ) * 100
        brightnessdiff = ((meanvalresh[:,3] - rat[:,21] ) / ((meanvalresh[:,3] + rat[:,21] )/2) ) * 100

        diffattributes = np.vstack((greennesdiff, heightdiff, texturediff, brightnessdiff)).T
        allatributes = np.concatenate((rat, diffattributes), axis=1)
    
        print("Avarage attributes for neighbours calculated.. all done!")
        
    elif args.att == 'rgbnir': # RGB and nir
        print("Using NDVI, Greennes, texture and vissible brightness")
        meancalc = [2, 3, 20, 21] # NDVI and Greennes

        print("Calculating avarage attributes for per neighbour:")
        for neighbours in tqdm.tqdm(resultarr):
            wantedneigh = rat[np.logical_or.reduce([rat[:,0] == x for x in neighbours])]
            for i in meancalc:
                result.append(np.mean(wantedneigh[0][i]))
                
        # Collect results in array and reshape to attribute table format
        meanvalues = np.asarray(result)
        meanvalresh = meanvalues.reshape(len(regionslist), 4)

        # Calculate differences in percentage
        ndvidiff = ((meanvalresh[:,0] - rat[:,2] ) / ((meanvalresh[:,0] + rat[:,2] )/2) ) * 100 
        greennesdiff = ((meanvalresh[:,1] - rat[:,3] ) / ((meanvalresh[:,1] + rat[:,3] )/2) ) * 100
        texturediff = ((meanvalresh[:,2] - rat[:,20] ) / ((meanvalresh[:,2] + rat[:,20] )/2) ) * 100
        brightnessdiff = ((meanvalresh[:,3] - rat[:,21] ) / ((meanvalresh[:,3] + rat[:,21] )/2) ) * 100

        diffattributes = np.vstack((ndvidiff, greennesdiff, texturediff, brightnessdiff)).T
        allatributes = np.concatenate((rat, diffattributes), axis=1)
    
        print("Avarage attributes for neighbours calculated.. all done!")
        
    else:
        print("Using NDVI, Greennes, DSM, texture and vissible brightness")
        meancalc = [2, 3, 9, 20, 21] # use ndvi greennes and dsm

        print("Calculating avarage attributes for per neighbour:")
        for neighbours in tqdm.tqdm(resultarr):
            wantedneigh = rat[np.logical_or.reduce([rat[:,0] == x for x in neighbours])]
            for i in meancalc:
                result.append(np.mean(wantedneigh[0][i]))
                
        # Collect results in array and reshape to attribute table format
        meanvalues = np.asarray(result)
        meanvalresh = meanvalues.reshape(len(regionslist), 5)

        # Calculate differences in percentage
        ndvidiff = ((meanvalresh[:,0] - rat[:,2] ) / ((meanvalresh[:,0] + rat[:,2] )/2) ) * 100 
        greennesdiff = ((meanvalresh[:,1] - rat[:,2] ) / ((meanvalresh[:,1] + rat[:,2] )/2) ) * 100 
        heightdiff = ((meanvalresh[:,2] - rat[:,9] ) / ((meanvalresh[:,2] + rat[:,9] )/2) ) * 100
        texturediff = ((meanvalresh[:,3] - rat[:,20] ) / ((meanvalresh[:,3] + rat[:,20] )/2) ) * 100
        brightnessdiff = ((meanvalresh[:,4] - rat[:,21] ) / ((meanvalresh[:,4] + rat[:,21] )/2) ) * 100

        diffattributes = np.vstack((ndvidiff, greennesdiff, heightdiff, texturediff, brightnessdiff)).T
        allatributes = np.concatenate((rat, diffattributes), axis=1)
    
        print("Avarage attributes for neighbours calculated.. all done!")
        
    return(allatributes, resultarr)