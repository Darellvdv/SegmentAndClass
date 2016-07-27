# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:54:05 2016

@author: darellvdv
"""

def determine_neighbours(label_img):
    """
    DETERMINES NEIGHBOURS OF REGIONS IN NUMPY ARRAY
    
    label_img = image with unique labels
    
    resultarr = list of arrays with neighbours per region
    
    """
    # Determine amount of regions and generate large array
    n = label_img.max()
        
    print("Generating large temp array: check memory!")
       
    tmp = np.zeros((n+1, n+1), bool)

    print("Checking vertical adjacency..")
    # check the vertical adjacency
    a, b = label_img[:-1, :], label_img[1:, :]
    tmp[a[a!=b], b[a!=b]] = True

    print("Checking horizontal adjacency..")
    # check the horizontal adjacency
    a, b = label_img[:, :-1], label_img[:, 1:]
    tmp[a[a!=b], b[a!=b]] = True

    print("Registering adjaceny in all directions.. please monitor memory")
    # register adjacency in both directions (up, down) and (left,right), without copying array
    tmp |= tmp.T

    print("Convert large array to list with neighbours per region..")
    # Convert to list of arrays with neighbours per region
    np.column_stack(np.nonzero(tmp))
    resultarr = [np.flatnonzero(row) for row in tmp[1:]]
    
    print("Cleaning memory and dumping garbage..")
    # Remove large array out of memory and collect garbage
    del(a, b, row, tmp)
    gc.collect()
    
    return resultarr