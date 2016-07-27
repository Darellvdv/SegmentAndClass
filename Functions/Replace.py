# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:54:05 2016

@author: darellvdv
"""

def replace(arr, rep_dict):
    """
    REPLACES VALUES IN nD NUMPY ARRAY IN A VECTORIZED METHOD
    
    Assumes all elements of "arr" are keys of rep_dict
    arr = input array to be changed
    rep_dict = dictionarry containing org values and new values
    
    output = nD array with replaced values
        
    """
    # Removing the explicit "list"
    rep_keys, rep_vals = np.array(list(zip(*sorted(rep_dict.items()))))

    idces = np.digitize(arr, rep_keys, right=True)
    # Notice rep_keys[digitize(arr, rep_keys, right=True)] == arr

    return rep_vals[idces]