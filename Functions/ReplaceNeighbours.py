# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:54:05 2016

@author: darellvdv
"""

def replace_neighbours(classes, classified, classified_ID, org_neighbours):
    """
    VECTORIZED METHOD TO REPLACE VALUES IN A LIST OF ARRAYS AND CALCULATE PERCENTAGES
    
    classes = array of same length as len of org_label with new labels
    classified = numpy array with new classified values
    classified_ID = merged regions with a unique ID
    org_neighbours = output of calc_neighbours function
    
    replaced = list of arrays with replaced neighbours per region
    tablemerged = array indicating the percentages of neighbours for each region. First N columns are
                  the unique amount of classes. Last column is the classified value of that region  
    
    """
    # Set up dictionary for replacement of unique neighbour values to classified neighbour values
    listv = np.unique(classes).tolist()
    keys = []
    values = []
    for i in listv:
        keys.append(classified_ID[classified == i])
        leng = classified_ID[classified == i]
        values.append([i] * len(leng))
    
    keys = np.concatenate(keys).tolist()
    values = np.concatenate(values).tolist()

    alldict = dict(zip(keys, values))

    # Replace list of unique neightbours with classified neighbours
    # First get keys of final dict
    keys_clas = alldict.keys()
    values_clas = alldict.values()

    import numpy_indexed as npi
    arr = np.concatenate(org_neighbours)
    idx = npi.indices(keys_clas, arr, missing='mask')
    remap = np.logical_not(idx.mask)
    arr[remap] = np.array(values_clas)[idx[remap]]
    replaced = np.array_split(arr, np.cumsum([len(a) for a in org_neighbours][:-1]))

    # Determine classification % of neighbours and generate array for attribute table
    idx = [np.ones(len(a))*i for i, a in enumerate(replaced)]
    (rows, cols), table = npi.count_table(np.concatenate(idx), np.concatenate(replaced))
    table = table.astype(float)
    table = table / table.sum(axis=1, keepdims=True) * 100
    
    # Build classification table
    values_final = np.asarray(values_clas)
    tablemerged = np.column_stack((table, values_final))
    
    return replaced, tablemerged