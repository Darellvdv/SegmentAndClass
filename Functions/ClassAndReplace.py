# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:54:05 2016

@author: darellvdv
"""

def class_and_replace(org_label, classes):
    """
    CLASSIFIES AND REPLACES VALUES IN NUMPY ARRAY
    
    org_label = numpy array with original labels
    classes = array of same length as len of org_label with new labels
    
    classified = numpy array with new classified values
    classified_ID = merged regions with a unique ID
    
    """
    # Classify and replace
    keys = np.unique(org_label)
    values = classes #0 for non veg, 1 for veg
    dictionary = dict(zip(keys, values))
    classified = replace(org_label, dictionary)
    
    # Label merged regions with unique ID
    labeld = label(classified)
    number_of_labels = np.unique(labeld)
    newvals = np.asarray(list(range(1,(len(number_of_labels) + 1))))
    keys_ID = number_of_labels
    values_ID = newvals
    dictionary_ID = dict(zip(keys_ID, values_ID))
    classified_ID = replace(labeld, dictionary_ID)
    
    del(labeld)
    
    return classified, classified_ID