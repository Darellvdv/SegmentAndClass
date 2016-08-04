# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:56:22 2016

@author: darellvdv
"""
import gc, os, tqdm, errno, argparse
import numpy as np
import tifffile as tiff
from osgeo import gdal
from scipy import ndimage
from subprocess import call
from subprocess import Popen
from skimage.measure import label, regionprops
from scipy.ndimage.filters import gaussian_filter
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32

import warnings
warnings.filterwarnings("ignore")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to original image")
ap.add_argument("-s", "--segim", required=True, help="Path to segmented image")
ap.add_argument("-a", "--table", required=True, help="Path to .npy attributes")
ap.add_argument("-o", "--output", required=True, help="Path to the output folder")
ap.add_argument("-x", "--att", required=True, help="att = rgb, rgbnir, rgbheight, rgbnirheight")

#args = vars(ap.parse_args())
args = ap.parse_args()

print("Here's what we saw on the command line: ")
print("args.image",args.image)
print("args.segim",args.segim)
print("args.bat",args.table)
print("args.output",args.output)

# Set directories
inputpath = args.image
imagename = inputpath.split("/")
imagenamenoext = imagename[-1].split(".")
imagenamenoext[0]

outputpath = args.output

#---------------------#
# LOAD MAIN FUNCTIONS
#---------------------#

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
    
def project_raster(filename, output, dataset):
    """
    PROJECTS NUMPY ARRAY TO REFERENCE PROJECTION AND SAVES TO DISK
    
    filename = org filename as string
    output = output filename as string
    dataset = dataset to be projected (numpy array)
    
    """
    
    # Save classified image
    dataSource = gdal.Open(filename, GA_ReadOnly)
    dataSource.GetDriver().LongName
    geotransform = dataSource.GetGeoTransform()

    # Write the result to disk
    driver = gdal.GetDriverByName('GTiff')
    outDataSet = driver.Create(output, dataSource.RasterXSize, dataSource.RasterYSize, 1, GDT_Float32)
    outBand = outDataSet.GetRasterBand(1)
    outBand.WriteArray(dataset,0,0)
    outBand.SetNoDataValue(float('NaN'))

    # set the projection and extent information of the dataset
    outDataSet.SetProjection(dataSource.GetProjection())
    outDataSet.SetGeoTransform(dataSource.GetGeoTransform())

    # Flush to save
    outBand.FlushCache()
    outDataSet.FlushCache()
    
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
    if os.path.exists(args.output+'/'+imagenamenoext[0]+'_texture.tif') == True:
        texture = tiff.imread(args.output+'/'+imagenamenoext[0]+'_texture.tif')
    else:
        call('gdaldem roughness '+args.image+' '+args.output+'/'+imagenamenoext[0]+'_texture.tif -of GTiff -b 1')
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
    
# Set up classification functions:
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
    
def current_clas(classes, classified, classified_ID):
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
    values_clas = alldict.values()
    
    return(values_clas)

#------------#
# LOAD DATA
#------------#
tile_attributes = np.load(args.table)
seg = tiff.imread(args.segim)
org_img = tiff.imread(args.image)


# Prepare segmented image

regionslist = np.unique(seg) # UPDATE REGIONSLIST
# Construct list of new values from 1 to n
newvals = np.asarray(list(range(1,(len(regionslist) + 1))))
# Convert to dictionary to increase processing speed
keys = regionslist
values = newvals
dictionary = dict(zip(keys, values))
# Apply replace function to relabel image in a vectorized way
seg_rel = replace(seg, dictionary)

#--------------------------#
# START OF CLASSIFICATION
#--------------------------#

# 0 = New index value
# 1 = Old index value

# NDVI and greennes
# 2 = NDVI 3 = greennes

# Band values and ratios:
# 4 = Rmean, 5 = Gmean, 6 = Bmean, 7 = NIRmean
# 8 = Rvar, 9 = Gvar, 10 = Bvar, 11 = NIRvar
# 12 = R/G, 13 = G/R, 14 = B/NIR, 15 = G/NIR

# Height values:
# (16) = dsm, (17) = dtm, (18) = csm

# Regionprops:
# 16 (19) = perimeter, 17 (20) = area, 18 (21) = extent, 19 (22) = eccentricity, 20 (23) = roughness, 21 (24) = vissible brightness

# Neighbour props:
#  = distance to closest neighbour, 22 = NDVIdiff, 23 = Greennesdiff, 24 = Texturediff, 25 = Brightnessdiff
# Convert to dictionary to increase processing speed

print("Implementing classification rules..")

#-----------------------#
# BIG ROADS #
#-----------------------#

bigroad = (tile_attributes[:,3] < 71) & (tile_attributes[:,17] >= 1500) & (tile_attributes[:,4] >= 139) | \
          (tile_attributes[:,3] > 71) & (tile_attributes[:,23] <= 7.4) & (tile_attributes[:,9] < 29) & (tile_attributes[:,2] < 119)

# Bigroad:
classes_bigroad = np.zeros((len(tile_attributes), 1)).astype(int)
classes_bigroad[bigroad == True] = 2
classes_bigroad[bigroad != True] = 1

# Replace and merge
bigroad, bigroad_ID = class_and_replace(seg_rel, classes_bigroad)

# Determine neightbours, replace unique values with classified values and calculate percentages
bigroad_neighbours = determine_neighbours(bigroad_ID)
bigroad_replaced_neighbours, bigroad_table = replace_neighbours(classes_bigroad, bigroad, bigroad_ID, bigroad_neighbours)

# Construct attributes for merged roads
regionslist = np.unique(bigroad_ID) # FIRST UPDATE REGIONSLIST!
bigroad_tile_attributes = calc_region_attributes(org_img, bigroad_ID)
# Merge attributes with classified values and neighbours
bigroad_tile_attributes = np.column_stack((bigroad_tile_attributes, bigroad_table))

# Reclassify bigroads to new region attributes (less roughness and other band ratios)
bigroad_new = (bigroad_tile_attributes[:,-1] == 2) & (bigroad_tile_attributes[:,8] >= 500) & (bigroad_tile_attributes[:,20] < 7) & (bigroad_tile_attributes[:,17] >= 1200) | \
(bigroad_tile_attributes[:,-1] == 2) & (bigroad_tile_attributes[:,9] <= 310) & (bigroad_tile_attributes[:,20] < 7) & (bigroad_tile_attributes[:,17] >= 1200)

classes_bigroad_new = np.zeros((len(bigroad_tile_attributes), 1)).astype(int)
classes_bigroad_new[bigroad_new == True] = 2
classes_bigroad_new[bigroad_new != True] = 1

bigroad_new, bigroad_ID_new = class_and_replace(bigroad_ID, classes_bigroad_new)


# Save roadmask
filename = inputpath
output = (args.output+'/'+imagenamenoext[0]+'_roadmask.tif')
project_raster(filename, output, bigroad_new)


#------------------------#
# ALL VEGETATION & MASKS #
#------------------------#

# Classify all vegetation based on small segments
allvegc = (tile_attributes[:,3] < 71) & (tile_attributes[:,17] <= 1500) & (tile_attributes[:,2] > 110) | \
         (tile_attributes[:,3] > 71) & (tile_attributes[:,23] <= 7.4) & (tile_attributes[:,9] < 29) & (tile_attributes[:,2] > 119) | \
         (tile_attributes[:,3] > 71) & (tile_attributes[:,23] <= 7.4) & (tile_attributes[:,9] > 29)
         
# Replace original segments
classes_allveg = np.zeros((len(tile_attributes), 1)).astype(int)
classes_allveg[allvegc == True] = 2
classes_allveg[allvegc != True] = 1

# Replace and merge
allveg, allveg_ID = class_and_replace(seg_rel, classes_allveg)

# Save all vegetation in raster
filename = inputpath
output = (args.output+'/'+imagenamenoext[0]+'_vegetationall.tif')
project_raster(filename, output, allveg)

## CREATE MASKS ##

# Vegetation located on road
# Determine neightbours, replace unique values with classified values and calculate percentages
allveg_neighbours = determine_neighbours(allveg_ID)
allveg_replaced_neighbours, allveg_table = replace_neighbours(classes_allveg, allveg, allveg_ID, allveg_neighbours)

# Construct attributes for Vegetation
regionslist = np.unique(allveg_ID) # FIRST UPDATE REGIONSLIST!
allveg_tile_attributes = calc_region_attributes(org_img, allveg_ID)
# Merge attributes with classified values and neighbours
allveg_tile_attributes = np.column_stack((allveg_tile_attributes, allveg_table))

# Classify vegetation that is located on road
roadvegc = (allveg_tile_attributes[:,-1] == 2) & (allveg_tile_attributes[:,22] >= 90) & (allveg_tile_attributes[:,17] <= 2500)

# Create mask for vegetation on road
classes_roadveg = np.zeros((len(allveg_tile_attributes), 1)).astype(int)
classes_roadveg[roadvegc == True] = 2
classes_roadveg[roadvegc != True] = 1

roadvegmask, roadvegmask_ID = class_and_replace(allveg_ID, classes_roadveg)

#-----------------------#
#   UNHEALTHY VEG       #
#-----------------------#

# Where vegetation = true, filter out unhealthy veg

unhealthyvegc = (allvegc == True) & (tile_attributes[:,2] <= 116) & (tile_attributes[:,3] <= 81.5) & (tile_attributes[:,5] <= 250) 

# Replace seg_rel
classes_unhveg = np.zeros((len(tile_attributes), 1)).astype(int)
classes_unhveg[unhealthyvegc  == True] = 2
classes_unhveg[unhealthyvegc  != True] = 1

# Replace and merge
unhealthyveg, unhealthyveg_ID = class_and_replace(seg_rel, classes_unhveg)


#------------------------------#
# CONSTRUCT CLASSIFIED IMAGES  #
#------------------------------#

# Construct the classified vegetation map SEQUENCE IS IMPORTANT!
# 1 = unclassified, 2 = vegetation, 3 = unhealthy vegetation
classveg = np.copy(allveg)
classveg[allveg == 2] = 2
classveg[unhealthyveg == 2] = 3
classveg[roadvegmask == 2] = 1


# Save classified vegetation in raster
filename = inputpath
output = (args.output+'/'+imagenamenoext[0]+'_vegetationclass.tif')
project_raster(filename, output, classveg)


#--------------#
# DETECT GAPS  #
#--------------#

# Classify unclassified that is surrounded by vegetation
gapc = (allveg_tile_attributes[:,-1] == 1) & (allveg_tile_attributes[:,23] >= 90) & (allveg_tile_attributes[:,17] <= 7500)

# Create mask for vegetation on road
classes_gaps = np.zeros((len(allveg_tile_attributes), 1)).astype(int)
classes_gaps[gapc == True] = 2
classes_gaps[gapc != True] = 1

gaps, gaps_ID = class_and_replace(allveg_ID, classes_gaps)

# Save gaps in raster
filename = inputpath
output = (args.output+'/'+imagenamenoext[0]+'_gaps.tif')
project_raster(filename, output, gaps)


print("All done!")

#pinveg = (tile_attributes[:,3] >= 73) & (tile_attributes[:,12] >= 0.87) & (tile_attributes[:,14] < 0.6) & (tile_attributes[:,22] < 8.5) & (tile_attributes[:,16] < 3500) | (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] <= 1.1) & (tile_attributes[:,23] < 7.9) & (tile_attributes[:,5] <= 119) & (tile_attributes[:,4] >= 71) & (tile_attributes[:,11] < 133) & (tile_attributes[:,15] < 0.69)  
#unhealthyveg = (tile_attributes[:,3] >= 73) & (tile_attributes[:,12] >= 0.87) & (tile_attributes[:,14] < 0.6) & (tile_attributes[:,22] > 8.5) | (tile_attributes[:,3] >= 73) & (tile_attributes[:,12] >= 0.87) & (tile_attributes[:,14] > 0.6) | (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] <= 1.1) & (tile_attributes[:,23] < 7.9) & (tile_attributes[:,5] >= 119)
#road =  (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] >= 1.1) & (tile_attributes[:,18] < 0.52) | (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] >= 1.1) & (tile_attributes[:,18] > 0.52) & (tile_attributes[:,11] >= 59) | (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] <= 1.1) & (tile_attributes[:,23] < 7.9) & (tile_attributes[:,5] <= 119) & (tile_attributes[:,4] >= 71) & (tile_attributes[:,11] > 133)
#gap = (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] >= 1.1) & (tile_attributes[:,18] > 0.52) & (tile_attributes[:,11] <= 59) | (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] <= 1.1) & (tile_attributes[:,23] < 7.9) & (tile_attributes[:,5] <= 119) & (tile_attributes[:,4] >= 71) & (tile_attributes[:,11] < 133) & (tile_attributes[:,15] > 0.69) | (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] <= 1.1) & (tile_attributes[:,23] > 7.9)
#plastic = (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] <= 1.1) & (tile_attributes[:,23] < 7.9) & (tile_attributes[:,5] <= 119) & (tile_attributes[:,4] <= 71) 
#otherveg = (tile_attributes[:,3] >= 73) & (tile_attributes[:,12] >= 0.87) & (tile_attributes[:,14] < 0.6) & (tile_attributes[:,22] < 8.5) & (tile_attributes[:,16] > 3500) | (tile_attributes[:,3] >= 73) & (tile_attributes[:,12] <= 0.87)


#pinveg = (tile_attributes[:,2] >= 118) & (tile_attributes[:,12] >= 0.86) & (tile_attributes[:,23] < 2.9)
#otherveg = (tile_attributes[:,2] >= 118) & (tile_attributes[:,12] <= 0.86)
#bigroad = (tile_attributes[:,2] <= 118) & (tile_attributes[:,3] < 71) & (tile_attributes[:,20] < 12) & (tile_attributes[:,13] < 0.97)
#smallroad = (tile_attributes[:,2] <= 118) & (tile_attributes[:,3] < 71) & (tile_attributes[:,20] > 12) & (tile_attributes[:,16] > 181) & (tile_attributes[:,3] >= 63) | (tile_attributes[:,2] <= 118) & (tile_attributes[:,3] > 71) & (tile_attributes[:,23] >= 4.9)
#gap = (tile_attributes[:,2] >= 118) & (tile_attributes[:,12] >= 0.86) & (tile_attributes[:,23] > 2.9) | (tile_attributes[:,2] <= 118) & (tile_attributes[:,3] < 71) & (tile_attributes[:,20] > 12) & (tile_attributes[:,16] < 181) 
#unhealthyveg = (tile_attributes[:,2] <= 118) & (tile_attributes[:,3] > 71) & (tile_attributes[:,23] <= 4.9)
#urban = (tile_attributes[:,2] <= 118) & (tile_attributes[:,3] < 71) & (tile_attributes[:,20] < 12) & (tile_attributes[:,13] > 0.97) | (tile_attributes[:,2] <= 118) & (tile_attributes[:,3] < 71) & (tile_attributes[:,20] > 12) & (tile_attributes[:,16] > 181) & (tile_attributes[:,3] <= 63) 


#pinveg = (tile_attributes[:,2] >= 113) & (tile_attributes[:,3] < 81)
#otherveg = (tile_attributes[:,2] >= 113) & (tile_attributes[:,3] > 81)
#bigroad = (tile_attributes[:,2] <= 113) & (tile_attributes[:,3] < 71) & (tile_attributes[:,17] >= 1944)
#smallroad = (tile_attributes[:,2] <= 113) & (tile_attributes[:,3] < 71) & (tile_attributes[:,17] <= 1944) & (tile_attributes[:,19] >= 0.9) 
#gap = (tile_attributes[:,2] <= 113) & (tile_attributes[:,3] < 71) & (tile_attributes[:,17] <= 1944) & (tile_attributes[:,19] <= 0.9)
#unhealthyveg = (tile_attributes[:,2] <= 113) & (tile_attributes[:,3] > 71)


# Main classes
#roads = (tile_attributes[:,3] < 71) & (tile_attributes[:,17] >= 1700)
#allpinveg = (tile_attributes[:,3] > 71)
#roads = (tile_attributes[:,12] > 1.1) & (tile_attributes[:,19] < 0.96) & (tile_attributes[:,24] > -2.9) | (tile_attributes[:,12] > 1.1) & (tile_attributes[:,19] > 0.96) | (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] >= 1.1) & (tile_attributes[:,18] < 0.52) | (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] >= 1.1) & (tile_attributes[:,18] > 0.52) & (tile_attributes[:,11] >= 59) | (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] <= 1.1) & (tile_attributes[:,23] < 7.9) & (tile_attributes[:,5] <= 119) & (tile_attributes[:,4] >= 71) & (tile_attributes[:,11] > 133)
#allpinveg = (tile_attributes[:,3] >= 73) & (tile_attributes[:,12] >= 0.87) & (tile_attributes[:,14] < 0.6) & (tile_attributes[:,22] < 8.5) & (tile_attributes[:,16] < 3500) | (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] <= 1.1) & (tile_attributes[:,23] < 7.9) & (tile_attributes[:,5] <= 119) & (tile_attributes[:,4] >= 71) & (tile_attributes[:,11] < 133) & (tile_attributes[:,15] < 0.69) | (tile_attributes[:,3] >= 73) & (tile_attributes[:,12] >= 0.87) & (tile_attributes[:,14] < 0.6) & (tile_attributes[:,22] > 8.5) | (tile_attributes[:,3] >= 73) & (tile_attributes[:,12] >= 0.87) & (tile_attributes[:,14] > 0.6) | (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] <= 1.1) & (tile_attributes[:,23] < 7.9) & (tile_attributes[:,5] >= 119)

# Sub classes
#gaps = (roads == True) & (tile_attributes[:,3] <= 160) & (tile_attributes[:,5] <= 180 ) & (tile_attributes[:,11] <= 500) & (tile_attributes[:,15] >= 2)
#deathveg = (allveg == True) & (tile_attributes[:,1] <= 105)
#unhealthveg = (allveg == True) & (tile_attributes[:,1] <= 110)
#healthyveg = (allveg == True) & (tile_attributes[:,1] >= 110)

