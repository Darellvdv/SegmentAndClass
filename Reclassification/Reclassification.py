# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:56:22 2016

@author: darellvdv
"""
import numpy as np
import argparse
import tifffile as tiff
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32

import warnings
warnings.filterwarnings("ignore")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--segim", required=True, help="Path to the image")
ap.add_argument("-a", "--table", required=True, help="Path to .npy attributes")
ap.add_argument("-o", "--output", required=True, help="Path to the output folder")

#args = vars(ap.parse_args())
args = ap.parse_args()

print("Here's what we saw on the command line: ")
print("args.image",args.segim)
print("args.bat",args.table)
print("args.output",args.output)

# Set directories
inputpath = args.segim
imagename = inputpath.split("/")
imagenamenoext = imagename[-1].split(".")
imagenamenoext[0]

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

#------------#
# LOAD DATA
#------------#

#np.savetxt('D:/pinapple/output-3-3_ndvi2/_attributes.csv', tile_attributes, delimiter=',')


tile_attributes = np.load(args.table)
seg_rel = tiff.imread(args.segim)


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
bigroad = (tile_attributes[:,3] < 71) & (tile_attributes[:,17] >= 1500) & (tile_attributes[:,4] >= 139) | \
          (tile_attributes[:,3] > 71) & (tile_attributes[:,23] <= 7.4) & (tile_attributes[:,9] < 29) & (tile_attributes[:,2] < 119)
smallroad = (tile_attributes[:,3] < 71) & (tile_attributes[:,17] >= 1500) & (tile_attributes[:,4] <= 139) | \
            (tile_attributes[:,3] < 71) & (tile_attributes[:,17] <= 1500) & (tile_attributes[:,2] < 110) | \
            (tile_attributes[:,3] > 71) & (tile_attributes[:,23] >= 7.4)
allveg = (tile_attributes[:,3] < 71) & (tile_attributes[:,17] <= 1500) & (tile_attributes[:,2] > 110) | \
         (tile_attributes[:,3] > 71) & (tile_attributes[:,23] <= 7.4) & (tile_attributes[:,9] < 29) & (tile_attributes[:,2] > 119) | \
         (tile_attributes[:,3] > 71) & (tile_attributes[:,23] <= 7.4) & (tile_attributes[:,9] > 29)



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
roads = (tile_attributes[:,3] < 71) & (tile_attributes[:,17] >= 1700)
allpinveg = (tile_attributes[:,3] > 71)
#roads = (tile_attributes[:,12] > 1.1) & (tile_attributes[:,19] < 0.96) & (tile_attributes[:,24] > -2.9) | (tile_attributes[:,12] > 1.1) & (tile_attributes[:,19] > 0.96) | (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] >= 1.1) & (tile_attributes[:,18] < 0.52) | (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] >= 1.1) & (tile_attributes[:,18] > 0.52) & (tile_attributes[:,11] >= 59) | (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] <= 1.1) & (tile_attributes[:,23] < 7.9) & (tile_attributes[:,5] <= 119) & (tile_attributes[:,4] >= 71) & (tile_attributes[:,11] > 133)
#allpinveg = (tile_attributes[:,3] >= 73) & (tile_attributes[:,12] >= 0.87) & (tile_attributes[:,14] < 0.6) & (tile_attributes[:,22] < 8.5) & (tile_attributes[:,16] < 3500) | (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] <= 1.1) & (tile_attributes[:,23] < 7.9) & (tile_attributes[:,5] <= 119) & (tile_attributes[:,4] >= 71) & (tile_attributes[:,11] < 133) & (tile_attributes[:,15] < 0.69) | (tile_attributes[:,3] >= 73) & (tile_attributes[:,12] >= 0.87) & (tile_attributes[:,14] < 0.6) & (tile_attributes[:,22] > 8.5) | (tile_attributes[:,3] >= 73) & (tile_attributes[:,12] >= 0.87) & (tile_attributes[:,14] > 0.6) | (tile_attributes[:,3] <= 73) & (tile_attributes[:,12] <= 1.1) & (tile_attributes[:,23] < 7.9) & (tile_attributes[:,5] >= 119)

# Sub classes
#gaps = (roads == True) & (tile_attributes[:,3] <= 160) & (tile_attributes[:,5] <= 180 ) & (tile_attributes[:,11] <= 500) & (tile_attributes[:,15] >= 2)
#deathveg = (allveg == True) & (tile_attributes[:,1] <= 105)
#unhealthveg = (allveg == True) & (tile_attributes[:,1] <= 110)
#healthyveg = (allveg == True) & (tile_attributes[:,1] >= 110)

# Save roadmask
classes = np.zeros((len(tile_attributes), 1))
classes[roads == True] = 1
keys = np.unique(seg_rel)
values = classes #0 for non veg, 1 for veg
dictionary = dict(zip(keys, values))
classified = replace(seg_rel, dictionary)
filename = inputpath
output = (args.output+'/'+imagenamenoext[0]+'_roadmask.tif')
project_raster(filename, output, classified)

# Save vegetation mask
classes = np.zeros((len(tile_attributes), 1))
classes[allpinveg == True] = 1
keys = np.unique(seg_rel)
values = classes #0 for non veg, 1 for veg
dictionary = dict(zip(keys, values))
classified = replace(seg_rel, dictionary)
filename = inputpath
output = (args.output+'/'+imagenamenoext[0]+'_vegetationmask.tif')
project_raster(filename, output, classified)

# Relabel other classes and save output
print("Relabeling image to new classes..")
classes = np.zeros((len(tile_attributes), 1))
classes[bigroad == True] = 1
classes[smallroad == True] = 2
classes[allveg == True] = 3


keys = np.unique(seg_rel)
values = classes #0 for non veg, 1 for veg
dictionary = dict(zip(keys, values))

classified = replace(seg_rel, dictionary)

print("Saving classified image")
filename = inputpath
output = (args.output+'/'+imagenamenoext[0]+'_classified_new.tif')

project_raster(filename, output, classified)
print("All done!")