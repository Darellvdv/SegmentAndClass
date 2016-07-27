# -*- coding: utf-8 -*-
"""
Created on Fri Jul 08 12:11:35 2016

@author: darellvdv
"""
#----------------#
# IMPORT MODULES #
#----------------#
import gc
import os
import tqdm
import errno
import argparse
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

#------------------#
# IMPORT FUNCTIONS #
#------------------#
functiondir = '{0}\\Functions'.format(os.getcwd())

from Replace import replace
from GetExtent import GetExtent
from ProjectRaster import project_raster
from ClassAndReplace import class_and_replace
from DetermineNeighbours import determine_neighbours
from ReplaceNeighbours import replace_neighbours
from CalcRegionAttributes import calc_region_attributes
from CalcRegionsDifferences import calc_region_differences

#----------------------#
# CONSTRUCT PARSE ARGS
#----------------------#

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-b", "--bat", required=True, help="Path to the OTB .bat file")
ap.add_argument("-o", "--output", required=True, help="Path to the output folder")
ap.add_argument("-s", "--dsm", required=False, help="Path to the dsm")
ap.add_argument("-t", "--dtm", required=False, help="Path to the dtm")
ap.add_argument("-p", "--pre", required=False, help="Preprocessing yes/no")
ap.add_argument("-x", "--att", required=False, help="att = rgb, rgbnir, rgbheight, rgbnirheight")

#args = vars(ap.parse_args())
args = ap.parse_args()

# Set directories
inputpath = args.image
imagename = inputpath.split("/")
imagenamenoext = imagename[-1].split(".")
imagenamenoext[0]

outputpath = args.output
batscriptpath = args.bat

print("Segmentation and classification chain started!")

print("Here's what we saw on the command line: ")
print("args.image",args.image)
print("args.output",args.output)
print("args.bat",args.bat)

# Make sure output path exists, otherwise create it
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
make_sure_path_exists(outputpath)
outputpath = os.chdir(outputpath)

#-----------------------------#
# LOAD FILES AND REMOVE ALPHA
#-----------------------------#

# Load original images
print("Loading image..")
img_org = tiff.imread(inputpath)

# Check for and remove alpha band
#alphabandno=np.shape(img)[2]
if args.att == "rgb":
    print("Removing alpha band of RGB image..")
    call('gdal_translate -b 1 -b 2 -b 3 ' +args.image+' '+args.output+'/'+imagenamenoext[0]+'_noalpha.tif', shell=True)
elif args.att == "rgbnir":
    print("Removing alpha band of RGBNIR image..")
    call('gdal_translate -b 1 -b 4 ' +args.image+' '+args.output+'/'+imagenamenoext[0]+'_noalpha.tif', shell=True)
    
# Remove old image and load new image
#if  alphabandno == 3 or alphabandno == 4 or alphabandno == 5:
print("Deleting org image and loading new image without alpha band..")
    #del(img)
img = tiff.imread(args.output+'/'+imagenamenoext[0]+'_noalpha.tif')
#else:
#    pass

#-----------------------------------#
# LOAD DSM AND DTM AND CLIP TO TILE
#-----------------------------------#
if not args.dsm:
    print("No DSM and DTM selected, moving on to segmentation")
    pass
else:
    print("Loading DSM and DTM..")
    DSM = args.dsm
    DTM = args.dtm
    
    dataSource = gdal.Open(DSM, GA_ReadOnly)
    geotransform = dataSource.GetGeoTransform()
    cols = dataSource.RasterXSize
    rows = dataSource.RasterYSize
    ext = GetExtent(geotransform,cols,rows)

    x1 = str(ext[0][0])
    y1 = str(ext[1][1])
    x2 = str(ext[2][0])
    y2 = str(ext[3][1])

    print("Clipping DSM and DTM to tile..")
    # System call to GDAL to clip DTM and DSM to extent of tile
    driver = gdal.GetDriverByName('GTiff') 
    
    outDs = driver.Create(args.output+'_dsm.tif', cols, rows, 1, GDT_Float32)
    call('gdal_translate -projwin '+str(ext[0][0])+' '+str(ext[3][1])+' '+str(ext[2][0])+' '+str(ext[1][1])+' -of GTiff '+DSM+' '+args.output+'\_dsm.tif')
    
    # set the projection and extent information of the dataset
    outDs.SetProjection(dataSource.GetProjection())
    outDs.SetGeoTransform(dataSource.GetGeoTransform())

    # Flush to save
    outDs.FlushCache()

    outDs = driver.Create(args.output+'_dtm.tif', cols, rows, 1, GDT_Float32)
    call('gdal_translate -projwin '+str(ext[0][0])+' '+str(ext[3][1])+' '+str(ext[2][0])+' '+str(ext[1][1])+' -of GTiff '+DTM+' '+args.output+'\_dtm.tif')

    # set the projection and extent information of the dataset
    outDs.SetProjection(dataSource.GetProjection())
    outDs.SetGeoTransform(dataSource.GetGeoTransform())

    # Flush to save
    outDs.FlushCache()

#------------------------------------#
# EXECUTE BAT SCRIPT AND LOAD OUTPUT
#------------------------------------#

# Execute batch script
if args.pre == 'no':
    print("Skipping pre-processing")
else:
    print("Starting meanshift segementation framework via batch script...")

    tempdir = "D:\tmp"

    # Make sure to set tempdir in bat file!
    p = Popen([args.bat, args.output, (args.output+'/'+imagenamenoext[0]+'_noalpha.tif'), imagenamenoext[0]], shell = True)
    stdout, stderr = p.communicate()

    print("Segmentation done!")

# Load image products from batch file
print("Loading Meanshift segmentation output..")

# Load segmentated image
seg = tiff.imread(args.output+'/'+imagenamenoext[0]+'_segmentation_merged.tif')
print("Segmented image loaded!")

 
#-----------------#   
# START OF SCRIPT 
#-----------------#
 
# Define regions of org label image
print("Creating list of new regions and re-label segmented image..")

regionslist = np.unique(seg) # UPDATE REGIONSLIST
# Construct list of new values from 1 to n
newvals = np.asarray(list(range(1,(len(regionslist) + 1))))
# Convert to dictionary to increase processing speed
keys = regionslist
values = newvals
dictionary = dict(zip(keys, values))
# Apply replace function to relabel image in a vectorized way
seg_rel = replace(seg, dictionary)

# Remove org seg
print("Deleting old segmented image..")
del(seg)

# CALCULATE RASTER ATTRIBUTES TO NUMPY ARRAY
print("Starting calc region attributes function")
attributes1 = calc_region_attributes(img_org, seg_rel)

# Delete garbage from memory
print("Deleting garbage from memory")

if not args.dsm:
    pass
else:
    del(dsm)
    del(dtm)

gc.collect() # remove garbage and continue to next image

# Calculate regeion differences (memory intensive!)
print("Starting calc neighbour attributes function, check memory!")
tile_attributes, resultarr = calc_region_differences(seg_rel, attributes1)

#--------------------------------#
# SAVE OUTPUT AND MERGE WITH SHP
#--------------------------------#
np.savetxt(args.output+'/'+imagenamenoext[0]+'_attributes.csv', tile_attributes, delimiter=',')
np.save(args.output+'/'+imagenamenoext[0]+'_attributes.npy', tile_attributes)

# Set directories
apppath = (os.path.dirname(args.bat)+'/Reclassification/MergeAttributes.R')
# Call Rscript for merging
print("Calling R script for attribute merging")
shapein = (args.output+'/'+imagenamenoext[0]+'_segmentation_merged.shp')
csvin = (args.output+'/'+imagenamenoext[0]+'_attributes.csv')
imagein = (imagenamenoext[0]+'_segmentation_merged')

argsR = [shapein, csvin, imagein]
command = 'Rscript'
cmd = [command, apppath] + argsR
x = call(cmd, universal_newlines=True)

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

#-----------------------#
print("Implementing classification rules..")
# First class all BigRoad and all Vegetation:
bigroad = (tile_attributes[:,3] < 71) & (tile_attributes[:,17] >= 1500) & (tile_attributes[:,4] >= 139) | \
          (tile_attributes[:,3] > 71) & (tile_attributes[:,23] <= 7.4) & (tile_attributes[:,9] < 29) & (tile_attributes[:,2] < 119)

allveg = (tile_attributes[:,3] < 71) & (tile_attributes[:,17] <= 1500) & (tile_attributes[:,2] > 110) | \
         (tile_attributes[:,3] > 71) & (tile_attributes[:,23] <= 7.4) & (tile_attributes[:,9] < 29) & (tile_attributes[:,2] > 119) | \
         (tile_attributes[:,3] > 71) & (tile_attributes[:,23] <= 7.4) & (tile_attributes[:,9] > 29)
         
# Save roadvegmask
classes_roadveg = np.zeros((len(tile_attributes), 1)).astype(int) # Length of all atributes from first segmentation
classes_roadveg[bigroad != True] = 1 
classes_roadveg[allveg != True] = 1 
classes_roadveg[bigroad == True] = 2
classes_roadveg[allveg == True] = 3

# Merge and reclassify
roadveg, roadveg_ID = class_and_replace(seg_rel, classes_roadveg)

# Determine neightbours, replace unique values with classified values and calculate percentages
roadveg_neighbours = determine_neighbours(roadveg_ID)
roadveg_replaced_neighbours, roadveg_table = replace_neighbours(classes_roadveg, roadveg, roadveg_ID, roadveg_neighbours)

# Construct attributes for merged regions
regionslist = np.unique(roadveg_ID) # FIRST UPDATE REGIONSLIST!
roadveg_tile_attributes = calc_region_attributes(img_org, roadveg_ID)
roadveg_tile_attributes = np.hstack((roadveg_tile_attributes, roadveg_table))

# Vegetation on road:
missedroad = (roadveg_tile_attributes[:,25] == 1) & (roadveg_tile_attributes[:,23] >= 60) & (roadveg_tile_attributes[:,3] < 71) | \
             (roadveg_tile_attributes[:,25] == 3) & (roadveg_tile_attributes[:,23] >= 75.0)
             
classes_missedroad = np.zeros((len(roadveg_tile_attributes), 1)).astype(int)
classes_missedroad[missedroad != True] = 0
classes_missedroad[missedroad == True] = 1

# Reclassify
missedroad, missedroad_ID = class_and_replace(roadveg_ID, classes_missedroad)

# Save roadvegmask
filename = inputpath
output = (args.output+'/'+imagenamenoext[0]+'_roadvegmask.tif')
project_raster(filename, output, roadveg)

# Save missedroad
filename = inputpath
output = (args.output+'/'+imagenamenoext[0]+'_missedroad.tif')
project_raster(filename, output, missedroad)

#print("Saving classified image")
#filename = inputpath
#output = (args.output+'/'+imagenamenoext[0]+'_classified.tif')
#project_raster(filename, output, classified_roadveg_rep)

print("All done!")