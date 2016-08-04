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

# --------------------- #
# CONSTRUCT PARSE ARGS
# --------------------- #

# construct the argument parser and collect the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-b", "--bat", required=True, help="Path to the OTB .bat file")
ap.add_argument("-o", "--output", required=True, help="Path to the output folder")
ap.add_argument("-s", "--dsm", required=False, help="Path to the dsm")
ap.add_argument("-t", "--dtm", required=False, help="Path to the dtm")
ap.add_argument("-p", "--pre", required=False, help="Preprocessing yes/no")
ap.add_argument("-x", "--att", required=False, help="att = rgb, rgbnir, rgbheight, rgbnirheight")
ap.add_argument("-r", "--rang", required=False, help="Euclidean spectral range")
ap.add_argument("-a", "--spatialr", required=False, help="Connected component spatial range")
ap.add_argument("-m", "--minsiz", required=False, help="Minimal size of segments")
ap.add_argument("-z", "--tilesize", required=False, help="Tilesize for processing")

args = ap.parse_args()

# Set directories and extract filename
inputpath = args.image
imagename = inputpath.split("/")
imagenamenoext = imagename[-1].split(".")
imagenamenoext[0]

outputpath = args.output
batscriptpath = args.bat

# Print settings input
print("Segmentation and classification chain started!")

print("Here's what we saw on the command line: ")
print("The following tile will be processed: {0}".format(args.image))
print("Images will be processed as: {0}".format(args.att))
print("The output will be saved in: {0}".format(args.output))
if args.pre == 'yes':
    print("The spectral euclidean distance is {0} and the connected component spatial range is "
          "{1}".format(args.rang, args.spatialr))
    print("The minimum object area is {0} and tiles will be processed at {1} x {1}".format(args.minsiz, args.tilesize))


# Make sure output path exists, otherwise create it
def make_sure_path_exists(path):

    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

make_sure_path_exists(outputpath)

# ---------------------------- #
# LOAD FILES AND REMOVE ALPHA
# ---------------------------- #

# Load original images
print("Loading image..")
org_img = tiff.imread(inputpath)

# Check for and remove alpha band
# alphabandno=np.shape(img)[2]
# if args.att == "rgb":
#    print("Removing alpha band of RGB image..")
#    call('gdal_translate -b 1 -b 2 -b 3 ' +args.image+' '+args.output+'/'+imagenamenoext[0]+'_noalpha.tif', shell=True)
# elif args.att == "rgbnir":
#    print("Removing alpha band of RGBNIR image..")
#    call('gdal_translate -b 1 -b 2 -b 3 -b 4 ' +args.image+' '+args.output+'/'+imagenamenoext[0]+'_noalpha.tif', shell=True)
    
# Remove old image and load new image
# if  alphabandno == 3 or alphabandno == 4 or alphabandno == 5:
# print("Deleting org image and loading new image without alpha band..")
    #del(img)
# img = tiff.imread(args.output+'/'+imagenamenoext[0]+'_noalpha.tif')
# else:
#    pass

# ---------------------------------- #
# LOAD DSM AND DTM AND CLIP TO TILE
# ---------------------------------- #
if not args.dsm:
    print("No DSM and DTM selected, moving on to segmentation")
    pass
else:
    print("Loading DSM and DTM..")
    DSM = args.dsm
    DTM = args.dtm
    
    # Get extent of clipping area
    def GetExtent(gt, cols, rows):
        """ Return list of corner coordinates from a geotransform
            gt: geotransform
            cols: number of columns in the dataset
            rows: number of rows in the dataset

            return:   coordinates of each corner
            """
        ext = []
        xarr = [0, cols]
        yarr = [0, rows]
            
        for px in xarr:
            for py in yarr:
                x=gt[0]+(px*gt[1])+(py*gt[2])
                y=gt[3]+(px*gt[4])+(py*gt[5])
                ext.append([x,y])
                print x,y
            yarr.reverse()
        return ext


    dataSource = gdal.Open(inputpath, GA_ReadOnly)
    geotransform = dataSource.GetGeoTransform()
    cols = dataSource.RasterXSize
    rows = dataSource.RasterYSize
    ext = GetExtent(geotransform, cols, rows)

    x1 = str(ext[0][0])
    y1 = str(ext[1][1])
    x2 = str(ext[2][0])
    y2 = str(ext[3][1])

    print("Clipping DSM and DTM to tile..")
    # System call to GDAL to clip DTM and DSM to extent of tile
    driver = gdal.GetDriverByName('GTiff')

    #outDs = driver.Create(args.output + '/' + imagenamenoext[0] + '_dsm.tif', cols, rows, 1, GDT_Float32)
    call('gdal_translate -projwin '+str(ext[0][0])+' '+str(ext[3][1])+' '+str(ext[2][0])+' '+str(ext[1][1])+
         ' -of GTiff '+DSM+' '+args.output+'/'+imagenamenoext[0]+'_dsm.tif')

    #outDs = driver.Create(args.output + '/' + imagenamenoext[0] + '_dtm.tif', cols, rows, 1, GDT_Float32)
    call('gdal_translate -projwin '+str(ext[0][0])+' '+str(ext[3][1])+' '+str(ext[2][0])+' '+str(ext[1][1])+
         ' -of GTiff '+DTM+' '+args.output+'/'+imagenamenoext[0]+'_dtm.tif')

# ----------------------------------- #
# EXECUTE BAT SCRIPT AND LOAD OUTPUT
# ----------------------------------- #

# Execute batch script
if args.pre == 'no':
    print("Skipping pre-processing")
else:
    print("Starting meanshift segementation framework via batch script...")

    # Call batch script with parsed arguments
    p = Popen([args.bat, args.output, args.image, imagenamenoext[0], args.rang, args.spatialr, args.minsiz,
               args.tilesize], shell=True)
    stdout, stderr = p.communicate()

    print("Segmentation done!")

# Load image products from batch file
print("Loading Meanshift segmentation output..")

# Load segmentated image
seg = tiff.imread(args.output+'/'+imagenamenoext[0]+'_segmentation_merged.tif')
print("Segmented image loaded!")

# ---------------------- #
# DEFINE MAIN FUNCTIONS 
# ---------------------- #


def calc_region_attributes(org_img, label_img, dsm=None, dtm=None):
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
    
    if args.att == 'rgb' or 'rgbheight':
        print("Calculating Greennes..")
        grvi = 200*(org_img[:, :, 1]*100.0) / (org_img[:, :, 1]*100.0 + org_img[:, :, 0]*100.0 + org_img[:, :, 2]*100.0)
        mean_grvi = ndimage.mean(grvi, labels=label_img, index=index)
        
    if args.att == 'rgbnir' or 'rgbnirheight':
        print("Calculating NDVI and Greennes..")
        grvi = 200*(org_img[:, :, 1]*100.0) / (org_img[:, :, 1]*100.0 + org_img[:, :, 0]*100.0 + org_img[:, :, 2]*100.0)
        mean_grvi = ndimage.mean(grvi, labels=label_img, index=index)
    
        ndvi = 100*(org_img[:, :, 3]*100.0 - org_img[:, :, 0]*100.0) / (org_img[:, :, 3]*100.0 + org_img[:, :, 0]*100.0) + 100
        mean_ndvi = ndimage.mean(ndvi, labels=label_img, index=index)

    # Calculate mean for all bands and heights and append to array
    if args.att == 'rgb' or 'rgbheight':
        w, h = 3, len(index)
    if args.att == 'rgbnir' or 'rgbheight':
        w, h = 4, len(index)
    meanbands = np.zeros((h, w))
    varbands = np.zeros((h, w))
        
    if args.att == 'rgb':
        print("Averaging spectral RGB bands:")
        for i in tqdm.tqdm(xrange(0, 3)):
            meanbands[:, i] = ndimage.mean(org_img[:, :, i].astype(float), labels=label_img, index=index)
            varbands[:, i] = ndimage.variance(org_img[:, :, i].astype(float), labels=label_img, index=index)
    else:
        print("Averaging spectral RGBNIR bands:")
        for i in tqdm.tqdm(xrange(0, 4)):
            meanbands[:, i] = ndimage.mean(org_img[:, :, i].astype(float), labels=label_img, index=index)
            varbands[:, i] = ndimage.variance(org_img[:, :, i].astype(float), labels=label_img, index=index)
            
    print("Calculating ratios:")
    # Red / Green
    redgreen = meanbands[:, 0] / meanbands[:, 1]
    # Green / Red
    greenred = meanbands[:, 1] / meanbands[:, 0]
    # Blue / NIR
    bluenir = meanbands[:, 2] / meanbands[:, 3]
    # Green / NIR
    greennir = meanbands[:, 1] / meanbands[:, 3]

    ratios = np.vstack((redgreen, greenred, bluenir, greennir)).T
            
    print("Calculating roughness..")
    if os.path.exists(args.output+'/'+imagenamenoext[0]+'_texture.tif'):
        texture = tiff.imread(args.output+'/'+imagenamenoext[0]+'_texture.tif')
    else:
        call('gdaldem roughness '+args.image+' '+args.output+'/'+imagenamenoext[0]+'_texture.tif -of GTiff -b 1')
        texture = tiff.imread(args.output+'/'+imagenamenoext[0]+'_texture.tif')
    blurred = gaussian_filter(texture, sigma=7)  # apply gaussian blur for smoothing
    meantexture = np.zeros((len(index), 1))
    meantexture[:, 0] = ndimage.mean(blurred.astype(float), labels=label_img, index=index)
    
    print("Calculating vissible brightness..")
    vissiblebright = (org_img[:, :, 0] + org_img[:, :, 1] + org_img[:, :, 2]) / 3
    meanbrightness = np.zeros((len(index), 1))
    meanbrightness[:, 0] = ndimage.mean(vissiblebright.astype(float), labels=label_img, index=index)
    
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
    if args.dsm != None:
        # Calculate crop surface model
        print("Calculating crop surface model..")
        dsmload = tiff.imread(args.output+'/'+imagenamenoext[0]+'_dsm.tif')
        print("loaded 1")
        dtmload = tiff.imread(args.output+'/'+imagenamenoext[0]+'_dtm.tif')
        print("loaded 2")
        csm = dsmload.astype(float) - dtmload.astype(float)
        heights = np.dstack((dsmload, dtmload, csm))
    
        # Resample heights
        print("Resampling heights..")
        heights_res = ndimage.zoom(heights, (2, 2, 1))
        print("height resampled!")
        w, h = 3, len(index)
        meanheight = np.zeros((h, w))
        for i in tqdm.tqdm(xrange(0, 2)):
            meanheight[:, i] = ndimage.mean(heights_res[:, :, i].astype(float), labels=label_img, index=index)
        
        data = np.concatenate((meanbands, varbands, ratios, meanheight, regionprops_calc, meantexture, meanbrightness), axis=1)
        if args.att == 'rgbheight':
            data2 = np.column_stack((index.astype(int), regionslist, mean_grvi, data))
        else:
            data2 = np.column_stack((index.astype(int), regionslist, mean_ndvi, mean_grvi, data))
        print("Data collected, attributes calculated..all done!")

    else:
        data = np.concatenate((meanbands, varbands, ratios, regionprops_calc, meantexture, meanbrightness), axis=1)
        if args.att == 'rgb':
            print("grvi, data")
            data2 = np.column_stack((index.astype(int), regionslist, mean_grvi, data))
        else:
            print("ndvi, grvi, data")
            data2 = np.column_stack((index.astype(int), regionslist, mean_ndvi, mean_grvi, data))
        print("Data collected and attributes calculated!")
        
    return(data2)


def replace(arr, rep_dict):
    """
    REPLACES VALUES IN nD NUMPY ARRAY IN A VECTORIZED METHOD

    arr = input array to be changed
    rep_dict = dictionarry containing org values and new values
    
    output = nD array with replaced values
        
    """
    # Removing the explicit "list"
    rep_keys, rep_vals = np.array(list(zip(*sorted(rep_dict.items()))))
    idces = np.digitize(arr, rep_keys, right=True)

    return rep_vals[idces]
    
    
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
    tmp[a[a != b], b[a != b]] = True

    print("Checking horizontal adjacency..")
    # check the horizontal adjacency
    a, b = label_img[:, :-1], label_img[:, 1:]
    tmp[a[a != b], b[a != b]] = True

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
    if args.att == 'rgb':  # Only RGB no height
        print("Using Greeennes, texture and vissible brightness")
        meancalc = [2, 20, 21]  # only use greennes

        print("Calculating avarage attributes for per neighbour:")
        for neighbours in tqdm.tqdm(resultarr):
            wantedneigh = rat[np.logical_or.reduce([rat[:, 0] == x for x in neighbours])]
            for i in meancalc:
                result.append(np.mean(wantedneigh[0][i]))
                
        # Collect results in array and reshape to attribute table format
        meanvalues = np.asarray(result)
        meanvalresh = meanvalues.reshape(len(regionslist), 3)

        # Calculate differences in percentage
        greennesdiff = ((meanvalresh[:, 0] - rat[:, 2]) / ((meanvalresh[:, 0] + rat[:, 2]) / 2)) * 100
        texturediff = ((meanvalresh[:, 1] - rat[:, 20]) / ((meanvalresh[:, 1] + rat[:, 20]) / 2)) * 100
        brightnessdiff = ((meanvalresh[:, 2] - rat[:, 21]) / ((meanvalresh[:, 2] + rat[:, 21]) / 2)) * 100

        diffattributes = np.vstack((greennesdiff, texturediff, brightnessdiff))
        allatributes = np.concatenate((rat, diffattributes), axis=1)
    
        print("Avarage attributes for neighbours calculated.. all done!")
        
    elif args.att == 'rgbheight': # RGB and height
        print("Using Greennes, height, texture and vissible brightness")
        meancalc = [2, 9, 20, 21]  # Greennes and height

        print("Calculating avarage attributes for per neighbour:")
        for neighbours in tqdm.tqdm(resultarr):
            wantedneigh = rat[np.logical_or.reduce([rat[:, 0] == x for x in neighbours])]
            for i in meancalc:
                result.append(np.mean(wantedneigh[0][i]))
                
        # Collect results in array and reshape to attribute table format
        meanvalues = np.asarray(result)
        meanvalresh = meanvalues.reshape(len(regionslist), 4)

        # Calculate differences in percentage
        greennesdiff = ((meanvalresh[:, 0] - rat[:, 2]) / ((meanvalresh[:, 0] + rat[:, 2]) / 2)) * 100
        heightdiff = ((meanvalresh[:, 1] - rat[:, 9]) / ((meanvalresh[:, 1] + rat[:, 9]) / 2)) * 100
        texturediff = ((meanvalresh[:, 2] - rat[:, 20]) / ((meanvalresh[:, 2] + rat[:, 20]) / 2)) * 100
        brightnessdiff = ((meanvalresh[:, 3] - rat[:, 21]) / ((meanvalresh[:, 3] + rat[:, 21]) / 2)) * 100

        diffattributes = np.vstack((greennesdiff, heightdiff, texturediff, brightnessdiff)).T
        allatributes = np.concatenate((rat, diffattributes), axis=1)
    
        print("Avarage attributes for neighbours calculated.. all done!")
        
    elif args.att == 'rgbnir':  # RGB and nir
        print("Using NDVI, Greennes, texture and vissible brightness")
        meancalc = [2, 3, 20, 21]  # NDVI and Greennes

        print("Calculating avarage attributes for per neighbour:")
        for neighbours in tqdm.tqdm(resultarr):
            wantedneigh = rat[np.logical_or.reduce([rat[:, 0] == x for x in neighbours])]
            for i in meancalc:
                result.append(np.mean(wantedneigh[0][i]))
                
        # Collect results in array and reshape to attribute table format
        meanvalues = np.asarray(result)
        meanvalresh = meanvalues.reshape(len(regionslist), 4)

        # Calculate differences in percentage
        ndvidiff = ((meanvalresh[:, 0] - rat[:, 2]) / ((meanvalresh[:, 0] + rat[:, 2]) / 2)) * 100
        greennesdiff = ((meanvalresh[:, 1] - rat[:, 3]) / ((meanvalresh[:, 1] + rat[:, 3]) / 2)) * 100
        texturediff = ((meanvalresh[:, 2] - rat[:, 20]) / ((meanvalresh[:, 2] + rat[:, 20]) / 2)) * 100
        brightnessdiff = ((meanvalresh[:, 3] - rat[:, 21]) / ((meanvalresh[:, 3] + rat[:, 21]) / 2)) * 100

        diffattributes = np.vstack((ndvidiff, greennesdiff, texturediff, brightnessdiff)).T
        allatributes = np.concatenate((rat, diffattributes), axis=1)
    
        print("Avarage attributes for neighbours calculated!")
        
    else:
        print("Using NDVI, Greennes, DSM, texture and vissible brightness")
        meancalc = [2, 3, 9, 20, 21]  # use ndvi greennes and dsm

        print("Calculating avarage attributes for per neighbour:")
        for neighbours in tqdm.tqdm(resultarr):
            wantedneigh = rat[np.logical_or.reduce([rat[:, 0] == x for x in neighbours])]
            for i in meancalc:
                result.append(np.mean(wantedneigh[0][i]))
                
        # Collect results in array and reshape to attribute table format
        meanvalues = np.asarray(result)
        meanvalresh = meanvalues.reshape(len(regionslist), 5)

        # Calculate differences in percentage
        ndvidiff = ((meanvalresh[:, 0] - rat[:, 2]) / ((meanvalresh[:, 0] + rat[:, 2]) / 2)) * 100
        greennesdiff = ((meanvalresh[:, 1] - rat[:, 2]) / ((meanvalresh[:, 1] + rat[:, 2]) / 2)) * 100
        heightdiff = ((meanvalresh[:, 2] - rat[:, 9]) / ((meanvalresh[:, 2] + rat[:, 9]) / 2)) * 100
        texturediff = ((meanvalresh[:, 3] - rat[:, 20]) / ((meanvalresh[:, 3] + rat[:, 20]) / 2)) * 100
        brightnessdiff = ((meanvalresh[:, 4] - rat[:, 21]) / ((meanvalresh[:, 4] + rat[:, 21]) / 2)) * 100

        diffattributes = np.vstack((ndvidiff, greennesdiff, heightdiff, texturediff, brightnessdiff)).T
        allatributes = np.concatenate((rat, diffattributes), axis=1)
    
        print("Avarage attributes for neighbours calculated!")
        
    return allatributes, resultarr

   
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
    outBand.WriteArray(dataset, 0, 0)
    outBand.SetNoDataValue(float('NaN'))

    # set the projection and extent information of the dataset
    outDataSet.SetProjection(dataSource.GetProjection())
    outDataSet.SetGeoTransform(dataSource.GetGeoTransform())

    # Flush to save
    outBand.FlushCache()
    outDataSet.FlushCache()


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
    values = classes  #0 for non veg, 1 for veg
    dictionary = dict(zip(keys, values))
    classified = replace(org_label, dictionary)
    
    # Label merged regions with unique ID
    labeld = label(classified)
    number_of_labels = np.unique(labeld)
    newvals = np.asarray(list(range(1, (len(number_of_labels) + 1))))
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
    
# ---------------- #
# START OF SCRIPT 
# ---------------- #
 
# Define regions of org label image
print("Creating list of new regions and re-label segmented image..")

regionslist = np.unique(seg)  # UPDATE REGIONSLIST
# Construct list of new values from 1 to n
newvals = np.asarray(list(range(1, (len(regionslist) + 1))))
# Convert to dictionary to increase processing speed
keys = regionslist
values = newvals
dictionary = dict(zip(keys, values))
# Apply replace function to relabel image in a vectorized way
seg_rel = replace(seg, dictionary)

# Remove org seg
print("Deleting old segmented image..")
del seg, keys, values, dictionary

# CALCULATE RASTER ATTRIBUTES TO NUMPY ARRAY
print("Starting calc region attributes function")
attributes1 = calc_region_attributes(org_img, seg_rel)

# Delete garbage from memory
print("Deleting garbage from memory")

if not args.dsm:
    pass
else:
    del DSM, DTM, dsm, dtm

gc.collect()  # remove garbage and continue to next image

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

# ------------------------- #
# START OF CLASSIFICATION
# ------------------------- #

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
# 16 (19) = perimeter, 17 (20) = area, 18 (21) = extent, 19 (22) = eccentricity, 20 (23) = roughness,
# 21 (24) = vissible brightness

# Neighbour props:
#  = distance to closest neighbour, 22 = NDVIdiff, 23 = Greennesdiff, 24 = Texturediff, 25 = Brightnessdiff
# Convert to dictionary to increase processing speed

# ---------------------- #
print("Implementing classification rules..")

# -------------- #
#   BIG ROADS    #
# -------------- #

bigroad = (tile_attributes[:, 3] < 71) & (tile_attributes[:, 17] >= 1500) & (tile_attributes[:, 4] >= 139) | \
          (tile_attributes[:, 3] > 71) & (tile_attributes[:, 23] <= 7.4) & (tile_attributes[:, 9] < 29) & \
          (tile_attributes[:, 2] < 119)

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
regionslist = np.unique(bigroad_ID)  # FIRST UPDATE REGIONSLIST!
bigroad_tile_attributes = calc_region_attributes(org_img, bigroad_ID)
# Merge attributes with classified values and neighbours
bigroad_tile_attributes = np.column_stack((bigroad_tile_attributes, bigroad_table))

# Reclassify bigroads to new region attributes (less roughness and other band ratios)
bigroad_new = (bigroad_tile_attributes[:, -1] == 2) & (bigroad_tile_attributes[:, 8] >= 500) & \
              (bigroad_tile_attributes[:, 20] < 7) & (bigroad_tile_attributes[:, 17] >= 1200) | \
              (bigroad_tile_attributes[:, -1] == 2) & (bigroad_tile_attributes[:, 9] <= 310) & \
              (bigroad_tile_attributes[:, 20] < 7) & (bigroad_tile_attributes[:, 17] >= 1200)

classes_bigroad_new = np.zeros((len(bigroad_tile_attributes), 1)).astype(int)
classes_bigroad_new[bigroad_new == True] = 2
classes_bigroad_new[bigroad_new != True] = 1

bigroad_new, bigroad_ID_new = class_and_replace(bigroad_ID, classes_bigroad_new)

# Save roadmask
filename = inputpath
output = (args.output+'/'+imagenamenoext[0]+'_roadmask.tif')
project_raster(filename, output, bigroad_new)


# ----------------------- #
# ALL VEGETATION & MASKS  #
# ----------------------- #

# Classify all vegetation based on small segments
allvegc = (tile_attributes[:, 3] < 71) & (tile_attributes[:, 17] <= 1500) & (tile_attributes[:, 2] > 110) | \
         (tile_attributes[:, 3] > 71) & (tile_attributes[:, 23] <= 7.4) & (tile_attributes[:, 9] < 29) & \
         (tile_attributes[:, 2] > 119) | \
         (tile_attributes[:, 3] > 71) & (tile_attributes[:, 23] <= 7.4) & (tile_attributes[:, 9] > 29)
         
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
regionslist = np.unique(allveg_ID)  # FIRST UPDATE REGIONSLIST!
allveg_tile_attributes = calc_region_attributes(org_img, allveg_ID)
# Merge attributes with classified values and neighbours
allveg_tile_attributes = np.column_stack((allveg_tile_attributes, allveg_table))

# Classify vegetation that is located on road
roadvegc = (allveg_tile_attributes[:, -1] == 2) & (allveg_tile_attributes[:, 22] >= 90) & \
           (allveg_tile_attributes[:, 17] <= 2500)

# Create mask for vegetation on road
classes_roadveg = np.zeros((len(allveg_tile_attributes), 1)).astype(int)
classes_roadveg[roadvegc == True] = 2
classes_roadveg[roadvegc != True] = 1

roadvegmask, roadvegmask_ID = class_and_replace(allveg_ID, classes_roadveg)

# ----------------- #
#   UNHEALTHY VEG   #
# ----------------- #

# Where vegetation = true, filter out unhealthy veg

unhealthyvegc = (allvegc == True) & (tile_attributes[:, 2] <= 116) & (tile_attributes[:, 3] <= 81.5) & \
                (tile_attributes[:, 5] <= 250)

# Replace seg_rel
classes_unhveg = np.zeros((len(tile_attributes), 1)).astype(int)
classes_unhveg[unhealthyvegc == True] = 2
classes_unhveg[unhealthyvegc != True] = 1

# Replace and merge
unhealthyveg, unhealthyveg_ID = class_and_replace(seg_rel, classes_unhveg)

# ---------------------------- #
# CONSTRUCT CLASSIFIED IMAGES  #
# ---------------------------- #

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

# ------------ #
# DETECT GAPS  #
# ------------ #

# Classify unclassified that is surrounded by vegetation
gapc = (allveg_tile_attributes[:, -1] == 1) & (allveg_tile_attributes[:, 23] >= 90) & \
       (allveg_tile_attributes[:, 17] <= 7500)

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
