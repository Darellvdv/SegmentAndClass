# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:54:05 2016

@author: darellvdv
"""

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