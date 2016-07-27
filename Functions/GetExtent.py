# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:54:05 2016

@author: darellvdv
"""

def GetExtent(gt,cols,rows):
        ''' Return list of corner coordinates from a geotransform

            type gt:   C{tuple/list}
            param gt: geotransform
            type cols:   C{int}
            param cols: number of columns in the dataset
            type rows:   C{int}
            param rows: number of rows in the dataset
            rtype:    C{[float,...,float]}
            return:   coordinates of each corner
            '''
        ext=[]
        xarr=[0,cols]
        yarr=[0,rows]
            
        for px in xarr:
            for py in yarr:
                x=gt[0]+(px*gt[1])+(py*gt[2])
                y=gt[3]+(px*gt[4])+(py*gt[5])
                ext.append([x,y])
                print x,y
            yarr.reverse()
            
        return ext