Framework to segment and classify images

Start_chain.bat
---------------
dir = set dir of images to be processed
scriptdir = set correct path to OBIA_chain.py
batchdir = set correct path to LSMSSegmentation_chain.bat

-pre = yes, no. Start pre-processing (LSMMSsegmentation step) or skip to attributes calculation
-att = rgb, rgbnir, rgbheight, rgbnirheight. Set if images have NIR band and if height is used. IF height is used:
-dsm = path to dsm
-dtm = path to dtm

LSMSSegmentation_chain.bat
--------------------------
appfolder = set correct path to OTBdlls folder
tilesize = set the tilesize
spatialr = spatial range
ranger = radius range

OBIA_chain.py
-------------
For classification rules, see line 636


Good settings:

On avarage processing takes about 2 hours per tile on 'The Beast'. 

General:
-tilesize (bigger tiles = faster, but influences results when there is shadow (less is better))

Meanshiftsmoothing: 
-spatialr 10 (below 10 = less smoothing, faster) (above 10 = more smoothing, slower)
-maxiter 10 (above 10 is more itterations = more smoothing, slower)
-modesearch 0 (do not change, otherwise tile merging not possible)

Segmentation:
-ranger 7.5 (above 7.5 = less segments, sometimes misses parts) (below 7.5 = more segments, slower)
-spatialr (not sure yet)
-minsize 0 (otherwise segments are removed)

SmallRegionsMerging:
-minsize 300 (above 300, slow but less segments) (below 300, fast but a lot of segments)