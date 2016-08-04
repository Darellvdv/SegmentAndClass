Multithreaded framework to segment and classify images

Start_chain_Multicore.bat
---------------
Rscript = path to Multicore.R
indir = path to images folder
outdir = path to output folder (will be created if non-existend)
pythonscript = path to OBIA_chain.py
batchdir = path to LSMSSegmentation_chain.bat


Multicore.R
------------
-corenr: set the amount of cores

At the system call Python line:
-pre = yes, no. Start pre-processing (LSMMSsegmentation step) or skip to attributes calculation
-att = rgb, rgbnir, rgbheight, rgbnirheight. Set if images have NIR band and if height is used. IF height is used:
-dsm = path to dsm
-dtm = path to dtm


LSMSSegmentation_chain.bat
--------------------------
appfolder = set correct path to OTBdlls folder
tilesize = set the tilesize


Segmentation settings
---------------------
Processing time heavily depends on the settings and thus the amount of segments. Most of the proccessing time 
is taken up by filtering (smoothing) and small segment merging (+80% of the time). Setting lower smoothing and
segment merging thresholds improves processing time, but generates more (unwanted) segments.  

General:
-tilesize: Bigger tiles = faster, but influences results when there is shadow. A tilesize of 500 is best and will generate
	   20 x 20 tiles for each image that is 10000 x 10000.

-Ranger: best settings are around 2% of the spectral range of the input image. For a range of 0-255 this would mean
	 a ranger setting around 5. Lower values means more segments.

-Spatial: The connected component parameter is the spatial range in pixel values. For a dataset with a pixel size of
	  4.5cm a range from 5 is best.

-Maxiter: Times the meanshift smoothing can itterate over a region before it reaches its threshold. Higher levels means
	  smoother results, but heavily increase proccesing time.

-Thresh:  Threshold the meanshift smoothing can reach before it stops. Increasing this threshold heavily increases 
	  processing time

-Minsize: The minimum size for segments in pixel values. In the segmentation step, this minsize is quick and will remove
	  all segments below the threshold. In the SmallRegionsMerging step, this paramater merges regions below the threshold
	  to surrounding segments based on spectral characteristics. This parameter is the biggest infleunce of proccessing 
	  time and can take up to 75% of the total processing time. Settings above 300 will take large amount of time.


OBIA_chain.py
-------------
For classification rules, see line 636 
 