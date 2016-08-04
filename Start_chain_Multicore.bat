:: Multithreaded Segmentation and Classification framework
:: 
:: For help see: 10.0.03/EagleView_KB/...
:: Or readme.txt in the application folder


:: Declare variables
:: * = required

@echo off
cls

::------------------::
:: FOLDER STRUCTURE ::
::------------------::

:: Input folder*:
SET indir="D:\Othertrees"

:: Output folder* (will be created if not existing):
SET outdir="D:\Othertrees\output"

:: Path to DSM and DTM (optional)
SET dsm=
SET dtm=

::----------::
:: SETTINGS ::
::----------::

:: Input file format* (rgb, rgbnir, rgbheight, rgbnirheight):
SET att="rgb"

:: Pre-processing* (yes/no):
SET pre="yes"

:: Euclidean spectral range*:
SET rang="5"
:: Connected component spatial range*:
SET spatialr="5"
:: Minimal segment size*:
SET minsiz="300"
:: Tilesize for processing*:
SET tilesize="500"

:: Amount of cores for processing*:
SET cores=1

::--------------::
:: SCRIPT PATHS ::
::--------------::

:: Multicore R script*:
SET Rscript="D:/pinapple/SegmentAndClass/Multicore.R"
:: OBIA Python script#:
SET pythonscript="D:/pinapple/SegmentAndClass/OBIA_chain_v3.py"
:: OTB segmentation batch script*:
SET batchdir="D:/pinapple/SegmentAndClass/LSMSSegmentation_chain.bat"


:: Start Segementation framework:
IF [%dsm%]==[] (
   call Rscript %Rscript% %indir% %outdir% %pythonscript% %batchdir% %att% %pre% %rang% %spatialr% %minsiz% %tilesize% %cores% 
 ) ELSE (
   call Rscript %Rscript% %indir% %outdir% %pythonscript% %batchdir% %att% %pre% %rang% %spatialr% %minsiz% %tilesize% %cores% %dsm% %dtm%
 )
::						    1		  2 		 3			   4		5	  6     7       8         9         10        11	 12    13
PAUSE
