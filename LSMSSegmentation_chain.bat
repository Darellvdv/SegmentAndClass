@echo off

SET appfolder="D:\pinapple\SegmentAndClass\OTBdlls\bin"
SET output=%1
SET img=%2
SET imgname=%3
SET ranger=%4
SET spatialr=%5
SET minsiz=%6
SET tilesize=%7

:: SET ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 

call %appfolder%\otbcli_MeanShiftSmoothing -in %img% -fout %output%\%imgname%_MeanShift_Filter_range.tif -ranger %ranger% -spatialr %spatialr% -maxiter 10 -thres 0.1 -modesearch 0
call %appfolder%\otbcli_LSMSSegmentation -in %output%\%imgname%_MeanShift_Filter_range.tif -out %output%\%imgname%_segmentation.tif uint32 -tmpdir %output% -ranger %ranger% -spatialr %spatialr% -minsize 0 -tilesizex %tilesize% -tilesizey %tilesize%
call %appfolder%\otbcli_LSMSSmallRegionsMerging -in %output%\%imgname%_MeanShift_Filter_range.tif -inseg %output%\%imgname%_segmentation.tif -out %output%\%imgname%_segmentation_merged.tif uint32 -minsize %minsiz% -tilesizex %tilesize% -tilesizey %tilesize%
call %appfolder%\otbcli_LSMSVectorization -in %img% -inseg %output%\%imgname%_segmentation_merged.tif -out %output%\%imgname%_segmentation_merged.shp -tilesizex %tilesize% -tilesizey %tilesize%





