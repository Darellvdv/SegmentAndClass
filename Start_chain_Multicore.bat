@echo off

SET "Rscript=D:/pinapple/SegmentAndClass/Multicore.R"

SET "indir=D:/pinapple/smalltest"
SET "outdir=D:/pinapple/smalltest/Output"

SET "pythonscript=D:/pinapple/SegmentAndClassPin/Main.py"
SET "batchdir=D:/pinapple/SegmentAndClassPin/LSMSSegmentation_chain.bat"

call Rscript %Rscript% %indir% %outdir% %pythonscript% %batchdir%

PAUSE
